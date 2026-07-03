"""
predict_video.py
───────────────────────────────────────────────────────────────────────────
End-to-end inference pipeline:

    .mp4  →  MediaPipe Pose  →  Extract joints 23-32
          →  Tensor (4, 300, 10, 1)
          →  CTR-GCN  (best_lower_limb_ctrgcn.pth)
          →  Predicted exercise  +  Confidence  +  All class probabilities

Usage:
    python predict_video.py  --video  path/to/exercise.mp4

    # custom model checkpoint
    python predict_video.py  --video  path/to/exercise.mp4 \
                             --model  models/best_lower_limb_ctrgcn.pth

Requirements:
    pip install opencv-python mediapipe torch
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

# ── Local modules ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.loader import CLASS_NAMES, EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M
from graph.lower_limb import LowerLimbGraph
from model.ctrgcn import Model

# ── MediaPipe landmark indices we care about ───────────────────────────────────
MP_JOINT_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]    # 10 lower-limb joints

# ── MediaPipe Pose settings ────────────────────────────────────────────────────
MP_MODEL_COMPLEXITY   = 1
MP_MIN_DETECT_CONF    = 0.5
MP_MIN_TRACK_CONF     = 0.5

# ── Default checkpoint ─────────────────────────────────────────────────────────
DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "best_lower_limb_ctrgcn.pth"


# ─────────────────────────────────────────────────────────────────────────────
#  Frame extraction  (video → per-frame joint data)
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str) -> np.ndarray:
    """
    Run MediaPipe Pose on every frame of the video and return a skeleton array.

    Returns:
        frames_data : (T_raw, V, C)  float32
            T_raw = number of frames in the video
            V     = 10 lower-limb joints
            C     = 4 (x, y, z, visibility)

        If a frame has no detected pose, all values are set to 0.
    """
    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(
        static_image_mode        = False,
        model_complexity         = MP_MODEL_COMPLEXITY,
        smooth_landmarks         = True,
        enable_segmentation      = False,
        min_detection_confidence = MP_MIN_DETECT_CONF,
        min_tracking_confidence  = MP_MIN_TRACK_CONF,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"  Video : {Path(video_path).name}")
    print(f"  Frames: {total_frames}  |  FPS: {fps:.1f}")

    all_frames: list[np.ndarray] = []   # each entry: (V, C) = (10, 4)
    failed_frames = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = pose.process(rgb)

        frame_joints = np.zeros((EXPECTED_V, EXPECTED_C), dtype=np.float32)

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            for node_idx, mp_id in enumerate(MP_JOINT_IDS):
                lm = lms[mp_id]
                frame_joints[node_idx] = [lm.x, lm.y, lm.z, lm.visibility]
        else:
            failed_frames += 1

        all_frames.append(frame_joints)

        if len(all_frames) % 50 == 0:
            print(f"    processed {len(all_frames)}/{total_frames} frames …\r",
                  end="", flush=True)

    cap.release()
    pose.close()
    print()   # newline after \r

    print(f"  Frames extracted : {len(all_frames)}")
    print(f"  Failed detections: {failed_frames} "
          f"({failed_frames / max(len(all_frames), 1) * 100:.1f}%)")

    return np.stack(all_frames, axis=0)   # (T_raw, V, C)


# ─────────────────────────────────────────────────────────────────────────────
#  Tensor construction  (T_raw, V, C) → (C, T, V, M)
# ─────────────────────────────────────────────────────────────────────────────

def build_tensor(frames: np.ndarray) -> torch.Tensor:
    """
    Resample / pad the raw frame array to exactly T=300 frames and
    reshape to the expected CTR-GCN input format (C, T, V, M).

    Resampling strategy:
      • T_raw > T : uniformly sample T frames (no information loss via stride)
      • T_raw < T : tile (loop) the frames to fill T frames
      • T_raw == T: use as-is

    Args:
        frames : (T_raw, V, C)  numpy float32

    Returns:
        tensor : (C=4, T=300, V=10, M=1)  torch float32
    """
    T_raw = frames.shape[0]
    T_out = EXPECTED_T

    if T_raw == 0:
        raise ValueError("No frames were extracted from the video.")

    if T_raw >= T_out:
        # Uniform temporal sampling
        indices = np.linspace(0, T_raw - 1, T_out, dtype=int)
        resampled = frames[indices]                   # (T_out, V, C)
    else:
        # Tile (loop) the video to reach T_out frames
        repeats = -(-T_out // T_raw)                  # ceiling division
        tiled   = np.tile(frames, (repeats, 1, 1))    # (repeats*T_raw, V, C)
        resampled = tiled[:T_out]                     # (T_out, V, C)

    # (T, V, C) → (C, T, V) → (C, T, V, M=1)
    tensor = np.transpose(resampled, (2, 0, 1))       # (C, T, V)
    tensor = tensor[:, :, :, np.newaxis]              # (C, T, V, 1)

    assert tensor.shape == (EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M), \
        f"Built tensor shape mismatch: {tensor.shape}"

    return torch.from_numpy(tensor).float()


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> Model:
    """Load the saved CTR-GCN checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Restore hyper-parameters saved alongside the weights
    num_class   = ckpt.get("num_class",   9)
    num_point   = ckpt.get("num_point",  10)
    num_person  = ckpt.get("num_person",  1)
    in_channels = ckpt.get("in_channels", 4)

    graph = LowerLimbGraph()
    model = Model(
        num_class=num_class, num_point=num_point,
        num_person=num_person, in_channels=in_channels,
        graph=graph,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch   = ckpt.get("epoch",   "?")
    val_acc = ckpt.get("val_acc", float("nan"))
    print(f"  Checkpoint epoch : {epoch}")
    print(f"  Val acc at save  : {val_acc * 100:.2f}%")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(model: Model, tensor: torch.Tensor, device: torch.device) -> dict:
    """
    Run CTR-GCN forward pass.

    Args:
        tensor : (C, T, V, M)  — will be unsqueezed to (1, C, T, V, M)

    Returns dict with:
        predicted_class  : str   exercise name
        predicted_idx    : int   class index
        confidence       : float probability of the top class
        probabilities    : dict  {class_name: probability}  for all 9 classes
    """
    x = tensor.unsqueeze(0).to(device)        # (1, C, T, V, M)

    with torch.no_grad():
        logits = model(x)                     # (1, 9)
        probs  = F.softmax(logits, dim=1)[0]  # (9,)

    top_idx  = probs.argmax().item()
    top_prob = probs[top_idx].item()

    all_probs = {
        CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
        for i in range(len(probs))
    }

    return {
        "predicted_class": CLASS_NAMES[top_idx],
        "predicted_idx":   top_idx,
        "confidence":      round(top_prob * 100, 2),
        "probabilities":   all_probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CTR-GCN inference on a single exercise video."
    )
    p.add_argument(
        "--video", required=True,
        help="Path to input video file (.mp4 / .avi / .mov …)"
    )
    p.add_argument(
        "--model", default=str(DEFAULT_MODEL),
        help=f"Path to CTR-GCN checkpoint (default: {DEFAULT_MODEL})"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    bar = "─" * 55
    print(f"\n{bar}")
    print("  CTR-GCN Lower-Limb Inference")
    print(bar)
    print(f"  Device     : {device}")
    print(f"  Model ckpt : {args.model}")
    print(f"  Video      : {args.video}\n")

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not Path(args.video).exists():
        print(f"[ERROR] Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"[ERROR] Checkpoint not found: {args.model}", file=sys.stderr)
        print("  Train the model first:  python train_lower_limb_ctrgcn.py",
              file=sys.stderr)
        sys.exit(1)

    # ── Step 1 : Extract skeleton from video ─────────────────────────────────
    print("[1/4] Running MediaPipe Pose …")
    t0     = time.perf_counter()
    frames = extract_frames(args.video)
    print(f"      Done in {time.perf_counter()-t0:.1f}s\n")

    # ── Step 2 : Build input tensor ───────────────────────────────────────────
    print("[2/4] Building input tensor …")
    tensor = build_tensor(frames)
    print(f"      Tensor shape : {tuple(tensor.shape)}")
    print(f"      (C={tensor.shape[0]}, T={tensor.shape[1]}, "
          f"V={tensor.shape[2]}, M={tensor.shape[3]})\n")

    # ── Step 3 : Load model ───────────────────────────────────────────────────
    print("[3/4] Loading CTR-GCN checkpoint …")
    model = load_model(args.model, device)
    print()

    # ── Step 4 : Inference ────────────────────────────────────────────────────
    print("[4/4] Running inference …")
    t0     = time.perf_counter()
    result = predict(model, tensor, device)
    ms     = (time.perf_counter() - t0) * 1000

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"\n{bar}")
    print("  RESULT")
    print(bar)
    print(f"  Predicted exercise : {result['predicted_class'].upper()}")
    print(f"  Confidence         : {result['confidence']:.2f}%")
    print(f"  Inference time     : {ms:.1f} ms\n")

    print("  All class probabilities:")
    sorted_probs = sorted(result["probabilities"].items(),
                          key=lambda kv: kv[1], reverse=True)
    for rank, (cls_name, prob) in enumerate(sorted_probs, 1):
        bar_fill = "█" * int(prob / 5)
        marker   = " ◄" if rank == 1 else ""
        print(f"    {rank:2d}. {cls_name:12s}  {prob:6.2f}%  {bar_fill}{marker}")

    print(bar)


if __name__ == "__main__":
    main()
