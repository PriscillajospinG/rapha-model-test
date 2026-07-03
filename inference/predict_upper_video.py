"""
inference/predict_upper_video.py
───────────────────────────────────────────────────────────────────────────
End-to-end upper-limb inference pipeline (MediaPipe Tasks API >= 0.10):

    .mp4  →  MediaPipe PoseLandmarker  →  Extract joints 11-16, 23-24
          →  Tensor (4, 300, 8, 1)
          →  CTR-GCN  (best_upper_limb_ctrgcn.pth)
          →  Predicted exercise  +  Confidence  +  All class probabilities

Usage:
    python inference/predict_upper_video.py --video path/to/exercise.mp4

    # custom model checkpoint
    python inference/predict_upper_video.py --video path/to/exercise.mp4 \\
                             --model models/best_upper_limb_ctrgcn.pth

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
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# ── Local modules ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.upper_loader import (
    EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M,
    load_class_map,
)
from graph.upper_limb import UpperLimbGraph
from model.ctrgcn import Model

# ── MediaPipe landmark indices → upper-limb graph nodes ───────────────────────
MP_JOINT_IDS = [11, 12, 13, 14, 15, 16, 23, 24]   # 8 upper-limb joints

# ── MediaPipe Pose settings ────────────────────────────────────────────────────
MP_MODEL_COMPLEXITY   = 1
MP_MIN_DETECT_CONF    = 0.5
MP_MIN_TRACK_CONF     = 0.5

# ── Default paths ──────────────────────────────────────────────────────────────
_BASE_DIR     = Path(__file__).parent.parent
DEFAULT_MODEL = _BASE_DIR / "models" / "best_upper_limb_ctrgcn.pth"
CLASS_MAP_CSV = _BASE_DIR / "processed_dataset_upper" / "upper_class_map.csv"
MODEL_TASK    = _BASE_DIR / "pose_landmarker_full.task"


# ─────────────────────────────────────────────────────────────────────────────
#  Frame extraction  (video → per-frame joint data)
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str) -> np.ndarray:
    """
    Run MediaPipe PoseLandmarker (Tasks API) on every frame of the video.

    Returns:
        frames_data : (T_raw, V, C)  float32
            T_raw = number of frames in the video
            V     = 8 upper-limb joints (nodes 0-7)
            C     = 4 (x, y, z, visibility)
    """
    if not MODEL_TASK.exists():
        raise FileNotFoundError(
            f"PoseLandmarker model not found: {MODEL_TASK}\n"
            "Download with:\n"
            "  curl -o pose_landmarker_full.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_val      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"  Video : {Path(video_path).name}")
    print(f"  Frames: {total_frames}  |  FPS: {fps_val:.1f}")

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_TASK))
    options = mp_vision.PoseLandmarkerOptions(
        base_options                  = base_options,
        running_mode                  = RunningMode.VIDEO,
        num_poses                     = 1,
        min_pose_detection_confidence = MP_MIN_DETECT_CONF,
        min_pose_presence_confidence  = MP_MIN_DETECT_CONF,
        min_tracking_confidence       = MP_MIN_TRACK_CONF,
    )

    all_frames:   list[np.ndarray] = []
    failed_frames = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            rgb       = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms     = int(frame_idx * 1000 / fps_val)
            result    = landmarker.detect_for_video(mp_image, ts_ms)

            frame_joints = np.zeros((EXPECTED_V, EXPECTED_C), dtype=np.float32)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lms = result.pose_landmarks[0]
                for node_idx, mp_id in enumerate(MP_JOINT_IDS):
                    lm = lms[mp_id]
                    frame_joints[node_idx] = [lm.x, lm.y, lm.z, lm.visibility]
            else:
                failed_frames += 1

            all_frames.append(frame_joints)
            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"    processed {frame_idx}/{total_frames} frames …\r",
                      end="", flush=True)

    cap.release()
    print()

    print(f"  Frames extracted : {len(all_frames)}")
    print(f"  Failed detections: {failed_frames} "
          f"({failed_frames / max(len(all_frames), 1) * 100:.1f}%)")

    return np.stack(all_frames, axis=0)  # (T_raw, V, C)


# ─────────────────────────────────────────────────────────────────────────────
#  Tensor construction  (T_raw, V, C) → (C, T, V, M)
# ─────────────────────────────────────────────────────────────────────────────

def build_tensor(frames: np.ndarray) -> torch.Tensor:
    """
    Resample / pad the raw frame array to exactly T=300 frames and
    reshape to the expected CTR-GCN input format (C, T, V, M).

    Returns:
        tensor : (C=4, T=300, V=8, M=1)  torch float32
    """
    T_raw = frames.shape[0]
    T_out = EXPECTED_T

    if T_raw == 0:
        raise ValueError("No frames were extracted from the video.")

    if T_raw >= T_out:
        indices   = np.linspace(0, T_raw - 1, T_out, dtype=int)
        resampled = frames[indices]
    else:
        repeats   = -(-T_out // T_raw)
        tiled     = np.tile(frames, (repeats, 1, 1))
        resampled = tiled[:T_out]

    # (T, V, C) → (C, T, V) → (C, T, V, M=1)
    tensor = np.transpose(resampled, (2, 0, 1))
    tensor = tensor[:, :, :, np.newaxis]

    assert tensor.shape == (EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M), \
        f"Built tensor shape mismatch: {tensor.shape}"

    return torch.from_numpy(tensor).float()


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> tuple[Model, dict[int, str]]:
    """Load the saved upper-limb CTR-GCN checkpoint.

    Returns:
        (model, class_names_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_class   = ckpt.get("num_class",   4)
    num_point   = ckpt.get("num_point",   8)
    num_person  = ckpt.get("num_person",  1)
    in_channels = ckpt.get("in_channels", 4)
    class_names = ckpt.get("class_names", {})

    if not class_names:
        class_names = load_class_map(str(CLASS_MAP_CSV))

    graph = UpperLimbGraph()
    model = Model(
        num_class   = num_class,
        num_point   = num_point,
        num_person  = num_person,
        in_channels = in_channels,
        graph       = graph,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch   = ckpt.get("epoch",   "?")
    val_acc = ckpt.get("val_acc", float("nan"))
    print(f"  Checkpoint epoch : {epoch}")
    print(f"  Val acc at save  : {val_acc * 100:.2f}%")
    print(f"  Classes          : {class_names}")
    return model, class_names


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model:       Model,
    tensor:      torch.Tensor,
    device:      torch.device,
    class_names: dict[int, str],
) -> dict:
    """Run CTR-GCN forward pass. Returns prediction dict."""
    x = tensor.unsqueeze(0).to(device)   # (1, C, T, V, M)

    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1)[0]

    top_idx  = probs.argmax().item()
    top_prob = probs[top_idx].item()

    all_probs = {
        class_names.get(i, f"class_{i}"): round(probs[i].item() * 100, 2)
        for i in range(len(probs))
    }

    return {
        "predicted_class": class_names.get(top_idx, f"class_{top_idx}"),
        "predicted_idx":   top_idx,
        "confidence":      round(top_prob * 100, 2),
        "probabilities":   all_probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CTR-GCN upper-limb inference on a single exercise video."
    )
    p.add_argument("--video",  required=True,
                   help="Path to input video file (.mp4 / .avi / .mov …)")
    p.add_argument("--model",  default=str(DEFAULT_MODEL),
                   help=f"Path to CTR-GCN checkpoint (default: {DEFAULT_MODEL})")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    bar = "─" * 58
    print(f"\n{bar}")
    print("  CTR-GCN Upper-Limb Inference")
    print(bar)
    print(f"  Device     : {device}")
    print(f"  Model ckpt : {args.model}")
    print(f"  Video      : {args.video}\n")

    if not Path(args.video).exists():
        print(f"[ERROR] Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"[ERROR] Checkpoint not found: {args.model}", file=sys.stderr)
        print("  Train first:  python training/train_upper_limb_ctrgcn.py",
              file=sys.stderr)
        sys.exit(1)

    # ── Step 1 ────────────────────────────────────────────────────────────────
    print("[1/4] Running MediaPipe PoseLandmarker …")
    t0     = time.perf_counter()
    frames = extract_frames(args.video)
    print(f"      Done in {time.perf_counter()-t0:.1f}s\n")

    # ── Step 2 ────────────────────────────────────────────────────────────────
    print("[2/4] Building input tensor …")
    tensor = build_tensor(frames)
    print(f"      Tensor shape : {tuple(tensor.shape)}")
    print(f"      (C={tensor.shape[0]}, T={tensor.shape[1]}, "
          f"V={tensor.shape[2]}, M={tensor.shape[3]})\n")

    # ── Step 3 ────────────────────────────────────────────────────────────────
    print("[3/4] Loading CTR-GCN checkpoint …")
    model, class_names = load_model(args.model, device)
    print()

    # ── Step 4 ────────────────────────────────────────────────────────────────
    print("[4/4] Running inference …")
    t0     = time.perf_counter()
    result = predict(model, tensor, device, class_names)
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
        print(f"    {rank:2d}. {cls_name:22s}  {prob:6.2f}%  {bar_fill}{marker}")

    print(bar)


if __name__ == "__main__":
    main()
