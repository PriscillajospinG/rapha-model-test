"""
inference/predict_face_video.py
───────────────────────────────────────────────────────────────────────────
End-to-end facial rehabilitation inference pipeline (MediaPipe FaceMesh):

    .mp4  →  MediaPipe FaceMesh  →  Extract 33 landmarks (x, y, z)
          →  Tensor (3, 300, 33, 1)
          →  CTR-GCN  (best_face_ctrgcn.pth)
          →  Predicted exercise  +  Confidence  +  All class probabilities

Usage:
    python inference/predict_face_video.py --video path/to/exercise.mp4

    # Custom checkpoint
    python inference/predict_face_video.py --video path/to/exercise.mp4 \\
                             --model models/best_face_ctrgcn.pth

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
from dataset.face_loader import (
    EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M,
    load_face_class_map,
)
from graph.face_graph import FaceGraph
from graph.face_landmark_mapping import FACE_LANDMARK_IDS, NUM_FACE_NODES
from model.ctrgcn import Model

# ── Default paths ──────────────────────────────────────────────────────────────
_BASE_DIR     = Path(__file__).parent.parent
DEFAULT_MODEL = _BASE_DIR / "models" / "best_face_ctrgcn.pth"
CLASS_MAP_CSV = _BASE_DIR / "datasets/face" / "face_class_map.csv"


# ─────────────────────────────────────────────────────────────────────────────
#  Frame extraction  (video → per-frame landmark data)
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str) -> np.ndarray:
    """
    Run MediaPipe FaceMesh on every frame of the video.

    Returns:
        frames_data : (T_raw, V=33, C=3)  float32
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_val      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"  Video : {Path(video_path).name}")
    print(f"  Frames: {total_frames}  |  FPS: {fps_val:.1f}")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode        = False,
        max_num_faces            = 1,
        refine_landmarks         = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )

    all_frames:    list[np.ndarray] = []
    failed_frames  = 0
    frame_idx      = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        # (V, C) array for this frame
        frame_lms = np.zeros((EXPECTED_V, EXPECTED_C), dtype=np.float32)

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            face_lm_list = results.multi_face_landmarks[0].landmark
            for node_idx, mp_id in enumerate(FACE_LANDMARK_IDS):
                lm = face_lm_list[mp_id]
                frame_lms[node_idx] = [lm.x, lm.y, lm.z]
        else:
            failed_frames += 1

        all_frames.append(frame_lms)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"    processed {frame_idx}/{total_frames} frames …\r",
                  end="", flush=True)

    cap.release()
    face_mesh.close()
    print()

    print(f"  Frames extracted : {len(all_frames)}")
    print(f"  Failed detections: {failed_frames} "
          f"({failed_frames / max(len(all_frames), 1) * 100:.1f}%)")

    return np.stack(all_frames, axis=0)   # (T_raw, V, C)


# ─────────────────────────────────────────────────────────────────────────────
#  Tensor construction  (T_raw, V, C) → (C, T, V, M)
# ─────────────────────────────────────────────────────────────────────────────

def build_tensor(frames: np.ndarray) -> torch.Tensor:
    """
    Resample / pad raw frame array to exactly T=300 frames and reshape
    to CTR-GCN input format (C, T, V, M).

    Returns:
        tensor : (C=3, T=300, V=33, M=1)  torch float32
    """
    T_raw = frames.shape[0]
    T_out = EXPECTED_T

    if T_raw == 0:
        raise ValueError("No frames extracted from the video.")

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

    expected = (EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M)
    assert tensor.shape == expected, \
        f"Built tensor shape mismatch: {tensor.shape} ≠ {expected}"

    return torch.from_numpy(tensor).float()


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> tuple[Model, dict[int, str]]:
    """Load the saved face CTR-GCN checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_class   = ckpt.get("num_class",   6)
    num_point   = ckpt.get("num_point",   33)
    num_person  = ckpt.get("num_person",  1)
    in_channels = ckpt.get("in_channels", 3)
    class_names = ckpt.get("class_names", {})

    if not class_names:
        class_names = load_face_class_map(str(CLASS_MAP_CSV))

    graph = FaceGraph()
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
        description="CTR-GCN facial rehabilitation inference on a single exercise video."
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

    bar = "─" * 60
    print(f"\n{bar}")
    print("  CTR-GCN Face Rehabilitation Inference")
    print(bar)
    print(f"  Device     : {device}")
    print(f"  Model ckpt : {args.model}")
    print(f"  Video      : {args.video}\n")

    if not Path(args.video).exists():
        print(f"[ERROR] Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"[ERROR] Checkpoint not found: {args.model}", file=sys.stderr)
        print("  Train first:  python training/train_face_ctrgcn.py",
              file=sys.stderr)
        sys.exit(1)

    # ── Step 1 ────────────────────────────────────────────────────────────────
    print("[1/4] Running MediaPipe FaceMesh …")
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
    sorted_probs = sorted(
        result["probabilities"].items(), key=lambda kv: kv[1], reverse=True
    )
    for rank, (cls_name, prob) in enumerate(sorted_probs, 1):
        bar_fill = "█" * int(prob / 5)
        marker   = " ◄" if rank == 1 else ""
        print(f"    {rank:2d}. {cls_name:25s}  {prob:6.2f}%  {bar_fill}{marker}")

    print(bar)


if __name__ == "__main__":
    main()
