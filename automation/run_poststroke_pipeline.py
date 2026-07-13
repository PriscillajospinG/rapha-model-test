"""
automation/run_poststroke_pipeline.py
=====================================
One-command automation for the Post-Stroke Rehabilitation CTR-GCN pipeline.

Pipeline steps:
    1. preprocessing/extract_poststroke_dataset.py   — MediaPipe Holistic extraction
    2. preprocessing/build_poststroke_ctrgcn_dataset.py — Build skeleton tensors
    3. preprocessing/split_poststroke_dataset.py        — Train / test split
    4. training/train_poststroke_ctrgcn.py           — Model training

Usage:
    python automation/run_poststroke_pipeline.py
    python automation/run_poststroke_pipeline.py --skip-extraction
    python automation/run_poststroke_pipeline.py --only-train
    python automation/run_poststroke_pipeline.py --skip-training

Flags:
    --skip-extraction   Skip steps 1–3 (assumes skeletons already built).
    --skip-training     Run extraction / build / split, then stop.
    --only-train        Jump straight to step 4 (skeletons must exist).
    --only-inference    Placeholder; inference not yet implemented here.

Logs saved to: logs/poststroke_pipeline.log
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable regardless of how this script is invoked
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from automation.utils import (
    StepResult,
    create_directory,
    format_time,
    log_message,
    print_error,
    print_info,
    print_pipeline_summary,
    print_step,
    print_success,
    print_warning,
    run_command,
    setup_logger,
    suggest_fix,
    validate_skeletons,
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PIPELINE_NAME = "Post-Stroke"

STEPS: list[dict] = [
    {
        "number": 1,
        "description": "Extracting Post-Stroke Dataset (MediaPipe Holistic)",
        "script": PROJECT_ROOT / "preprocessing" / "extract_poststroke_dataset.py",
    },
    {
        "number": 2,
        "description": "Building Post-Stroke CTR-GCN Skeleton Tensors",
        "script": PROJECT_ROOT / "preprocessing" / "build_poststroke_ctrgcn_dataset.py",
    },
    {
        "number": 3,
        "description": "Splitting Post-Stroke Dataset into Train / Test Sets",
        "script": PROJECT_ROOT / "preprocessing" / "split_poststroke_dataset.py",
    },
    {
        "number": 4,
        "description": "Training Post-Stroke CTR-GCN Model",
        "script": PROJECT_ROOT / "training" / "train_poststroke_ctrgcn.py",
    },
]

SKELETONS_DIR = PROJECT_ROOT / "datasets/post_stroke" / "skeletons"
LOG_FILE      = PROJECT_ROOT / "logs" / "poststroke_pipeline.log"
MODEL_PATH    = PROJECT_ROOT / "models" / "best_poststroke_ctrgcn.pth"
RESULTS_DIR   = PROJECT_ROOT / "results/lower_limb/post_stroke"


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_poststroke_pipeline.py",
        description="One-command automation for the Post-Stroke Rehabilitation CTR-GCN pipeline.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip steps 1-3 (dataset extraction and building). "
             "Assumes skeletons/*.npy files already exist.",
    )
    group.add_argument(
        "--skip-training",
        action="store_true",
        help="Run extraction / build / split only; skip model training.",
    )
    group.add_argument(
        "--only-train",
        action="store_true",
        help="Jump directly to the training step (skeletons must already exist).",
    )
    group.add_argument(
        "--only-inference",
        action="store_true",
        help="(Placeholder) Run inference only — not yet implemented.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Step runner
# ──────────────────────────────────────────────────────────────────────────────

def run_step(
    step: dict,
    total: int,
    logger,
) -> StepResult:
    """
    Execute a single pipeline step and return its result.
    """
    print_step(step["number"], total, step["description"])
    log_message(logger, "info", f"=== STEP {step['number']}/{total}: {step['description']} ===")

    script = Path(step["script"])
    if not script.exists():
        msg = f"Script not found: {script}"
        print_error(msg)
        log_message(logger, "error", msg)
        return StepResult(returncode=1, stdout="", stderr=msg, elapsed=0.0)

    result = run_command(
        cmd=[sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        logger=logger,
    )

    elapsed_str = format_time(result.elapsed)

    if result.success:
        print_success(f"Step {step['number']} completed in {elapsed_str}.")
        log_message(logger, "info", f"Step {step['number']} OK — elapsed: {elapsed_str}")
    else:
        print_error(f"Step {step['number']} FAILED  (exit code {result.returncode})")
        print_error(f"Script : {script}")
        fix = suggest_fix(result.combined_output)
        print_warning(f"Suggested Fix:\n  {fix}")
        log_message(logger, "error", f"Step {step['number']} FAILED — exit {result.returncode}")
        log_message(logger, "error", f"Suggested Fix: {fix}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    Entry point for the Post-Stroke pipeline.

    Returns:
        0 on success, 1 on any step failure.
    """
    args = parse_args()

    if args.only_inference:
        print_warning("--only-inference is not yet implemented for this pipeline.")
        return 0

    # Setup directories and logger
    create_directory(LOG_FILE.parent)
    logger = setup_logger(LOG_FILE, logger_name="poststroke_pipeline")
    log_message(logger, "info", f"{'='*60}")
    log_message(logger, "info", f"Post-Stroke Pipeline started — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(logger, "info", f"Args: {vars(args)}")

    print_info(f"Log file: {LOG_FILE.relative_to(PROJECT_ROOT)}")

    # Decide which steps to run
    if args.only_train or args.skip_extraction:
        steps_to_run = [s for s in STEPS if s["number"] == 4]
    elif args.skip_training:
        steps_to_run = [s for s in STEPS if s["number"] < 4]
    else:
        steps_to_run = STEPS

    total_steps = len(steps_to_run)

    for idx, step in enumerate(steps_to_run, start=1):
        step = dict(step)
        step["number"] = idx

        # Validate skeletons before training
        if STEPS.index(next(s for s in STEPS if s["description"] == step["description"])) == 3:
            if not validate_skeletons(SKELETONS_DIR, PIPELINE_NAME):
                log_message(logger, "error", "Skeleton validation failed — aborting.")
                return 1

        result = run_step(step, total_steps, logger)

        if not result.success:
            log_message(logger, "error", "Pipeline aborted due to step failure.")
            return 1

    log_message(logger, "info", "Pipeline completed successfully.")

    print_pipeline_summary(
        pipeline_name=PIPELINE_NAME,
        model_path=MODEL_PATH,
        results_dir=RESULTS_DIR,
        total_elapsed=0.0,
        val_accuracy="See training log for details",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
