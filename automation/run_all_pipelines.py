"""
automation/run_all_pipelines.py
================================
Master CLI for the AI Physiotherapy CTR-GCN Automation System.

Default behaviour (no flags):
    Automatically detect which datasets are ready and run the appropriate
    pipeline for each of Lower Limb, Upper Limb, and Face.

    Detection order per pipeline
    ──────────────────────────────
    1. Preprocessed tensors found?
       → Skip extraction / build / split.
       → Run training script directly.

    2. Raw dataset found?
       → Run full pipeline:  extraction → tensor build → split → train.

    3. Neither found?
       → Print error and skip that pipeline (others continue).

Interactive menu (--menu flag):
    1  →  Run Lower Limb Pipeline
    2  →  Run Upper Limb Pipeline
    3  →  Run Face Pipeline
    4  →  Run Post-Stroke Pipeline
    5  →  Run ALL Pipelines  (sequential)
    6  →  Exit

Usage:
    python automation/run_all_pipelines.py            # auto-detect (default)
    python automation/run_all_pipelines.py --menu     # interactive menu
    python automation/run_all_pipelines.py --run 1    # non-interactive: lower only
    python automation/run_all_pipelines.py --run 5    # non-interactive: all
    python automation/run_all_pipelines.py --run 1 --skip-extraction
    python automation/run_all_pipelines.py --run 5 --only-train

Logs saved to: logs/master_pipeline.log
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project-root bootstrap — ensures imports work from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from automation.utils import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    RESET,
    YELLOW,
    PreprocessedStatus,
    check_preprocessed_tensors,
    create_directory,
    format_time,
    log_message,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
    run_command,
    setup_logger,
)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline registry  (used by the interactive menu / --run flag)
# ──────────────────────────────────────────────────────────────────────────────

PIPELINES: dict[int, dict] = {
    1: {
        "label":  "Lower Limb Pipeline",
        "module": "automation.run_lower_pipeline",
        "log":    PROJECT_ROOT / "logs" / "lower_pipeline.log",
    },
    2: {
        "label":  "Upper Limb Pipeline",
        "module": "automation.run_upper_pipeline",
        "log":    PROJECT_ROOT / "logs" / "upper_pipeline.log",
    },
    3: {
        "label":  "Face Pipeline",
        "module": "automation.run_face_pipeline",
        "log":    PROJECT_ROOT / "logs" / "face_pipeline.log",
    },
    4: {
        "label":  "Post-Stroke Pipeline",
        "module": "automation.run_poststroke_pipeline",
        "log":    PROJECT_ROOT / "logs" / "poststroke_pipeline.log",
    },
}

LOG_FILE = PROJECT_ROOT / "logs" / "master_pipeline.log"

# ──────────────────────────────────────────────────────────────────────────────
# Auto-detect pipeline configuration
# ──────────────────────────────────────────────────────────────────────────────

AUTO_PIPELINES: list[dict] = [
    {
        "label":          "Lower Limb",
        "processed_dir":  PROJECT_ROOT / "datasets/lower_limb",
        "raw_dir":        PROJECT_ROOT / "datasets/lower_limb/raw",
        "train_script":   PROJECT_ROOT / "training"      / "train_lower_limb_ctrgcn.py",
        "extract_script": PROJECT_ROOT / "preprocessing" / "extract_lower_limb_dataset.py",
        "build_script":   PROJECT_ROOT / "preprocessing" / "build_ctrgcn_dataset.py",
        "split_script":   PROJECT_ROOT / "preprocessing" / "split_dataset.py",
    },
    {
        "label":          "Upper Limb",
        "processed_dir":  PROJECT_ROOT / "datasets/upper_limb",
        "raw_dir":        PROJECT_ROOT / "datasets/upper_limb/raw",
        "train_script":   PROJECT_ROOT / "training"      / "train_upper_limb_ctrgcn.py",
        "extract_script": PROJECT_ROOT / "preprocessing" / "extract_upper_limb_dataset.py",
        "build_script":   PROJECT_ROOT / "preprocessing" / "build_ctrgcn_upper_dataset.py",
        "split_script":   PROJECT_ROOT / "preprocessing" / "split_upper_dataset.py",
    },
    {
        "label":          "Face",
        "processed_dir":  PROJECT_ROOT / "datasets/face",
        "raw_dir":        PROJECT_ROOT / "datasets/face/raw",
        "train_script":   PROJECT_ROOT / "training"      / "train_face_ctrgcn.py",
        "extract_script": PROJECT_ROOT / "preprocessing" / "extract_face_dataset.py",
        "build_script":   PROJECT_ROOT / "preprocessing" / "build_ctrgcn_face_dataset.py",
        "split_script":   PROJECT_ROOT / "preprocessing" / "split_face_dataset.py",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Auto-detect mode helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_tensor_check(label: str, status: PreprocessedStatus) -> None:
    """Print a formatted pre-flight check for preprocessed tensors."""
    width = 50
    border = "=" * width
    print(f"\n{CYAN}{BOLD}{border}{RESET}")
    print(f"{BOLD}  {label.upper()} PIPELINE{RESET}")
    print(f"{CYAN}{BOLD}{border}{RESET}")

    def _tick(ok: bool, text: str) -> None:
        icon = f"{GREEN}[OK]{RESET}" if ok else f"{RED}[  ]{RESET}"
        print(f"  {icon} {text}")

    _tick(status.skeletons_ok, f"Found preprocessed skeletons  ({status.skeletons_dir})")
    _tick(status.train_csv_ok, f"Found train_labels.csv        ({status.train_csv})")
    _tick(status.test_csv_ok,  f"Found test_labels.csv         ({status.test_csv})")
    print(f"{CYAN}{BOLD}{'=' * width}{RESET}\n")


def _run_auto_pipeline(pipeline: dict, logger) -> bool:
    """
    Execute one pipeline entry in auto-detect mode.

    Decision tree:
      1. Preprocessed tensors present? → train only.
      2. Raw dataset present?          → full pipeline.
      3. Neither?                      → error, return False.
    """
    label         = pipeline["label"]
    processed_dir = Path(pipeline["processed_dir"])
    raw_dir       = Path(pipeline["raw_dir"])
    train_script  = Path(pipeline["train_script"])

    # ── Step 1: check preprocessed tensors ───────────────────────────────────
    status = check_preprocessed_tensors(processed_dir)
    _print_tensor_check(label, status)

    if status.ready:
        print_info(
            f"[{label}] Preprocessed tensors found. Skipping extraction and preprocessing."
        )
        log_message(
            logger, "info",
            f"[{label}] Preprocessed tensors found — going straight to training."
        )
        print_info(f"[{label}] Starting training...")
        return _run_training(label, train_script, logger)

    # ── Step 2: check raw dataset ─────────────────────────────────────────────
    raw_exists = raw_dir.exists() and any(raw_dir.iterdir()) if raw_dir.exists() else False

    if raw_exists:
        print_warning(
            f"[{label}] No preprocessed tensors found. "
            f"Raw dataset detected at: {raw_dir}"
        )
        log_message(logger, "info", f"[{label}] Running full pipeline from raw data.")
        return _run_full_pipeline(pipeline, logger)

    # ── Step 3: nothing found ─────────────────────────────────────────────────
    msg = (
        f"[{label}] ERROR: No raw dataset or preprocessed tensors found.\n"
        f"         Checked processed : {processed_dir}\n"
        f"         Checked raw       : {raw_dir}"
    )
    print_error(msg)
    log_message(logger, "error", msg)
    return False


def _run_training(label: str, script: Path, logger) -> bool:
    """Run only the training script for *label*."""
    script = Path(script)
    if not script.exists():
        print_error(f"[{label}] Training script not found: {script}")
        log_message(logger, "error", f"[{label}] Training script missing: {script}")
        return False

    print_info(f"[{label}] Running: python {script.relative_to(PROJECT_ROOT)}")
    log_message(logger, "info", f"[{label}] Running training: {script}")

    result = run_command(
        cmd=[sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        logger=logger,
    )

    if result.success:
        print_success(f"[{label}] Training completed successfully.")
        log_message(logger, "info", f"[{label}] Training OK — {format_time(result.elapsed)}")
    else:
        print_error(f"[{label}] Training FAILED (exit code {result.returncode}).")
        log_message(logger, "error", f"[{label}] Training FAILED — exit {result.returncode}")

    return result.success


def _run_full_pipeline(pipeline: dict, logger) -> bool:
    """
    Run the full four-step pipeline:
    extract → build → split → train.
    """
    label = pipeline["label"]

    steps = [
        ("Extraction",   pipeline.get("extract_script")),
        ("Tensor Build", pipeline.get("build_script")),
        ("Split",        pipeline.get("split_script")),
        ("Training",     pipeline.get("train_script")),
    ]

    for step_name, script in steps:
        if script is None:
            print_warning(f"[{label}] No script for '{step_name}' — skipping.")
            continue

        script = Path(script)
        if not script.exists():
            print_error(f"[{label}] Script not found for '{step_name}': {script}")
            log_message(logger, "error", f"[{label}] Missing script: {script}")
            return False

        print_info(f"[{label}] {step_name} → {script.relative_to(PROJECT_ROOT)}")
        log_message(logger, "info", f"[{label}] {step_name}: {script}")

        result = run_command(
            cmd=[sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            logger=logger,
        )

        if not result.success:
            print_error(f"[{label}] '{step_name}' step FAILED (exit {result.returncode}).")
            log_message(
                logger, "error",
                f"[{label}] '{step_name}' FAILED — exit {result.returncode}"
            )
            return False

        print_success(f"[{label}] {step_name} completed.")

    return True


def run_all_auto(logger) -> int:
    """
    Auto-detect mode: run all three core pipelines (Lower Limb, Upper Limb,
    Face) with intelligent skipping of preprocessing when tensors are present.

    Returns:
        0 if all pipelines succeeded, 1 if any pipeline failed.
    """
    width = 58

    print(f"\n{CYAN}{BOLD}{'=' * width}{RESET}")
    print(f"{BOLD}  AUTO-DETECT MODE — AI PHYSIOTHERAPY CTR-GCN{RESET}")
    print(f"{DIM}  Checking for preprocessed tensors before running...{RESET}")
    print(f"{CYAN}{BOLD}{'=' * width}{RESET}\n")

    log_message(logger, "info", "Auto-detect mode started.")

    t_master = time.perf_counter()
    results: list[tuple[str, bool, float]] = []

    for pipeline in AUTO_PIPELINES:
        label = pipeline["label"]
        t0 = time.perf_counter()
        ok = _run_auto_pipeline(pipeline, logger)
        elapsed = time.perf_counter() - t0
        results.append((label, ok, elapsed))

        if not ok:
            print_warning(
                f"[{label}] Pipeline did not complete — continuing with remaining pipelines."
            )

    total_elapsed = time.perf_counter() - t_master
    _print_auto_summary(results, total_elapsed)

    all_ok = all(ok for _, ok, _ in results)
    log_message(
        logger,
        "info" if all_ok else "error",
        f"Auto-detect run complete — total {format_time(total_elapsed)} — "
        f"{'ALL OK' if all_ok else 'SOME FAILED'}",
    )
    return 0 if all_ok else 1


def _print_auto_summary(
    results: list[tuple[str, bool, float]],
    total_elapsed: float,
) -> None:
    """Print the final summary table for auto-detect mode."""
    width  = 58
    border = "=" * width

    print(f"\n{CYAN}{BOLD}{border}{RESET}")
    print(f"{BOLD}  FINAL REPORT{RESET}")
    print(f"{CYAN}{BOLD}{border}{RESET}")

    for label, ok, elapsed in results:
        status_str = f"{GREEN}SUCCESS{RESET}" if ok else f"{RED}FAILED{RESET}"
        print(f"  {label:<14} : {status_str}  ({format_time(elapsed)})")

    print(f"{CYAN}{'─' * width}{RESET}")
    print(f"  Total time : {BOLD}{format_time(total_elapsed)}{RESET}")

    all_ok = all(ok for _, ok, _ in results)
    if all_ok:
        print(f"\n  {GREEN}{BOLD}All pipelines completed successfully ✔{RESET}")
    else:
        failed = [lbl for lbl, ok, _ in results/lower_limb if not ok]
        print(f"\n  {RED}{BOLD}Failed: {', '.join(failed)}{RESET}")

    print(f"{CYAN}{BOLD}{border}{RESET}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Menu rendering  (used by --menu / interactive mode)
# ──────────────────────────────────────────────────────────────────────────────

_LOGO = r"""
   ___  ____   ____  _                     __  ___  __
  / _ |/  _/  / __ \/ /  __ _______  ___ / /_/ _ \/ /  _____  _____
 / __ |/ /   / /_/ / _ \/ // / __/ |/ (_-</ _ / // / _ \/ -_) __/ _ \
/_/ |_/___/  / .___/_//_/\_, /_/  |___/___/\__/____/\___/\__/_/ \___/
            /_/          /___/
"""


def print_menu() -> None:
    """Render the main navigation menu to stdout."""
    width = 58

    print(f"\n{CYAN}{BOLD}{'=' * width}{RESET}")
    print(f"{MAGENTA}{BOLD}{_LOGO}{RESET}")
    print(f"{CYAN}{BOLD}  AI PHYSIOTHERAPY — CTR-GCN AUTOMATION SYSTEM{RESET}")
    print(f"{DIM}  Skeleton-based exercise recognition pipeline{RESET}")
    print(f"{CYAN}{BOLD}{'=' * width}{RESET}\n")

    print(f"  {GREEN}{BOLD}[1]{RESET}  Run Lower Limb Pipeline")
    print(f"  {GREEN}{BOLD}[2]{RESET}  Run Upper Limb Pipeline")
    print(f"  {GREEN}{BOLD}[3]{RESET}  Run Face Pipeline")
    print(f"  {GREEN}{BOLD}[4]{RESET}  Run Post-Stroke Pipeline")
    print(f"  {BLUE}{BOLD}[5]{RESET}  Run {BOLD}ALL{RESET} Pipelines  (full project)")
    print(f"  {RED}{BOLD}[6]{RESET}  Exit\n")
    print(f"{CYAN}{'─' * width}{RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runner  (used by interactive / --run modes)
# ──────────────────────────────────────────────────────────────────────────────

def _patch_argv(flag_args: argparse.Namespace) -> list[str]:
    """Build a patched argv list for sub-pipeline parsers."""
    patched: list[str] = [sys.argv[0]]

    if flag_args.skip_extraction:
        patched.append("--skip-extraction")
    elif flag_args.skip_training:
        patched.append("--skip-training")
    elif flag_args.only_train:
        patched.append("--only-train")
    elif flag_args.only_inference:
        patched.append("--only-inference")

    return patched


def run_pipeline(pipeline_id: int, flag_args: argparse.Namespace, logger) -> bool:
    """
    Dynamically import and invoke the ``main()`` of the requested pipeline
    module.

    Returns:
        True on success, False on failure.
    """
    info  = PIPELINES[pipeline_id]
    label = info["label"]

    print_header(f"Starting: {label}")
    log_message(logger, "info", f"Launching: {label}")

    t_start       = time.perf_counter()
    original_argv = sys.argv[:]
    sys.argv      = _patch_argv(flag_args)

    try:
        import importlib
        module    = importlib.import_module(info["module"])
        exit_code = module.main()
    except SystemExit as exc:
        exit_code = int(exc.code) if exc.code is not None else 0
    except Exception as exc:
        print_error(f"Unexpected exception in {label}: {exc}")
        log_message(logger, "error", f"{label} raised: {exc}")
        exit_code = 1
    finally:
        sys.argv = original_argv

    elapsed     = time.perf_counter() - t_start
    elapsed_str = format_time(elapsed)

    if exit_code == 0:
        print_success(f"{label} finished in {elapsed_str}.")
        log_message(logger, "info", f"{label} OK — elapsed: {elapsed_str}")
        return True
    else:
        print_error(f"{label} FAILED  (exit code {exit_code})")
        log_message(logger, "error", f"{label} FAILED — exit {exit_code}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Summary helper  (interactive / --run modes)
# ──────────────────────────────────────────────────────────────────────────────

def _print_master_summary(
    results: list[tuple[str, bool, float]],
    total_elapsed: float,
) -> None:
    """Print the master completion table after running multiple pipelines."""
    width  = 58
    border = "=" * width

    print(f"\n{CYAN}{BOLD}{border}{RESET}")
    print(f"{BOLD}  MASTER PIPELINE SUMMARY{RESET}")
    print(f"{CYAN}{BOLD}{border}{RESET}")

    all_ok = all(ok for _, ok, _ in results)

    for label, ok, elapsed in results:
        icon   = f"{GREEN}OK{RESET}"   if ok else f"{RED}FAIL{RESET}"
        status = f"{GREEN}PASSED{RESET}" if ok else f"{RED}FAILED{RESET}"
        print(f"  [{icon}]  {label:<28} {status}  ({format_time(elapsed)})")

    print(f"{CYAN}{'─' * width}{RESET}")
    print(f"  Total wall-clock time : {BOLD}{format_time(total_elapsed)}{RESET}")

    if all_ok:
        print(f"\n  {GREEN}{BOLD}All pipelines completed successfully ✔{RESET}")
    else:
        failed = [lbl for lbl, ok, _ in results/lower_limb if not ok]
        print(f"\n  {RED}{BOLD}Pipelines that failed: {', '.join(failed)}{RESET}")

    print(f"{CYAN}{BOLD}{border}{RESET}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Interactive menu loop
# ──────────────────────────────────────────────────────────────────────────────

def interactive_menu(flag_args: argparse.Namespace, logger) -> int:
    """
    Present the main CLI menu and process the user's selection.
    Loops until the user chooses Exit (6) or EOF (Ctrl+D).

    Returns:
        0 if user exits cleanly; 1 if a pipeline fails.
    """
    overall_exit = 0

    while True:
        print_menu()

        try:
            choice_raw = input(f"  {BOLD}Enter your choice [1-6]: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{YELLOW}Exiting — goodbye!{RESET}")
            break

        if not choice_raw.isdigit():
            print_warning("Please enter a number between 1 and 6.")
            continue

        choice = int(choice_raw)

        if choice == 6:
            print(
                f"\n{YELLOW}{BOLD}Exiting the AI Physiotherapy Automation System."
                f" Goodbye!{RESET}\n"
            )
            break

        elif choice in (1, 2, 3, 4):
            log_message(
                logger, "info",
                f"User selected option {choice}: {PIPELINES[choice]['label']}"
            )
            ok = run_pipeline(choice, flag_args, logger)
            if not ok:
                overall_exit = 1

        elif choice == 5:
            log_message(logger, "info", "User selected: Run ALL Pipelines")
            print_header("Running ALL Pipelines — Sequential Execution")

            t_master_start = time.perf_counter()
            run_results: list[tuple[str, bool, float]] = []

            for pid in sorted(PIPELINES.keys()):
                t0      = time.perf_counter()
                ok      = run_pipeline(pid, flag_args, logger)
                elapsed = time.perf_counter() - t0
                run_results.append((PIPELINES[pid]["label"], ok, elapsed))

                if not ok:
                    overall_exit = 1
                    print_warning(
                        f"{PIPELINES[pid]['label']} failed. "
                        "Continuing with remaining pipelines..."
                    )

            total_elapsed = time.perf_counter() - t_master_start
            _print_master_summary(run_results, total_elapsed)
            log_message(
                logger, "info",
                f"All-pipelines run complete — total {format_time(total_elapsed)}"
            )

        else:
            print_warning("Invalid choice. Please enter a number between 1 and 6.")

    return overall_exit


# ──────────────────────────────────────────────────────────────────────────────
# Non-interactive (--run flag) execution
# ──────────────────────────────────────────────────────────────────────────────

def non_interactive_run(run_choice: int, flag_args: argparse.Namespace, logger) -> int:
    """
    Execute a pipeline non-interactively when ``--run`` is provided.

    Returns:
        0 on success, 1 on failure.
    """
    if run_choice in (1, 2, 3, 4):
        log_message(logger, "info", f"Non-interactive run: {PIPELINES[run_choice]['label']}")
        ok = run_pipeline(run_choice, flag_args, logger)
        return 0 if ok else 1

    elif run_choice == 5:
        log_message(logger, "info", "Non-interactive run: ALL Pipelines")
        print_header("Running ALL Pipelines — Sequential Execution")

        t_master_start = time.perf_counter()
        run_results: list[tuple[str, bool, float]] = []
        overall_ok = True

        for pid in sorted(PIPELINES.keys()):
            t0      = time.perf_counter()
            ok      = run_pipeline(pid, flag_args, logger)
            elapsed = time.perf_counter() - t0
            run_results.append((PIPELINES[pid]["label"], ok, elapsed))
            if not ok:
                overall_ok = False
                print_warning(f"{PIPELINES[pid]['label']} failed — continuing...")

        total_elapsed = time.perf_counter() - t_master_start
        _print_master_summary(run_results, total_elapsed)
        return 0 if overall_ok else 1

    else:
        print_error(f"Invalid --run value: {run_choice}. Must be 1-5.")
        return 1


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_all_pipelines.py",
        description=(
            "Master CLI for the AI Physiotherapy CTR-GCN Automation System.\n\n"
            "Default (no flags): auto-detect mode.\n"
            "  Checks each pipeline for preprocessed tensors and runs training\n"
            "  directly if found; falls back to full pipeline if raw data exists.\n\n"
            "Use --menu for the interactive menu, or --run N for a single pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python automation/run_all_pipelines.py                # auto-detect (default)
  python automation/run_all_pipelines.py --menu         # interactive menu
  python automation/run_all_pipelines.py --run 1        # lower limb only
  python automation/run_all_pipelines.py --run 5        # all (full pipeline)
  python automation/run_all_pipelines.py --run 1 --only-train
        """,
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        help="Show the interactive menu instead of auto-detect mode.",
    )
    parser.add_argument(
        "--run",
        type=int,
        choices=[1, 2, 3, 4, 5],
        metavar="N",
        help=(
            "Run a pipeline non-interactively:\n"
            "  1 = Lower Limb\n"
            "  2 = Upper Limb\n"
            "  3 = Face\n"
            "  4 = Post-Stroke\n"
            "  5 = All Pipelines"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip dataset extraction and building steps.",
    )
    group.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training step; run extraction only.",
    )
    group.add_argument(
        "--only-train",
        action="store_true",
        help="Skip extraction; go straight to training.",
    )
    group.add_argument(
        "--only-inference",
        action="store_true",
        help="(Placeholder) Run inference only.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    Entry point for the master automation script.

    Routing:
        - No flags  → auto-detect mode (check tensors → train or full pipeline)
        - --menu    → interactive CLI menu
        - --run N   → non-interactive run of pipeline N

    Returns:
        0 on success, 1 on any pipeline failure.
    """
    args = parse_args()

    create_directory(LOG_FILE.parent)
    logger = setup_logger(LOG_FILE, logger_name="master_pipeline")
    log_message(logger, "info", "=" * 60)
    log_message(
        logger, "info",
        f"Master Pipeline started — {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    log_message(logger, "info", f"Args: {vars(args)}")

    print_info(f"Master log: {LOG_FILE.relative_to(PROJECT_ROOT)}")

    if args.run is not None:
        return non_interactive_run(args.run, args, logger)
    elif args.menu:
        return interactive_menu(args, logger)
    else:
        return run_all_auto(logger)


if __name__ == "__main__":
    sys.exit(main())
