"""
automation/run_all_pipelines.py
================================
Master CLI for the AI Physiotherapy CTR-GCN Automation System.

Presents an interactive menu that lets the user run any combination of
the three pipeline modules (Lower Limb, Upper Limb, Face) with a single
command.

Menu:
    1  →  Run Lower Limb Pipeline
    2  →  Run Upper Limb Pipeline
    3  →  Run Face Pipeline
    4  →  Run ALL Pipelines  (sequential)
    5  →  Exit

Usage:
    python automation/run_all_pipelines.py          # interactive menu
    python automation/run_all_pipelines.py --run 1  # non-interactive: run lower
    python automation/run_all_pipelines.py --run 4  # non-interactive: run all
    python automation/run_all_pipelines.py --run 1 --skip-extraction
    python automation/run_all_pipelines.py --run 4 --only-train

Global flags (forwarded to each individual pipeline):
    --skip-extraction
    --skip-training
    --only-train
    --only-inference

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
    create_directory,
    format_time,
    log_message,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
    setup_logger,
)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline registry
# ──────────────────────────────────────────────────────────────────────────────

# Each entry maps a menu number to display label and the importable main()
PIPELINES: dict[int, dict] = {
    1: {
        "label": "Lower Limb Pipeline",
        "module": "automation.run_lower_pipeline",
        "log":    PROJECT_ROOT / "logs" / "lower_pipeline.log",
    },
    2: {
        "label": "Upper Limb Pipeline",
        "module": "automation.run_upper_pipeline",
        "log":    PROJECT_ROOT / "logs" / "upper_pipeline.log",
    },
    3: {
        "label": "Face Pipeline",
        "module": "automation.run_face_pipeline",
        "log":    PROJECT_ROOT / "logs" / "face_pipeline.log",
    },
    4: {
        "label": "Post-Stroke Pipeline",
        "module": "automation.run_poststroke_pipeline",
        "log":    PROJECT_ROOT / "logs" / "poststroke_pipeline.log",
    },
}

LOG_FILE = PROJECT_ROOT / "logs" / "master_pipeline.log"

# ──────────────────────────────────────────────────────────────────────────────
# Menu rendering
# ──────────────────────────────────────────────────────────────────────────────

_LOGO = r"""
   ___  ____   ____  _                     __  ___  __
  / _ |/  _/  / __ \/ /  __ _______  ___ / /_/ _ \/ /  _____ _____
 / __ |/ /   / /_/ / _ \/ // / __/ |/ (_-< __/ // / _ \/ -_) __/ _ \
/_/ |_/___/  / .___/_//_/\_, /_/  |___/___/\__/____/\___/\__/_/ \___/
            /_/          /___/
"""


def print_menu() -> None:
    """Render the main navigation menu to stdout."""
    width = 58

    print(f"\n{CYAN}{BOLD}{'═' * width}{RESET}")
    print(f"{MAGENTA}{BOLD}{_LOGO}{RESET}")
    print(f"{CYAN}{BOLD}  AI PHYSIOTHERAPY — CTR-GCN AUTOMATION SYSTEM{RESET}")
    print(f"{DIM}  Skeleton-based exercise recognition pipeline{RESET}")
    print(f"{CYAN}{BOLD}{'═' * width}{RESET}\n")

    print(f"  {GREEN}{BOLD}[1]{RESET}  🦵  Run Lower Limb Pipeline")
    print(f"  {GREEN}{BOLD}[2]{RESET}  💪  Run Upper Limb Pipeline")
    print(f"  {GREEN}{BOLD}[3]{RESET}  😊  Run Face Pipeline")
    print(f"  {GREEN}{BOLD}[4]{RESET}  🧠  Run Post-Stroke Pipeline")
    print(f"  {BLUE}{BOLD}[5]{RESET}  🚀  Run {BOLD}ALL{RESET} Pipelines  (full project)")
    print(f"  {RED}{BOLD}[6]{RESET}  ⏻   Exit\n")
    print(f"{CYAN}{'─' * width}{RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────────────────────────────────────

def _patch_argv(flag_args: argparse.Namespace) -> list[str]:
    """
    Convert master-level flag namespace into a list of argv tokens that
    the individual pipeline parsers understand.

    This avoids passing sys.argv directly (which contains master-script flags)
    to the sub-pipeline parsers, preventing argument collisions.
    """
    patched: list[str] = [sys.argv[0]]   # argv[0] is ignored by argparse

    if flag_args.skip_extraction:
        patched.append("--skip-extraction")
    elif flag_args.skip_training:
        patched.append("--skip-training")
    elif flag_args.only_train:
        patched.append("--only-train")
    elif flag_args.only_inference:
        patched.append("--only-inference")

    return patched


def run_pipeline(
    pipeline_id: int,
    flag_args: argparse.Namespace,
    logger,
) -> bool:
    """
    Dynamically import and invoke the ``main()`` function of the requested
    pipeline module.

    Using Python import (rather than subprocess) keeps all output on the
    same terminal session and avoids spawning a new interpreter.

    Args:
        pipeline_id: 1 = Lower, 2 = Upper, 3 = Face.
        flag_args:   Parsed master-level flags forwarded to the sub-pipeline.
        logger:      Master logger.

    Returns:
        True on success, False on failure.
    """
    info = PIPELINES[pipeline_id]
    label = info["label"]

    print_header(f"Starting: {label}")
    log_message(logger, "info", f"Launching: {label}")

    t_start = time.perf_counter()

    # Temporarily replace sys.argv so the sub-pipeline parser is happy
    original_argv = sys.argv[:]
    sys.argv = _patch_argv(flag_args)

    try:
        # Import the pipeline module and call its main()
        import importlib
        module = importlib.import_module(info["module"])
        exit_code: int = module.main()
    except SystemExit as exc:
        # main() may call sys.exit(); treat non-zero as failure
        exit_code = int(exc.code) if exc.code is not None else 0
    except Exception as exc:
        print_error(f"Unexpected exception in {label}: {exc}")
        log_message(logger, "error", f"{label} raised: {exc}")
        exit_code = 1
    finally:
        sys.argv = original_argv   # restore argv

    elapsed = time.perf_counter() - t_start
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
# Summary helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_master_summary(
    results: list[tuple[str, bool, float]],
    total_elapsed: float,
) -> None:
    """
    Print the master completion table after running multiple pipelines.

    Args:
        results:       List of (pipeline_label, success, elapsed_seconds).
        total_elapsed: Wall-clock time for the entire master run.
    """
    width = 58
    border = "═" * width

    print(f"\n{CYAN}{BOLD}{border}{RESET}")
    print(f"{BOLD}  MASTER PIPELINE SUMMARY{RESET}")
    print(f"{CYAN}{BOLD}{border}{RESET}")

    all_ok = all(ok for _, ok, _ in results)

    for label, ok, elapsed in results:
        icon = f"{GREEN}✔{RESET}" if ok else f"{RED}✘{RESET}"
        status = f"{GREEN}PASSED{RESET}" if ok else f"{RED}FAILED{RESET}"
        print(f"  {icon}  {label:<28} {status}  ({format_time(elapsed)})")

    print(f"{CYAN}{BOLD}{'─' * width}{RESET}")
    print(f"  Total wall-clock time : {BOLD}{format_time(total_elapsed)}{RESET}")

    if all_ok:
        print(f"\n  {GREEN}{BOLD}All pipelines completed successfully ✔{RESET}")
    else:
        failed = [label for label, ok, _ in results if not ok]
        print(f"\n  {RED}{BOLD}Pipelines that failed: {', '.join(failed)}{RESET}")

    print(f"{CYAN}{BOLD}{border}{RESET}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Interactive menu loop
# ──────────────────────────────────────────────────────────────────────────────

def interactive_menu(flag_args: argparse.Namespace, logger) -> int:
    """
    Present the main CLI menu and process the user's selection.

    Loops until the user chooses Exit (5) or an EOF (Ctrl+D).

    Args:
        flag_args: Parsed flag namespace to forward to sub-pipelines.
        logger:    Master logger.

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
            print(f"\n{YELLOW}{BOLD}Exiting the AI Physiotherapy Automation System. Goodbye! 👋{RESET}\n")
            break

        elif choice in (1, 2, 3, 4):
            log_message(logger, "info", f"User selected option {choice}: {PIPELINES[choice]['label']}")
            t0 = time.perf_counter()
            ok = run_pipeline(choice, flag_args, logger)
            elapsed = time.perf_counter() - t0
            if not ok:
                overall_exit = 1

        elif choice == 5:
            # Run all pipelines sequentially
            log_message(logger, "info", "User selected: Run ALL Pipelines")
            print_header("Running ALL Pipelines — Sequential Execution")

            t_master_start = time.perf_counter()
            run_results: list[tuple[str, bool, float]] = []

            for pid in sorted(PIPELINES.keys()):
                t0 = time.perf_counter()
                ok = run_pipeline(pid, flag_args, logger)
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
            log_message(logger, "info", f"All-pipelines run complete — total {format_time(total_elapsed)}")

        else:
            print_warning("Invalid choice. Please enter a number between 1 and 6.")

    return overall_exit


# ──────────────────────────────────────────────────────────────────────────────
# Non-interactive (--run flag) execution
# ──────────────────────────────────────────────────────────────────────────────

def non_interactive_run(run_choice: int, flag_args: argparse.Namespace, logger) -> int:
    """
    Execute a pipeline non-interactively when ``--run`` is provided.

    Args:
        run_choice: Pipeline number (1-4).
        flag_args:  Flag namespace forwarded to sub-pipelines.
        logger:     Master logger.

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
            t0 = time.perf_counter()
            ok = run_pipeline(pid, flag_args, logger)
            elapsed = time.perf_counter() - t0
            run_results.append((PIPELINES[pid]["label"], ok, elapsed))
            if not ok:
                overall_ok = False
                print_warning(f"{PIPELINES[pid]['label']} failed — continuing...")

        total_elapsed = time.perf_counter() - t_master_start
        _print_master_summary(run_results, total_elapsed)
        return 0 if overall_ok else 1

    else:
        print_error(f"Invalid --run value: {run_choice}. Must be 1-4.")
        return 1


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_all_pipelines.py",
        description=(
            "Master CLI for the AI Physiotherapy CTR-GCN Automation System.\n"
            "Run without arguments for the interactive menu, or use --run to "
            "specify a pipeline non-interactively."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python automation/run_all_pipelines.py
  python automation/run_all_pipelines.py --run 1
  python automation/run_all_pipelines.py --run 4 --only-train
  python automation/run_all_pipelines.py --run 2 --skip-extraction
        """,
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
    # Pipeline flags forwarded to sub-pipelines
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

    Returns:
        0 on success, 1 on any pipeline failure.
    """
    args = parse_args()

    # Setup master log
    create_directory(LOG_FILE.parent)
    logger = setup_logger(LOG_FILE, logger_name="master_pipeline")
    log_message(logger, "info", "=" * 60)
    log_message(logger, "info", f"Master Pipeline started — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(logger, "info", f"Args: {vars(args)}")

    print_info(f"Master log: {LOG_FILE.relative_to(PROJECT_ROOT)}")

    if args.run is not None:
        return non_interactive_run(args.run, args, logger)
    else:
        return interactive_menu(args, logger)


if __name__ == "__main__":
    sys.exit(main())
