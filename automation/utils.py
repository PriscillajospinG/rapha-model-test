"""
automation/utils.py
===================
Shared utility functions for the Physiotherapy CTR-GCN automation system.

All pipeline scripts import from this module to avoid code duplication.

Functions:
    - run_command()        Execute a subprocess and stream output live.
    - print_header()       Print a styled section banner.
    - log_message()        Write a timestamped message to a log file.
    - format_time()        Convert elapsed seconds to HH:MM:SS string.
    - check_file_exists()  Validate that a file or directory is present.
    - create_directory()   Ensure a directory exists (mkdir -p).
    - validate_skeletons() Check that required .npy skeleton files exist.
    - suggest_fix()        Map known error patterns to human-readable tips.
"""

from __future__ import annotations

import subprocess
import sys
import time
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Sequence

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers (disabled automatically on Windows without ANSI support)
# ──────────────────────────────────────────────────────────────────────────────

def _supports_color() -> bool:
    """Return True if the terminal appears to support ANSI escape codes."""
    import os
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.name != "nt"


_COLOR = _supports_color()

RESET  = "\033[0m"  if _COLOR else ""
BOLD   = "\033[1m"  if _COLOR else ""
GREEN  = "\033[92m" if _COLOR else ""
YELLOW = "\033[93m" if _COLOR else ""
RED    = "\033[91m" if _COLOR else ""
CYAN   = "\033[96m" if _COLOR else ""
BLUE   = "\033[94m" if _COLOR else ""
MAGENTA= "\033[95m" if _COLOR else ""
DIM    = "\033[2m"  if _COLOR else ""

# ──────────────────────────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_header(title: str, width: int = 58) -> None:
    """
    Print a full-width banner around *title*.

    Example output:
        ══════════════════════════════════════════════════════════
         STEP 1 / 4  ·  Extracting Lower Limb Dataset
        ══════════════════════════════════════════════════════════
    """
    border = "═" * width
    print(f"\n{CYAN}{BOLD}{border}{RESET}")
    print(f"{BOLD} {title}{RESET}")
    print(f"{CYAN}{BOLD}{border}{RESET}\n")


def print_step(step: int, total: int, description: str) -> None:
    """Print a standardised step announcement banner."""
    title = f"STEP {step} / {total}  ·  {description}"
    print_header(title)


def print_success(message: str) -> None:
    """Print a green success line."""
    print(f"{GREEN}{BOLD}✔  {message}{RESET}")


def print_error(message: str) -> None:
    """Print a red error line."""
    print(f"{RED}{BOLD}✘  {message}{RESET}", file=sys.stderr)


def print_warning(message: str) -> None:
    """Print a yellow warning line."""
    print(f"{YELLOW}{BOLD}⚠  {message}{RESET}")


def print_info(message: str) -> None:
    """Print a dim informational line."""
    print(f"{DIM}ℹ  {message}{RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# Time helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """
    Convert a duration in seconds to a human-readable HH:MM:SS string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        A string like '0h 02m 34s'.
    """
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h}h {m:02d}m {s:02d}s"


# ──────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_directory(path: Path) -> Path:
    """
    Create *path* (and any missing parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved ``Path`` object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# File-existence helpers
# ──────────────────────────────────────────────────────────────────────────────

def check_file_exists(path: Path, label: str = "") -> bool:
    """
    Return True if *path* exists; print an error and return False otherwise.

    Args:
        path:  File or directory to check.
        label: Human-readable label printed in the error message.
    """
    path = Path(path)
    if path.exists():
        return True
    name = label or str(path)
    print_error(f"Required path not found: {name}")
    print_info(f"Expected at: {path.resolve()}")
    return False


def validate_skeletons(skeletons_dir: Path, pipeline_name: str) -> bool:
    """
    Verify that *skeletons_dir* exists and contains at least one .npy file.

    Args:
        skeletons_dir: Path to the skeletons directory to validate.
        pipeline_name: Human-readable pipeline name for error messages.

    Returns:
        True if validation passes; False otherwise.
    """
    skeletons_dir = Path(skeletons_dir)

    if not skeletons_dir.exists():
        print_error(
            f"[{pipeline_name}] Skeletons directory not found: {skeletons_dir}"
        )
        print_info("Run the extraction step first (or remove --skip-extraction).")
        return False

    npy_files = list(skeletons_dir.glob("*.npy"))
    if not npy_files:
        print_error(
            f"[{pipeline_name}] No .npy skeleton files found in: {skeletons_dir}"
        )
        print_info("The extraction step may have produced no output.")
        return False

    print_success(
        f"[{pipeline_name}] Skeleton validation passed — {len(npy_files)} files found."
    )
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessed-tensor detection
# ──────────────────────────────────────────────────────────────────────────────

class PreprocessedStatus:
    """
    Result of :func:`check_preprocessed_tensors`.

    Attributes:
        skeletons_ok:  True if skeletons/ dir with ≥1 .npy exists.
        train_csv_ok:  True if train_labels.csv exists.
        test_csv_ok:   True if test_labels.csv exists.
        skeletons_dir: Path that was checked.
        train_csv:     Path that was checked.
        test_csv:      Path that was checked.
    """

    def __init__(
        self,
        skeletons_ok: bool,
        train_csv_ok: bool,
        test_csv_ok: bool,
        skeletons_dir: Path,
        train_csv: Path,
        test_csv: Path,
    ) -> None:
        self.skeletons_ok = skeletons_ok
        self.train_csv_ok = train_csv_ok
        self.test_csv_ok = test_csv_ok
        self.skeletons_dir = skeletons_dir
        self.train_csv = train_csv
        self.test_csv = test_csv

    @property
    def ready(self) -> bool:
        """True when all three artefacts are present → training can start."""
        return self.skeletons_ok and self.train_csv_ok and self.test_csv_ok


def check_preprocessed_tensors(processed_dir: Path) -> "PreprocessedStatus":
    """
    Inspect *processed_dir* for the artefacts required to train directly.

    Expected layout::

        processed_dir/
            skeletons/          ← must contain ≥1 *.npy file
            train_labels.csv
            test_labels.csv

    Args:
        processed_dir: Root of the processed dataset directory.

    Returns:
        A :class:`PreprocessedStatus` describing what was found.
    """
    processed_dir = Path(processed_dir)
    skeletons_dir = processed_dir / "skeletons"
    train_csv = processed_dir / "train_labels.csv"
    test_csv = processed_dir / "test_labels.csv"

    skeletons_ok = skeletons_dir.is_dir() and bool(list(skeletons_dir.glob("*.npy")))
    train_csv_ok = train_csv.is_file()
    test_csv_ok = test_csv.is_file()

    return PreprocessedStatus(
        skeletons_ok=skeletons_ok,
        train_csv_ok=train_csv_ok,
        test_csv_ok=test_csv_ok,
        skeletons_dir=skeletons_dir,
        train_csv=train_csv,
        test_csv=test_csv,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Error diagnostics
# ──────────────────────────────────────────────────────────────────────────────

# Map regex patterns found in stderr/stdout to actionable fix suggestions.
_ERROR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"No such file or directory", re.IGNORECASE),
        "A required file or directory is missing.\n"
        "  → Check that all dataset folders exist and re-run the extraction step.",
    ),
    (
        re.compile(r"\.csv.*not found|FileNotFoundError.*\.csv", re.IGNORECASE),
        "A CSV label file is missing.\n"
        "  → Run the extraction step to regenerate frame label CSV files.",
    ),
    (
        re.compile(r"\.npy.*not found|no .npy files", re.IGNORECASE),
        "Skeleton .npy files are missing.\n"
        "  → Re-run the dataset building step (build_*_dataset.py).",
    ),
    (
        re.compile(r"ModuleNotFoundError|ImportError", re.IGNORECASE),
        "A Python dependency is missing.\n"
        "  → Run: pip install -r requirements.txt",
    ),
    (
        re.compile(r"CUDA.*out of memory|RuntimeError.*memory", re.IGNORECASE),
        "GPU ran out of memory.\n"
        "  → Reduce the batch size in the training script, or switch to CPU.",
    ),
    (
        re.compile(r"shape mismatch|size mismatch|tensor.*shape", re.IGNORECASE),
        "Tensor shape mismatch detected.\n"
        "  → Verify that the dataset was built with the correct number of joints/channels.\n"
        "  → Delete processed_dataset*/skeletons and re-run from Step 1.",
    ),
    (
        re.compile(r"checkpoint|\.pth.*not found", re.IGNORECASE),
        "A model checkpoint file is missing.\n"
        "  → Complete the training step before running inference.",
    ),
    (
        re.compile(r"Permission denied", re.IGNORECASE),
        "File system permission error.\n"
        "  → Check folder ownership / run with appropriate permissions.",
    ),
    (
        re.compile(r"KeyboardInterrupt", re.IGNORECASE),
        "Execution was interrupted by the user (Ctrl+C).",
    ),
]


def suggest_fix(output: str) -> str:
    """
    Scan *output* for known error patterns and return a suggested fix string.

    Args:
        output: Combined stdout + stderr from a failed subprocess call.

    Returns:
        A human-readable suggested fix string, or a generic fallback.
    """
    for pattern, suggestion in _ERROR_PATTERNS:
        if pattern.search(output):
            return suggestion
    return (
        "An unexpected error occurred.\n"
        "  → Review the log file for the full traceback.\n"
        "  → Ensure all dependencies are installed: pip install -r requirements.txt"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(log_file: Path, logger_name: str = "pipeline") -> logging.Logger:
    """
    Create and return a named ``logging.Logger`` that writes to *log_file*.

    The file handler appends to *log_file*; a StreamHandler is NOT added
    (console output is handled separately via print_* helpers).

    Args:
        log_file:    Destination log file path.
        logger_name: Internal logger name.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s  [%(levelname)-8s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_message(logger: logging.Logger, level: str, message: str) -> None:
    """
    Write a log entry at the given *level*.

    Args:
        logger:  A ``logging.Logger`` instance.
        level:   One of 'debug', 'info', 'warning', 'error', 'critical'.
        message: The log message text.
    """
    getattr(logger, level.lower(), logger.info)(message)


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess runner
# ──────────────────────────────────────────────────────────────────────────────

class StepResult:
    """Container for the result of a single pipeline step."""

    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        elapsed: float,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed = elapsed

    @property
    def success(self) -> bool:
        return self.returncode == 0

    @property
    def combined_output(self) -> str:
        return self.stdout + "\n" + self.stderr


def run_command(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    env: Optional[dict] = None,
) -> StepResult:
    """
    Execute *cmd* as a subprocess, streaming stdout/stderr to the console in
    real time and capturing both streams for logging and error analysis.

    Args:
        cmd:    Command token list, e.g. ``[sys.executable, "script.py", "--flag"]``.
        cwd:    Working directory for the subprocess. Defaults to the project root.
        logger: If provided, all output lines are written to this logger.
        env:    Optional environment variable overrides (merged with ``os.environ``).

    Returns:
        A ``StepResult`` with returncode, captured output, and elapsed time.
    """
    import os

    # Build the environment (inherit current env, then apply overrides)
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    # Force stdout/stderr to be unbuffered so output streams live
    proc_env.setdefault("PYTHONUNBUFFERED", "1")

    # Resolve working directory
    work_dir = Path(cwd) if cwd else Path(__file__).resolve().parent.parent

    cmd_str = " ".join(str(c) for c in cmd)
    print_info(f"Running: {cmd_str}")
    if logger:
        log_message(logger, "info", f"Running: {cmd_str}")

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    t_start = time.perf_counter()

    try:
        process = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(work_dir),
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout for live display
            text=True,
            bufsize=1,                  # line-buffered
        )

        assert process.stdout is not None
        for line in process.stdout:
            line_stripped = line.rstrip()
            print(line_stripped)
            stdout_lines.append(line_stripped)
            if logger:
                log_message(logger, "debug", line_stripped)

        process.wait()
        elapsed = time.perf_counter() - t_start

    except FileNotFoundError as exc:
        elapsed = time.perf_counter() - t_start
        error_msg = f"Executable not found: {exc}"
        print_error(error_msg)
        if logger:
            log_message(logger, "error", error_msg)
        return StepResult(
            returncode=127,
            stdout="",
            stderr=error_msg,
            elapsed=elapsed,
        )
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t_start
        print_warning("Execution interrupted by user (Ctrl+C).")
        if logger:
            log_message(logger, "warning", "KeyboardInterrupt received.")
        return StepResult(
            returncode=130,
            stdout="\n".join(stdout_lines),
            stderr="KeyboardInterrupt",
            elapsed=elapsed,
        )

    return StepResult(
        returncode=process.returncode,
        stdout="\n".join(stdout_lines),
        stderr="\n".join(stderr_lines),
        elapsed=elapsed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline summary banner
# ──────────────────────────────────────────────────────────────────────────────

def print_pipeline_summary(
    pipeline_name: str,
    model_path: Optional[Path],
    results_dir: Optional[Path],
    total_elapsed: float,
    val_accuracy: Optional[str] = None,
) -> None:
    """
    Print the completion banner shown after a successful pipeline run.

    Args:
        pipeline_name: Display name, e.g. 'Lower Limb'.
        model_path:    Path to the saved best-model checkpoint.
        results_dir:   Path to the results/output folder.
        total_elapsed: Total wall-clock seconds for the full pipeline.
        val_accuracy:  Optional validation accuracy string to display.
    """
    width = 58
    border = "═" * width
    print(f"\n{GREEN}{BOLD}{border}")
    print("  PIPELINE COMPLETED SUCCESSFULLY  ✔")
    print(border)
    print(f"  Pipeline       : {pipeline_name}")
    if val_accuracy:
        print(f"  Val Accuracy   : {val_accuracy}")
    print(f"  Training Time  : {format_time(total_elapsed)}")
    if model_path and Path(model_path).exists():
        print(f"  Checkpoint     : {model_path}")
    if results_dir and Path(results_dir).exists():
        print(f"  Results Folder : {results_dir}")
    print(f"{border}{RESET}\n")
