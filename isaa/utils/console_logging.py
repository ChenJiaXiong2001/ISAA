"""Small helpers for mirroring console output to a log file."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import math
from pathlib import Path
import sys
import time
from typing import IO, Iterator

_progress_lines_active = False
_progress_lines_count = 0


def format_duration(seconds: float | None) -> str:
    """Format a duration as a compact human-readable string."""
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return "--:--"
    total_seconds = int(round(float(seconds)))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        return f"{days}d{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_tqdm_duration(seconds: float | None) -> str:
    """Format elapsed/ETA duration in tqdm's compact style."""
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return "??:??"
    total_seconds = int(round(float(seconds)))
    hours, remainder = divmod(total_seconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_tqdm_progress(step: int, total_steps: int, elapsed_seconds: float) -> str:
    """Return a compact tqdm-like progress bar line."""
    total_steps = max(int(total_steps), 1)
    step = min(max(int(step), 0), total_steps)
    fraction = float(step) / float(total_steps)
    percent = int(round(fraction * 100.0))
    blocks = " ▏▎▍▌▋▊▉█"
    block_index = min(int(fraction * (len(blocks) - 1)), len(blocks) - 1)
    rate = float(step) / elapsed_seconds if elapsed_seconds > 0 and step > 0 else 0.0
    eta_seconds = estimate_eta(elapsed_seconds, step, total_steps - step)
    return (
        f"{percent:3d}%|{blocks[block_index]}| {step}/{total_steps} "
        f"[{format_tqdm_duration(elapsed_seconds)}<{format_tqdm_duration(eta_seconds)}, {rate:5.2f}it/s]"
    )


def format_timestamped_lines(*lines: str) -> str:
    """Prefix every summary line with the same timestamp."""
    prefix = time.strftime("[%Y-%m-%d %H:%M:%S]")
    return "\n".join(f"{prefix} {line}" for line in lines)


def estimate_eta(elapsed_seconds: float, completed_units: int, remaining_units: int) -> float | None:
    """Estimate remaining time from completed work units."""
    if completed_units <= 0 or remaining_units <= 0:
        return 0.0 if remaining_units <= 0 else None
    return elapsed_seconds / float(completed_units) * float(remaining_units)


class TeeStream:
    """Write text to the original stream and a log file."""

    def __init__(self, stream: IO[str], log_file: IO[str]) -> None:
        self.stream = stream
        self.log_file = log_file
        self.encoding = getattr(stream, "encoding", None)
        self.errors = getattr(stream, "errors", None)

    def write(self, text: str) -> int:
        written = self.stream.write(text)
        self.log_file.write(text)
        return written

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(self.stream.fileno())


def _write_terminal_only(text: str) -> None:
    stream = sys.stdout
    if isinstance(stream, TeeStream):
        stream.stream.write(text)
        stream.stream.flush()
        return
    stream.write(text)
    stream.flush()


def write_progress_lines(lines: list[str] | tuple[str, ...]) -> None:
    """Refresh one or more terminal lines in place."""
    global _progress_lines_active, _progress_lines_count
    if not lines:
        return
    if _progress_lines_active:
        _write_terminal_only("\r")
        for _ in range(max(_progress_lines_count - 1, 0)):
            _write_terminal_only("\x1b[1A\r")
    for index, line in enumerate(lines):
        suffix = "\n" if index < len(lines) - 1 else ""
        _write_terminal_only(f"\x1b[2K{line}{suffix}")
    _progress_lines_active = True
    _progress_lines_count = len(lines)


def finish_progress_lines() -> None:
    """Move to a fresh line after in-place progress output."""
    global _progress_lines_active, _progress_lines_count
    if not _progress_lines_active:
        return
    _write_terminal_only("\n")
    _progress_lines_active = False
    _progress_lines_count = 0


def console_log_path(config: dict) -> Path | None:
    """Resolve training.save_dir/console.log when a save dir is configured."""
    training_cfg = config.get("training", {})
    if not isinstance(training_cfg, dict):
        return None
    save_dir = str(training_cfg.get("save_dir", "")).strip()
    if not save_dir:
        return None
    return Path(save_dir) / "console.log"


@contextmanager
def mirror_console_to_file(path: Path | None) -> Iterator[None]:
    """Mirror stdout and stderr to path while preserving normal console output."""
    if path is None or isinstance(sys.stdout, TeeStream) or isinstance(sys.stderr, TeeStream):
        with nullcontext():
            yield
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with path.open("a", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = TeeStream(original_stderr, log_file)  # type: ignore[assignment]
        try:
            print(f"console_log: writing to {path}")
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
