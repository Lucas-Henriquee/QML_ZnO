"""Choose readable output names without overwriting previous runs."""

from pathlib import Path


def next_run_name(output_root: Path, prefix: str) -> str:
    index = 1
    while (output_root / f"{prefix}_{index:02d}").exists():
        index += 1
    return f"{prefix}_{index:02d}"
