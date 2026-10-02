"""Compatibility entry point; implementation is in scripts/legacy/visualization."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "legacy/visualization/analysis_plot2.py"), run_name="__main__")
