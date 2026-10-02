"""Locate one MD summary per material and adsorption site."""

from pathlib import Path


def find_md_summaries(results_dir: Path) -> list[Path]:
    paths = sorted(set(results_dir.glob("md/*/*/md_summary.csv")) |
                   set(results_dir.glob("md/*/*/md_summary_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No MD summary files found below {results_dir / 'md'}")
    by_site: dict[Path, list[Path]] = {}
    for path in paths:
        by_site.setdefault(path.parent, []).append(path)
    for directory, candidates in by_site.items():
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise ValueError(f"Multiple MD summaries in {directory}: {names}. "
                             "Select a source directory with one trajectory per site.")
    return paths
