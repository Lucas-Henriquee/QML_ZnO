from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# REGEX
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

XYZ_ENERGY_RE = re.compile(
    rf"energy\s*=\s*[\"']?({NUMBER_RE})"
)

LOG_ENERGY_RE = re.compile(
    rf"Extrapolated:\s*({NUMBER_RE})"
)

RUN_FOLDER_RE = re.compile(
    r"^(top_metal|top_oxygen|bridge)"
    r"_to_"
    r"(top_metal|top_oxygen|bridge)"
    r"_(0\.\d+)$"
)


# ARGUMENTS
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate NEB energy profiles and migration barrier comparison plots."
    )

    parser.add_argument(
        "--base",
        type=Path,
        default=Path("results/neb/ZnO"),
        help="Base directory containing NEB folders.",
    )

    parser.add_argument(
        "--selected-fraction",
        type=float,
        default=0.40,
        help=(
            "Climb-start fraction used for the final migration barrier "
            "comparison with one bar per pathway."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution of saved figures.",
    )

    return parser.parse_args()


# BASIC HELPERS
def fraction_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def pathway_label(initial_site: str, final_site: str) -> str:
    return f"{initial_site} → {final_site}"


def read_summary(folder: Path) -> dict:
    path = folder / "neb_run_summary.json"

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARNING] Could not read {path}: {exc}")
        return {}


# DISCOVER RUNS AUTOMATICALLY
def discover_runs(base: Path) -> list[dict]:
    runs = []

    if not base.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base}")

    for folder in sorted(base.iterdir()):

        if not folder.is_dir():
            continue

        match = RUN_FOLDER_RE.match(folder.name)

        if not match:
            continue

        initial_site = match.group(1)
        final_site = match.group(2)
        fraction = float(match.group(3))

        runs.append(
            {
                "folder": folder,
                "folder_name": folder.name,
                "initial_site": initial_site,
                "final_site": final_site,
                "fraction": fraction,
                "pathway": pathway_label(initial_site, final_site),
            }
        )

    return runs


# ENERGY READING
def read_xyz_energy(path: Path) -> float:

    if not path.exists():
        return float("nan")

    try:
        with path.open("r", errors="ignore") as f:
            f.readline()
            comment = f.readline()

        match = XYZ_ENERGY_RE.search(comment)

        if match:
            return float(match.group(1))

    except Exception as exc:
        print(f"[WARNING] Could not read XYZ energy from {path}: {exc}")

    return float("nan")


def read_log_energy(path: Path) -> float:

    if not path.exists():
        return float("nan")

    last_energy = None

    try:
        with path.open("r", errors="ignore") as f:

            for line in f:

                match = LOG_ENERGY_RE.search(line)

                if match:
                    last_energy = float(match.group(1))

    except Exception as exc:
        print(f"[WARNING] Could not read GPAW log {path}: {exc}")
        return float("nan")

    if last_energy is None:
        return float("nan")

    return last_energy


def get_n_internal_images(folder: Path) -> int:

    summary = read_summary(folder)

    possible_keys = [
        "internal_images",
        "n_internal_images",
        "images",
    ]

    for key in possible_keys:

        if key in summary:

            try:
                return int(summary[key])

            except (TypeError, ValueError):
                pass

    log_files = list(folder.glob("neb_image_*.txt"))

    return len(log_files)


def get_energies(folder: Path) -> list[float]:

    n_internal = get_n_internal_images(folder)

    if n_internal <= 0:
        print(
            f"[WARNING] Could not determine number of internal images: "
            f"{folder}"
        )
        return []

    energies = []

    # Initial endpoint
    initial_file = folder / "image_00.xyz"

    energies.append(
        read_xyz_energy(initial_file)
    )

    # Internal images
    for image_index in range(1, n_internal + 1):

        log_file = folder / f"neb_image_{image_index}.txt"

        energies.append(
            read_log_energy(log_file)
        )

    # Final endpoint
    final_index = n_internal + 1

    final_file = folder / f"image_{final_index:02d}.xyz"

    if not final_file.exists():

        print(
            f"[WARNING] Expected final image not found: {final_file}"
        )

    energies.append(
        read_xyz_energy(final_file)
    )

    return energies


# CALCULATE NEB METRICS
def calculate_neb_metrics(energies: list[float]) -> dict:
    if not energies:
        return {}

    initial_energy = energies[0]

    if not math.isfinite(initial_energy):
        return {}

    relative_energies = []

    for energy in energies:

        if math.isfinite(energy):
            relative_energies.append(
                energy - initial_energy
            )
        else:
            relative_energies.append(
                float("nan")
            )

    valid_indices = [
        i
        for i, value in enumerate(relative_energies)
        if math.isfinite(value)
    ]

    if not valid_indices:
        return {}

    barrier_index = max(
        valid_indices,
        key=lambda i: relative_energies[i]
    )

    barrier = relative_energies[barrier_index]

    if math.isfinite(relative_energies[-1]):
        reaction_energy = relative_energies[-1]
    else:
        reaction_energy = float("nan")

    return {
        "energies": energies,
        "relative_energies": relative_energies,
        "barrier_eV": float(barrier),
        "barrier_image_index": int(barrier_index),
        "reaction_energy_eV": float(reaction_energy),
        "n_total_images": len(energies),
        "n_internal_images": len(energies) - 2,
    }


# LOAD ALL RUNS
def analyze_runs(runs: list[dict]) -> list[dict]:
    results = []

    for run in runs:

        folder = run["folder"]

        energies = get_energies(folder)

        metrics = calculate_neb_metrics(energies)

        if not metrics:

            print(
                f"[WARNING] Invalid energy data: "
                f"{folder.name}"
            )

            continue

        missing = [
            i
            for i, value in enumerate(metrics["energies"])
            if not math.isfinite(value)
        ]

        if missing:

            print(
                f"[WARNING] {folder.name}: "
                f"missing energies at images {missing}"
            )

        summary = read_summary(folder)

        result = {
            **run,
            **metrics,
            "status": summary.get("status", ""),
        }

        results.append(result)

        print(
            f"{folder.name:40s} "
            f"barrier={metrics['barrier_eV']:8.4f} eV   "
            f"DeltaE={metrics['reaction_energy_eV']:8.4f} eV"
        )

    return results


# SAVE CSV TABLES
def save_energy_table(results: list[dict], out: Path):
    rows = []

    for result in results:

        for image_index, (energy, relative) in enumerate(
            zip(
                result["energies"],
                result["relative_energies"],
            )
        ):

            rows.append(
                {
                    "pathway": result["pathway"],
                    "initial_site": result["initial_site"],
                    "final_site": result["final_site"],
                    "climb_start_fraction": result["fraction"],
                    "folder": result["folder_name"],
                    "image_index": image_index,
                    "energy_eV": energy,
                    "relative_energy_eV": relative,
                }
            )

    df = pd.DataFrame(rows)

    path = out / "neb_energy_table.csv"

    df.to_csv(path, index=False)

    print(f"\nEnergy table saved -> {path}")


def save_summary_table(results: list[dict], out: Path):
    rows = []

    for result in results:

        rows.append(
            {
                "pathway": result["pathway"],
                "initial_site": result["initial_site"],
                "final_site": result["final_site"],
                "climb_start_fraction": result["fraction"],
                "n_internal_images": result["n_internal_images"],
                "n_total_images": result["n_total_images"],
                "barrier_eV": result["barrier_eV"],
                "barrier_image_index": result["barrier_image_index"],
                "reaction_energy_eV": result["reaction_energy_eV"],
                "status": result["status"],
                "folder": result["folder_name"],
            }
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["initial_site", "final_site", "climb_start_fraction"]
    )

    path = out / "neb_summary_table.csv"

    df.to_csv(path, index=False)

    print(f"Summary table saved -> {path}")


# INDIVIDUAL ENERGY PROFILE

def plot_individual_profile(result: dict, output_dir: Path, dpi: int):
    relative = result["relative_energies"]

    x = np.arange(len(relative))

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.plot(
        x,
        relative,
        marker="o",
        markersize=7,
        linewidth=2,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xticks(x)

    ax.set_xlabel(
        "NEB image index",
        fontsize=12,
    )

    ax.set_ylabel(
        "Relative energy (eV)",
        fontsize=12,
    )

    ax.set_title(
        f"NEB energy profile: "
        f"{result['initial_site']} → {result['final_site']}\n"
        f"climb-start fraction = {result['fraction']:.2f}",
        fontsize=14,
    )

    text = (
        f"Barrier = {result['barrier_eV']:.3f} eV\n"
        f"ΔE = {result['reaction_energy_eV']:.3f} eV"
    )

    ax.text(
        0.98,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "alpha": 0.1,
        },
    )

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    filename = (
        f"neb_profile_"
        f"{result['initial_site']}_to_{result['final_site']}_"
        f"climb_{fraction_tag(result['fraction'])}.png"
    )

    output_path = output_dir / filename

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


# OVERLAY FRACTIONS FOR EACH PATHWAY
def plot_pathway_overlays(
    results: list[dict],
    output_dir: Path,
    dpi: int,
):
    groups = defaultdict(list)

    for result in results:

        key = (
            result["initial_site"],
            result["final_site"],
        )

        groups[key].append(result)

    for (initial_site, final_site), group in groups.items():

        group = sorted(
            group,
            key=lambda item: item["fraction"],
        )

        fig, ax = plt.subplots(
            figsize=(8, 5.5)
        )

        for result in group:

            relative = result["relative_energies"]

            x = np.arange(len(relative))

            ax.plot(
                x,
                relative,
                marker="o",
                markersize=5,
                linewidth=1.8,
                label=(
                    f"climb {result['fraction']:.2f} "
                    f"(barrier={result['barrier_eV']:.3f} eV)"
                ),
            )

        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1.0,
        )

        ax.set_xlabel(
            "NEB image index",
            fontsize=12,
        )

        ax.set_ylabel(
            "Relative energy (eV)",
            fontsize=12,
        )

        ax.set_title(
            f"NEB energy profile comparison: "
            f"{initial_site} → {final_site}",
            fontsize=14,
        )

        ax.legend(
            fontsize=9,
        )

        ax.grid(
            alpha=0.25,
        )

        fig.tight_layout()

        filename = (
            f"neb_overlay_"
            f"{initial_site}_to_{final_site}.png"
        )

        output_path = output_dir / filename

        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Overlay saved -> {output_path}"
        )


# ALL MIGRATION BARRIERS
def plot_all_barriers(
    results: list[dict],
    output_dir: Path,
    dpi: int,
):
    pathways = sorted(
        set(result["pathway"] for result in results)
    )

    fractions = sorted(
        set(result["fraction"] for result in results)
    )

    lookup = {
        (result["pathway"], result["fraction"]):
            result["barrier_eV"]
        for result in results
    }

    x = np.arange(len(pathways))

    n_fractions = len(fractions)

    total_width = 0.8
    bar_width = total_width / n_fractions

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for fraction_index, fraction in enumerate(fractions):

        offset = (
            fraction_index
            - (n_fractions - 1) / 2
        ) * bar_width

        values = [
            lookup.get(
                (pathway, fraction),
                np.nan,
            )
            for pathway in pathways
        ]

        bars = ax.bar(
            x + offset,
            values,
            width=bar_width * 0.9,
            label=f"climb {fraction:.2f}",
        )

        for bar, value in zip(bars, values):

            if math.isfinite(value):

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    ax.set_xticks(x)

    ax.set_xticklabels(
        pathways,
        fontsize=10,
    )

    ax.set_ylabel(
        "Migration barrier (eV)",
        fontsize=12,
    )

    ax.set_title(
        "NEB Analysis — Migration Barrier Comparison",
        fontsize=15,
        fontweight="bold",
    )

    ax.legend(
        title="Climb-start fraction",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    output_path = (
        output_dir
        / "migration_barrier_comparison_all_fractions.png"
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"All-barrier comparison saved -> {output_path}"
    )


# FINAL BARRIER COMPARISON FOR ONE FRACTION
def plot_selected_fraction_barriers(
    results: list[dict],
    output_dir: Path,
    selected_fraction: float,
    dpi: int,
):
    selected = [
        result
        for result in results
        if math.isclose(
            result["fraction"],
            selected_fraction,
            abs_tol=1e-8,
        )
    ]

    if not selected:

        print(
            f"[WARNING] No runs found for "
            f"climb fraction {selected_fraction:.2f}"
        )
        return

    selected = sorted(
        selected,
        key=lambda item: (
            item["initial_site"],
            item["final_site"],
        ),
    )

    labels = [
        result["pathway"]
        for result in selected
    ]

    barriers = [
        result["barrier_eV"]
        for result in selected
    ]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        x,
        barriers,
        width=0.6,
    )

    for bar, value in zip(bars, barriers):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        fontsize=10,
    )

    ax.set_ylabel(
        "Migration barrier (eV)",
        fontsize=12,
    )

    ax.set_title(
        "NEB Analysis — Migration Barrier Comparison\n"
        f"climb-start fraction = {selected_fraction:.2f}",
        fontsize=15,
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    filename = (
        "migration_barrier_comparison_"
        f"climb_{fraction_tag(selected_fraction)}.png"
    )

    output_path = output_dir / filename

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Selected-fraction barrier comparison saved -> "
        f"{output_path}"
    )


# MAIN
def main():
    args = parse_args()

    base = args.base

    output_dir = base / "comparison_plots"

    profiles_dir = output_dir / "individual_profiles"

    overlays_dir = output_dir / "pathway_overlays"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlays_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("NEB ANALYSIS")
    print("=" * 70)

    print(f"Base directory: {base}")
    print(
        f"Selected climb fraction: "
        f"{args.selected_fraction:.2f}"
    )

    print()

    runs = discover_runs(base)

    if not runs:

        raise RuntimeError(
            f"No NEB run folders found inside {base}"
        )

    print(
        f"Found {len(runs)} NEB runs:"
    )

    for run in runs:

        print(
            f"  {run['folder_name']}"
        )

    print()

    # Read energies
    results = analyze_runs(runs)

    if not results:

        raise RuntimeError(
            "No valid NEB energies were found."
        )

    print()
    print(
        f"Successfully analyzed "
        f"{len(results)} runs."
    )

    # Tables
    save_energy_table(
        results,
        output_dir,
    )

    save_summary_table(
        results,
        output_dir,
    )

    # Individual plots
    print("\nGenerating individual profiles...")

    for result in results:

        plot_individual_profile(
            result,
            profiles_dir,
            args.dpi,
        )

    print("\nGenerating pathway overlays...")

    plot_pathway_overlays(
        results,
        overlays_dir,
        args.dpi,
    )

    # All barriers / all fractions
    print("\nGenerating migration barrier comparison...")

    plot_all_barriers(
        results,
        output_dir,
        args.dpi,
    )

    # Selected fraction
    plot_selected_fraction_barriers(
        results,
        output_dir,
        args.selected_fraction,
        args.dpi,
    )

    print()
    print("=" * 70)
    print("NEB ANALYSIS COMPLETED")
    print("=" * 70)

    print(f"Results saved in:")
    print(output_dir)


if __name__ == "__main__":
    main()