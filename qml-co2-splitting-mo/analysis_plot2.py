from __future__ import annotations

import csv
import math
from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
TDDFT_DIR   = RESULTS_DIR / "tddft"
OUTPUT_DIR  = RESULTS_DIR / "my_analysis"
MD_DIR = RESULTS_DIR / "md"
NEB_DIR = RESULTS_DIR / "neb_iddp"
DFT_CSV   = RESULTS_DIR / "dft_summary.csv"
TDDFT_CSV = RESULTS_DIR / "tddft_summary.csv"

CO2_BASELINE_BOND_ANG = 1.16

SITE_COLORS = {
    "top_metal":  "#268EE3",
    "top_oxygen": "#FF5622",
    "bridge":     "#4CAF50",
}
SITES = ["top_metal", "top_oxygen", "bridge"]

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"[error] file not found: {path}\n")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return float("nan")


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"[warning] no rows to write for {path.name}")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def get_x_labels(rows: list[dict], material_key="material", site_key="site") -> list[str]:
    return [f"{r[material_key]}\n{r[site_key]}" for r in rows]


def bar_chart(
    rows: list[dict],
    value_key: str,
    ylabel: str,
    title: str,
    path: Path,
    hline: float | None = None,
    hline_label: str = "",
    color_by_site: bool = True,
) -> None:
    labels = get_x_labels(rows)
    values = [safe_float(r.get(value_key)) for r in rows]
    colors = [SITE_COLORS.get(r.get("site", ""), "#9E9E9E") for r in rows] if color_by_site else ["#607D8B"] * len(rows)

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.9), 5))
    x = np.arange(len(labels))

    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.8)

    finite_vals = [v for v in values if math.isfinite(v)]
    if finite_vals:
        for bar, val in zip(bars, values):
            if math.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(finite_vals) * 0.015,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8
                )

    if hline is not None:
        ax.axhline(hline, color="red", linestyle="--", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    site_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SITE_COLORS.values()]
    site_labels  = list(SITE_COLORS.keys())
    if hline is not None:
        hline_handle = plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.2)
        all_handles  = site_handles + [hline_handle]
        all_labels   = site_labels  + [hline_label]
    else:
        all_handles = site_handles
        all_labels  = site_labels
    ax.legend(all_handles, all_labels, title="Site", fontsize=8, title_fontsize=8,
              loc="upper right", framealpha=0.7)

    fig.tight_layout()
    save_figure(fig, path)


#  DFT ANALYSIS
def dft_01_adsorption_energy(rows: list[dict], out: Path) -> None:

    table_rows = [
        {
            "material":             r["material"],
            "site":                 r["site"],
            "adsorption_energy_ev": safe_float(r.get("adsorption_energy_ev")),
            "dft_status":           r.get("dft_status", ""),
        }
        for r in rows
    ]
    write_table(out / "dft_table_adsorption_energy.csv", table_rows)
    bar_chart(
        rows=rows,
        value_key="adsorption_energy_ev",
        ylabel="Adsorption Energy (eV)",
        title="DFT Analysis 1 — Adsorption Energy per Site",
        path=out / "dft_01_adsorption_energy.png",
    )


def dft_02_co_bond_length(rows: list[dict], out: Path) -> None:

    augmented = []
    for r in rows:
        b1 = safe_float(r.get("co_bond_1_ang"))
        b2 = safe_float(r.get("co_bond_2_ang"))
        avg = (b1 + b2) / 2.0 if math.isfinite(b1) and math.isfinite(b2) else float("nan")
        augmented.append({**r, "_co_bond_avg": avg})

    table_rows = [
        {
            "material":        r["material"],
            "site":            r["site"],
            "co_bond_1_ang":   safe_float(r.get("co_bond_1_ang")),
            "co_bond_2_ang":   safe_float(r.get("co_bond_2_ang")),
            "co_bond_avg_ang": r["_co_bond_avg"],
            "stretch_ang":     r["_co_bond_avg"] - CO2_BASELINE_BOND_ANG
                               if math.isfinite(r["_co_bond_avg"]) else float("nan"),
        }
        for r in augmented
    ]
    write_table(out / "dft_table_co_bond_length.csv", table_rows)

    bar_chart(
        rows=[{**r, "co_bond_avg_ang": r["_co_bond_avg"]} for r in augmented],
        value_key="co_bond_avg_ang",
        ylabel="Average C-O Bond Length (Å)",
        title="DFT Analysis 2 — Average C-O Bond Length per Site",
        path=out / "dft_02_co_bond_length.png",
        hline=CO2_BASELINE_BOND_ANG,
        hline_label=f"Free CO₂ baseline ({CO2_BASELINE_BOND_ANG} Å)",
    )


def dft_03_oco_angle(rows: list[dict], out: Path) -> None:

    table_rows = [
        {
            "material":      r["material"],
            "site":          r["site"],
            "oco_angle_deg": safe_float(r.get("oco_angle_deg")),
            "bending_deg":   180.0 - safe_float(r.get("oco_angle_deg"))
                             if math.isfinite(safe_float(r.get("oco_angle_deg"))) else float("nan"),
        }
        for r in rows
    ]
    write_table(out / "dft_table_oco_angle.csv", table_rows)
    bar_chart(
        rows=rows,
        value_key="oco_angle_deg",
        ylabel="O-C-O Angle (°)",
        title="DFT Analysis 3 — O-C-O Angle per Site",
        path=out / "dft_03_oco_angle.png",
        hline=180.0,
        hline_label="Free CO₂ linear geometry (180°)",
    )


def dft_04_band_gap(rows: list[dict], out: Path) -> None:

    table_rows = [
        {
            "material":          r["material"],
            "site":              r["site"],
            "clean_band_gap_ev": safe_float(r.get("clean_band_gap_ev")),
            "ads_band_gap_ev":   safe_float(r.get("adsorbed_band_gap_ev")),
            "gap_change_ev":     safe_float(r.get("adsorbed_band_gap_ev")) -
                                 safe_float(r.get("clean_band_gap_ev")),
        }
        for r in rows
    ]
    write_table(out / "dft_table_band_gap.csv", table_rows)

    labels   = get_x_labels(rows)
    clean    = [safe_float(r.get("clean_band_gap_ev")) for r in rows]
    adsorbed = [safe_float(r.get("adsorbed_band_gap_ev")) for r in rows]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.9), 5))
    ax.bar(x - w / 2, clean,    width=w, label="Clean slab",   color="#90CAF9", edgecolor="white")
    ax.bar(x + w / 2, adsorbed, width=w, label="Adsorbed CO₂", color="#EF9A9A", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Band Gap (eV)", fontsize=11)
    ax.set_title("DFT Analysis 4 — Band Gap: Clean vs Adsorbed", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out / "dft_04_band_gap.png")


def dft_05_c_surface_distance(rows: list[dict], out: Path) -> None:

    table_rows = [
        {
            "material":               r["material"],
            "site":                   r["site"],
            "c_surface_distance_ang": safe_float(r.get("c_surface_distance_ang")),
        }
        for r in rows
    ]
    write_table(out / "dft_table_c_surface_distance.csv", table_rows)
    bar_chart(
        rows=rows,
        value_key="c_surface_distance_ang",
        ylabel="C-to-Surface Distance (Å)",
        title="DFT Analysis 5 — Carbon-to-Surface Distance per Site",
        path=out / "dft_05_c_surface_distance.png",
    )


#  TDDFT ANALYSIS
def tddft_01_onset_energy(rows: list[dict], out: Path) -> None:

    table_rows = [
        {
            "material":       r["material"],
            "site":           r["site"],
            "tddft_onset_ev": safe_float(r.get("tddft_onset_ev")),
            "tddft_status":   r.get("tddft_status", ""),
        }
        for r in rows
    ]
    write_table(out / "tddft_table_onset_energy.csv", table_rows)
    bar_chart(
        rows=rows,
        value_key="tddft_onset_ev",
        ylabel="Onset Energy (eV)",
        title="TDDFT Analysis 1 — Optical Absorption Onset Energy per Site",
        path=out / "tddft_01_onset_energy.png",
    )


def tddft_02_peak_energy(rows: list[dict], out: Path) -> None:

    table_rows = [
        {
            "material":             r["material"],
            "site":                 r["site"],
            "tddft_peak_energy_ev": safe_float(r.get("tddft_peak_energy_ev")),
        }
        for r in rows
    ]
    write_table(out / "tddft_table_peak_energy.csv", table_rows)
    bar_chart(
        rows=rows,
        value_key="tddft_peak_energy_ev",
        ylabel="Peak Energy (eV)",
        title="TDDFT Analysis 2 — Absorption Peak Energy per Site",
        path=out / "tddft_02_peak_energy.png",
    )


def tddft_03_total_oscillator_strength(rows: list[dict], out: Path) -> None:

    table_rows = [
        {
            "material":                        r["material"],
            "site":                            r["site"],
            "tddft_total_oscillator_strength": safe_float(r.get("tddft_total_oscillator_strength")),
        }
        for r in rows
    ]
    write_table(out / "tddft_table_total_osc.csv", table_rows)
    bar_chart(
        rows=rows,
        value_key="tddft_total_oscillator_strength",
        ylabel="Total Oscillator Strength (a.u.)",
        title="TDDFT Analysis 3 — Total Oscillator Strength per Site",
        path=out / "tddft_03_total_oscillator_strength.png",
    )


def tddft_04_osc_distribution(tddft_rows: list[dict], out: Path) -> None:

    all_transition_rows: list[dict] = []

    valid_cases = []
    for row in tddft_rows:
        material = row["material"]
        site     = row["site"]
        if row.get("tddft_status") != "ok":
            continue
        trans_path = TDDFT_DIR / material / site / "transitions.csv"
        if trans_path.exists():
            valid_cases.append((material, site, trans_path))

    if not valid_cases:
        print("[warning] no transitions.csv files found")
        return

    n_cases = len(valid_cases)
    ncols   = min(3, n_cases)
    nrows   = math.ceil(n_cases / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        "TDDFT Analysis 4 — Oscillator Strength Distribution per Site",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for idx, (material, site, trans_path) in enumerate(valid_cases):
        row_idx = idx // ncols
        col_idx = idx % ncols
        ax = axes[row_idx][col_idx]

        with trans_path.open("r", newline="", encoding="utf-8") as f:
            trans_data = list(csv.DictReader(f))

        energies  = [safe_float(t.get("energy_ev")) for t in trans_data]
        strengths = [safe_float(t.get("oscillator_strength")) for t in trans_data]

        for t in trans_data:
            all_transition_rows.append({
                "material":           material,
                "site":               site,
                "transition_index":   t.get("transition_index", ""),
                "energy_ev":          safe_float(t.get("energy_ev")),
                "oscillator_strength": safe_float(t.get("oscillator_strength")),
            })

        color = SITE_COLORS.get(site, "#9E9E9E")

        valid = [(e, s) for e, s in zip(energies, strengths)
                 if math.isfinite(e) and math.isfinite(s)]
        if valid:
            e_vals, s_vals = zip(*valid)
            markerline, stemlines, baseline = ax.stem(
                e_vals, s_vals,
                linefmt=color,
                markerfmt="o",
                basefmt="k-",
            )
            markerline.set_color(color)
            markerline.set_markersize(4)
            stemlines.set_color(color)
            stemlines.set_linewidth(1.2)

        ax.set_title(f"{material} | {site}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Energy (eV)", fontsize=9)
        ax.set_ylabel("Oscillator Strength (a.u.)", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(valid_cases), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    save_figure(fig, out / "tddft_04_osc_distribution.png")

    if all_transition_rows:
        write_table(out / "tddft_table_osc_distribution.csv", all_transition_rows)

def ev_to_nm(ev):
    ev = np.asarray(ev)
    return 1240 / np.clip(ev, 1e-6, None)

def nm_to_ev(nm):
    nm = np.asarray(nm)
    return 1240 / np.clip(nm, 1e-6, None)

def tddft_05_optical_spectra(tddft_rows: list[dict], out: Path) -> None:

    SIGMA    = 0.1
    E_MIN    = 0.1
    E_MAX    = 5.0
    N_POINTS = 500

    e_grid = np.linspace(E_MIN, E_MAX, N_POINTS)

    _PALETTE = [
        "#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
        "#FF9800", "#00BCD4", "#E91E63", "#8BC34A",
        "#795548", "#607D8B", "#F44336", "#3F51B5",
    ]
    _palette_iter = iter(_PALETTE)
    case_color: dict[tuple[str, str], str] = {}
    for row in tddft_rows:
        key = (row["material"], row["site"])
        if key not in case_color:
            case_color[key] = next(_palette_iter, "#9E9E9E")

    fig, ax = plt.subplots(figsize=(10, 5))

    spectra_peaks: list[dict] = []
    reference_peak: float | None = None

    plotted = 0
    for row in tddft_rows:
        material = row["material"]
        site     = row["site"]
        if row.get("tddft_status") != "ok":
            continue

        trans_path = TDDFT_DIR / material / site / "transitions.csv"
        if not trans_path.exists():
            print(f"[warning] transitions.csv not found for {material}/{site}")
            continue

        with trans_path.open("r", newline="", encoding="utf-8") as f:
            trans_data = list(csv.DictReader(f))

        energies  = np.array([safe_float(t.get("energy_ev")) for t in trans_data])
        strengths = np.array([safe_float(t.get("oscillator_strength")) for t in trans_data])

        spectrum = np.zeros(N_POINTS)
        for e_center, osc in zip(energies, strengths):
            if math.isfinite(e_center) and math.isfinite(osc):
                spectrum += osc * np.exp(-((e_grid - e_center) ** 2) / (2 * SIGMA ** 2))

        max_val = spectrum.max()
        if max_val > 0:
            spectrum_norm = spectrum / max_val
        else:
            continue

        peak_energy_broadened = float(e_grid[np.argmax(spectrum)])

        if reference_peak is None:
            reference_peak = peak_energy_broadened

        shift = peak_energy_broadened - reference_peak

        spectra_peaks.append({
            "material":                material,
            "site":                    site,
            "broadened_peak_ev":       peak_energy_broadened,
            "shift_from_reference_ev": shift,
            "reference_site":          f"{tddft_rows[0]['material']}_{tddft_rows[0]['site']}",
        })

        color = case_color.get((material, site), "#9E9E9E")
        label = f"{material} | {site}  (peak={peak_energy_broadened:.2f} eV, Δ={shift:+.2f} eV)"
        ax.plot(e_grid, spectrum_norm, label=label, color=color,
                linewidth=1.8, alpha=0.85)
        plotted += 1

    if plotted == 0:
        print("[warning] no valid transitions.csv files")
        plt.close(fig)
        return

    ax.axvline(1.77, linestyle="--", color="black", linewidth=1)
    ax.axvline(3.10, linestyle="--", color="black", linewidth=1)
    # Infrared, Visible, UV region
    ax.axvspan(0.0, 1.77, alpha=0.15, color="red", label="Infrared")
    ax.axvspan(1.77, 3.10, alpha=0.12, color="green", label="Visible")
    ax.axvspan(3.10, E_MAX, alpha=0.12, color="violet", label="UV")
    
    ax.set_xlabel("Energy (eV)", fontsize=11)
    ax.set_ylabel("Normalized Absorption (a.u.)", fontsize=11)
    ax.set_title(
        f"TDDFT Analysis 5 — Site-Dependent Optical Shift\n"
        f"(Gaussian broadening σ = {SIGMA} eV)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=8, ncol=2, framealpha=0.7)
    ax.set_xlim(E_MIN, E_MAX)
    secax = ax.secondary_xaxis(
        'top',
        functions=(ev_to_nm, nm_to_ev)
    )

    secax.set_xlabel("Wavelength (nm)", fontsize=11)
    secax.set_xticks([300, 400, 500, 700, 1000, 2000, 4000])
    secax.set_xticklabels(["300", "400", "500", "700", "1000", "2000", "4000"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out / "tddft_05_optical_spectra.png")
    #save_figure(fig, out / "sigma_comparison_0.05.png")
    if spectra_peaks:
       write_table(out / "tddft_table_optical_shift.csv", spectra_peaks)

# Dynamics Molecular
def md_read_site_summary(material: str, site: str) -> list[dict]:
    path = MD_DIR / material / site / "md_summary.csv"
    if not path.exists():
        print(f"[warning] MD summary not found: {path}")
        return []
    return read_csv(path)


def md_plot_time_series(
    rows: list[dict],
    y_key: str,
    ylabel: str,
    title: str,
    path: Path,
    hline: float | None = None,
    hline_label: str = "",
) -> None:
    if not rows:
        return

    time_ps = [safe_float(r.get("time_ps")) for r in rows]
    values = [safe_float(r.get(y_key)) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_ps, values, linewidth=1.8)

    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1.2, label=hline_label)
        ax.legend(fontsize=9)

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, path)


def md_analyze_site(material: str, site: str, out: Path) -> dict | None:
    rows = md_read_site_summary(material, site)
    if not rows:
        return None

    site_out = out / "md" / material / site
    site_out.mkdir(parents=True, exist_ok=True)

    md_plot_time_series(
        rows,
        y_key="temperature_K",
        ylabel="Temperature (K)",
        title=f"MD — Temperature vs Time ({material} | {site})",
        path=site_out / "md_temperature_vs_time.png",
        hline=300.0,
        hline_label="Target 300 K",
    )

    md_plot_time_series(
        rows,
        y_key="c_surface_distance_ang",
        ylabel="C-to-Surface Distance (Å)",
        title=f"MD — CO₂ Distance from Surface ({material} | {site})",
        path=site_out / "md_c_surface_distance_vs_time.png",
    )

    md_plot_time_series(
        rows,
        y_key="oco_angle_deg",
        ylabel="O-C-O Angle (degree)",
        title=f"MD — O-C-O Angle vs Time ({material} | {site})",
        path=site_out / "md_oco_angle_vs_time.png",
        hline=180.0,
        hline_label="Linear CO₂",
    )

    time_ps = [safe_float(r.get("time_ps")) for r in rows]
    epot = [safe_float(r.get("potential_energy_ev")) for r in rows]
    ekin = [safe_float(r.get("kinetic_energy_ev")) for r in rows]
    etot = [safe_float(r.get("total_energy_ev")) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_ps, epot, label="Potential energy")
    ax.plot(time_ps, ekin, label="Kinetic energy")
    ax.plot(time_ps, etot, label="Total energy")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"MD — Energy vs Time ({material} | {site})", fontsize=13, fontweight="bold", pad=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, site_out / "md_energy_vs_time.png")

    first = rows[0]
    last = rows[-1]

    distances = [safe_float(r.get("c_surface_distance_ang")) for r in rows]
    angles = [safe_float(r.get("oco_angle_deg")) for r in rows]
    co_avg = [safe_float(r.get("co_bond_avg_ang")) for r in rows]
    temps = [safe_float(r.get("temperature_K")) for r in rows]

    return {
        "material": material,
        "site": site,
        "initial_c_surface_distance_ang": safe_float(first.get("c_surface_distance_ang")),
        "average_c_surface_distance_ang": float(np.nanmean(distances)),
        "final_c_surface_distance_ang": safe_float(last.get("c_surface_distance_ang")),
        "average_oco_angle_deg": float(np.nanmean(angles)),
        "average_co_bond_ang": float(np.nanmean(co_avg)),
        "average_temperature_K": float(np.nanmean(temps)),
        "final_time_ps": safe_float(last.get("time_ps")),
    }


def md_all_analyses(out: Path) -> None:
    print("\nMD Analyses ")

    summary_rows = []

    for material_dir in sorted(MD_DIR.glob("*")):
        if not material_dir.is_dir():
            continue

        material = material_dir.name

        for site in SITES:
            result = md_analyze_site(material, site, out)
            if result is not None:
                summary_rows.append(result)

    if summary_rows:
        write_table(out / "md_table_summary_by_site.csv", summary_rows)

#  NEB ANALYSIS
def neb_read_endpoint_energy(xyz_path: Path) -> float:

    if not xyz_path.exists():
        print(f"[warning] endpoint XYZ not found: {xyz_path}")
        return float("nan")

    with xyz_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    if len(lines) < 2:
        return float("nan")

    header = lines[1]

    match = re.search(r"energy=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", header)

    if not match:
        print(f"[warning] no energy found in endpoint XYZ: {xyz_path}")
        return float("nan")

    return float(match.group(1))


def neb_read_internal_energy(txt_path: Path) -> float:

    if not txt_path.exists():
        print(f"[warning] internal NEB txt not found: {txt_path}")
        return float("nan")

    energy = float("nan")

    pattern = re.compile(r"Extrapolated:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

    with txt_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                energy = float(match.group(1))

    if not math.isfinite(energy):
        print(f"[warning] no extrapolated energy found in {txt_path}")

    return energy


def neb_read_path_energies(path_dir: Path) -> list[float]:

    image_paths = sorted(path_dir.glob("image_*.xyz"))

    if len(image_paths) < 2:
        print(f"[warning] not enough image_*.xyz files in {path_dir}")
        return []

    energies = []

    # Initial endpoint
    energies.append(neb_read_endpoint_energy(image_paths[0]))

    # Internal images
    n_internal = len(image_paths) - 2

    for idx in range(1, n_internal + 1):
        txt_path = path_dir / f"neb_image_{idx}.txt"
        energies.append(neb_read_internal_energy(txt_path))

    # Final endpoint
    energies.append(neb_read_endpoint_energy(image_paths[-1]))

    return energies


def neb_parse_path_name(path_name: str) -> tuple[str, str]:
    if "_to_" not in path_name:
        return path_name, ""

    initial_site, final_site = path_name.split("_to_", 1)
    return initial_site, final_site


def neb_analyze_path(material: str, path_dir: Path, out: Path) -> dict | None:
    path_name = path_dir.name
    initial_site, final_site = neb_parse_path_name(path_name)

    energies = neb_read_path_energies(path_dir)

    if not energies:
        return None

    if not energies or not math.isfinite(energies[0]):
        print(f"[warning] invalid NEB energies for {path_name}")
        return None

    energy_initial = energies[0]
    relative_energies = [
        e - energy_initial if math.isfinite(e) else float("nan")
        for e in energies
    ]

    finite_relative = [
        e for e in relative_energies
        if math.isfinite(e)
    ]

    if not finite_relative:
        print(f"[warning] no finite relative energies for {path_name}")
        return None

    barrier = max(finite_relative)
    highest_image = int(np.nanargmax(relative_energies))
    reaction_energy = relative_energies[-1]

    neb_out = out / "neb"
    neb_out.mkdir(parents=True, exist_ok=True)

    # Energy profile table for this path
    profile_rows = []
    for i, (e_abs, e_rel) in enumerate(zip(energies, relative_energies)):
        profile_rows.append({
            "material": material,
            "path": path_name,
            "initial_site": initial_site,
            "final_site": final_site,
            "image_index": i,
            "energy_ev": e_abs,
            "relative_energy_ev": e_rel,
        })

    write_table(
        neb_out / f"neb_table_energy_profile_{path_name}.csv",
        profile_rows,
    )

    # Energy profile plot
    x = np.arange(len(relative_energies))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        x,
        relative_energies,
        marker="o",
        linewidth=1.8,
    )

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.scatter(
        [highest_image],
        [barrier],
        s=70,
        zorder=3,
        label=f"Barrier = {barrier:.3f} eV",
    )

    ax.set_xticks(x)
    ax.set_xlabel("NEB Image Index")
    ax.set_ylabel("Relative Energy (eV)")
    ax.set_title(
        f"NEB Energy Profile — {material}\n{initial_site} → {final_site}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, neb_out / f"neb_01_energy_profile_{path_name}.png")

    return {
        "material": material,
        "path": path_name,
        "initial_site": initial_site,
        "final_site": final_site,
        "n_images": len(energies),
        "energy_initial_ev": energy_initial,
        "energy_final_ev": energies[-1],
        "reaction_energy_ev": reaction_energy,
        "barrier_ev": barrier,
        "highest_image": highest_image,
    }


def neb_barrier_comparison(rows: list[dict], out: Path) -> None:
    if not rows:
        return

    neb_out = out / "neb"
    neb_out.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(
        rows,
        key=lambda r: safe_float(r.get("barrier_ev")),
    )

    labels = [
        f"{r['initial_site']}\n→\n{r['final_site']}"
        for r in sorted_rows
    ]

    values = [
        safe_float(r.get("barrier_ev"))
        for r in sorted_rows
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        values,
        width=0.6,
        edgecolor="white",
        linewidth=0.8,
    )

    finite_vals = [v for v in values if math.isfinite(v)]

    if finite_vals:
        ymax = max(finite_vals)

        for bar, val in zip(bars, values):
            if math.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.015,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Energy Barrier (eV)")
    ax.set_title(
        "NEB Analysis — Migration Barrier Comparison",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, neb_out / "neb_02_barrier_comparison.png")


def neb_all_analyses(out: Path) -> None:
    print("\nNEB Analyses ")

    summary_rows = []

    for material_dir in sorted(NEB_DIR.glob("*")):
        if not material_dir.is_dir():
            continue

        material = material_dir.name

        for path_dir in sorted(material_dir.glob("*_to_*")):
            if not path_dir.is_dir():
                continue

            result = neb_analyze_path(material, path_dir, out)

            if result is not None:
                summary_rows.append(result)

    if not summary_rows:
        print("[warning] no NEB paths analyzed")
        return

    summary_rows = sorted(
        summary_rows,
        key=lambda r: safe_float(r.get("barrier_ev")),
    )

    ranked_rows = []

    for rank, row in enumerate(summary_rows, start=1):
        ranked_rows.append({
            "rank": rank,
            **row,
        })

    write_table(out / "neb_table_barriers.csv", ranked_rows)
    neb_barrier_comparison(ranked_rows, out)

#  MAIN
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dft_rows = read_csv(DFT_CSV)

    tddft_rows = read_csv(TDDFT_CSV)

    dft_ok   = [r for r in dft_rows   if r.get("dft_status")   == "ok"]
    tddft_ok = [r for r in tddft_rows if r.get("tddft_status") == "ok"]

    if not dft_ok:
        print("[warning] no successful DFT rows")
    if not tddft_ok:
        print("[warning] no successful TDDFT rows")

    print("DFT Analyses ")
    dft_01_adsorption_energy(dft_ok, OUTPUT_DIR)
    dft_02_co_bond_length(dft_ok, OUTPUT_DIR)
    dft_03_oco_angle(dft_ok, OUTPUT_DIR)
    dft_04_band_gap(dft_ok, OUTPUT_DIR)
    dft_05_c_surface_distance(dft_ok, OUTPUT_DIR)

    print("\nTDDFT Analyses ")
    tddft_01_onset_energy(tddft_ok, OUTPUT_DIR)
    tddft_02_peak_energy(tddft_ok, OUTPUT_DIR)
    tddft_03_total_oscillator_strength(tddft_ok, OUTPUT_DIR)
    tddft_04_osc_distribution(tddft_ok, OUTPUT_DIR)  
    tddft_05_optical_spectra(tddft_ok, OUTPUT_DIR)    

    print("\nMD Analyses ")
    md_all_analyses(OUTPUT_DIR)

    print("\nNEB Analyses ")
    neb_all_analyses(OUTPUT_DIR)
    
    print(f"\n All done\n")

if __name__ == "__main__":
    main()