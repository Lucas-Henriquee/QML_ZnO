from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "datasets"
MD_WINDOWS_PER_SITE = 10

SITE_LABELS = {
    "top_metal": 0,
    "top_oxygen": 1,
    "bridge": 2,
}

SUMMARY_COLUMNS = [
    "material",
    "site",
    "feature_tddft_total_oscillator_strength",
    "feature_tddft_onset_ev",
    "feature_tddft_peak_energy_ev",
    "feature_dft_adsorption_energy_ev",
    "feature_dft_c_surface_distance_a",
    "feature_dft_gap_change_ev",
    "feature_co2rr_deltaG_CO2_to_COOH_ev",
    "feature_co2rr_deltaG_COOH_to_CO_ev",
    "feature_co2rr_limiting_potential_v",
    "feature_md_c_surface_distance_mean_a",
    "feature_md_c_surface_distance_std_a",
    "feature_md_mean_oco_angle_deg",
    "label_site",
    "label_site_id",
]

EXPANDED_COLUMNS = [
    "material",
    "site",
    "feature_tddft_transition_energy_ev",
    "feature_tddft_transition_oscillator_strength",
    "feature_tddft_transition_delta_from_peak_ev",
    "feature_dft_adsorption_energy_ev",
    "feature_dft_c_surface_distance_a",
    "feature_dft_gap_change_ev",
    "feature_co2rr_deltaG_CO2_to_COOH_ev",
    "feature_co2rr_deltaG_COOH_to_CO_ev",
    "feature_co2rr_limiting_potential_v",
    "feature_md_window_c_surface_distance_mean_a",
    "feature_md_window_c_surface_distance_std_a",
    "feature_md_window_mean_oco_angle_deg",
    "label_site",
    "label_site_id",
]

MD_UNIQUE_COLUMNS = [
    "material",
    "site",
    "md_window_id",
    "feature_md_window_c_surface_distance_mean_a",
    "feature_md_window_c_surface_distance_std_a",
    "feature_md_window_mean_oco_angle_deg",
    "label_site",
    "label_site_id",
]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}. Available: {list(df.columns)}")


def normalize_site(site: str) -> str:
    aliases = {
        "topMetal": "top_metal",
        "top_metal": "top_metal",
        "topOxygen": "top_oxygen",
        "top_oxygen": "top_oxygen",
        "bridge": "bridge",
    }
    return aliases.get(str(site), str(site))


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = df.copy()
    for col in columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def add_site_labels(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["label_site"] = data["site"]
    data["label_site_id"] = data["site"].map(SITE_LABELS)

    if data["label_site_id"].isna().any():
        unknown = sorted(data.loc[data["label_site_id"].isna(), "site"].unique())
        raise ValueError(f"Unknown site names: {unknown}")

    data["label_site_id"] = data["label_site_id"].astype(int)
    return data


def save_dataset(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data = df[columns].copy()
    data.to_csv(path, index=False)

    feature_cols = [col for col in data.columns if col.startswith("feature_")]
    label_cols = [col for col in data.columns if col.startswith("label_")]

    (path.parent / f"{path.stem}_feature_columns.txt").write_text(
        "\n".join(feature_cols) + "\n",
        encoding="utf-8",
    )
    (path.parent / f"{path.stem}_label_columns.txt").write_text(
        "\n".join(label_cols) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved: {path}")
    print(f"Samples: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Labels: {len(label_cols)}")
    print(data["label_site"].value_counts().sort_index())


def load_tddft_summary() -> pd.DataFrame:
    path = require_file(RESULTS_DIR / "tddft_summary.csv")
    df = pd.read_csv(path)

    require_columns(
        df,
        [
            "material",
            "site",
            "tddft_total_oscillator_strength",
            "tddft_onset_ev",
            "tddft_peak_energy_ev",
        ],
        "TDDFT summary",
    )

    if "tddft_status" in df.columns:
        df = df[df["tddft_status"].astype(str).str.lower() == "ok"].copy()

    cols = [
        "tddft_total_oscillator_strength",
        "tddft_onset_ev",
        "tddft_peak_energy_ev",
    ]
    df = numeric(df, cols)

    out = df[["material", "site"]].copy()
    out["site"] = out["site"].map(normalize_site)
    out["feature_tddft_total_oscillator_strength"] = df["tddft_total_oscillator_strength"]
    out["feature_tddft_onset_ev"] = df["tddft_onset_ev"]
    out["feature_tddft_peak_energy_ev"] = df["tddft_peak_energy_ev"]
    return out


def load_dft_features() -> pd.DataFrame:
    path = require_file(RESULTS_DIR / "dft_summary.csv")
    df = pd.read_csv(path)

    require_columns(
        df,
        [
            "material",
            "site",
            "adsorption_energy_ev",
            "c_surface_distance_ang",
            "clean_band_gap_ev",
            "adsorbed_band_gap_ev",
        ],
        "DFT summary",
    )

    if "dft_status" in df.columns:
        df = df[df["dft_status"].astype(str).str.lower() == "ok"].copy()

    cols = [
        "adsorption_energy_ev",
        "c_surface_distance_ang",
        "clean_band_gap_ev",
        "adsorbed_band_gap_ev",
    ]
    df = numeric(df, cols)

    out = df[["material", "site"]].copy()
    out["site"] = out["site"].map(normalize_site)
    out["feature_dft_adsorption_energy_ev"] = df["adsorption_energy_ev"]
    out["feature_dft_c_surface_distance_a"] = df["c_surface_distance_ang"]
    out["feature_dft_gap_change_ev"] = df["adsorbed_band_gap_ev"] - df["clean_band_gap_ev"]
    return out


def load_co2rr_features() -> pd.DataFrame:
    path = require_file(RESULTS_DIR / "co2rr_pathway_summary.csv")
    df = pd.read_csv(path)

    require_columns(
        df,
        [
            "material",
            "site",
            "deltaG_CO2_to_COOH_ev",
            "deltaG_COOH_to_CO_ev",
            "limiting_potential_v",
        ],
        "CO2RR summary",
    )

    if "pathway_status" in df.columns:
        df = df[df["pathway_status"].astype(str).str.lower() == "ok"].copy()

    cols = [
        "deltaG_CO2_to_COOH_ev",
        "deltaG_COOH_to_CO_ev",
        "limiting_potential_v",
    ]
    df = numeric(df, cols)

    out = df[["material", "site"]].copy()
    out["site"] = out["site"].map(normalize_site)
    out["feature_co2rr_deltaG_CO2_to_COOH_ev"] = df["deltaG_CO2_to_COOH_ev"]
    out["feature_co2rr_deltaG_COOH_to_CO_ev"] = df["deltaG_COOH_to_CO_ev"]
    out["feature_co2rr_limiting_potential_v"] = df["limiting_potential_v"]
    return out


def load_md_summary_features() -> pd.DataFrame:
    paths = sorted(RESULTS_DIR.glob("md/*/*/md_summary_*.csv"))
    if not paths:
        raise FileNotFoundError("No MD summary files found in results/md/*/*/")

    rows = []

    for path in paths:
        material = path.parent.parent.name
        site = normalize_site(path.parent.name)
        df = pd.read_csv(path)

        cols = ["c_surface_distance_ang", "oco_angle_deg"]
        require_columns(df, cols, f"MD summary {path}")
        df = numeric(df, cols)

        rows.append(
            {
                "material": material,
                "site": site,
                "feature_md_c_surface_distance_mean_a": df["c_surface_distance_ang"].mean(),
                "feature_md_c_surface_distance_std_a": df["c_surface_distance_ang"].std(ddof=1),
                "feature_md_mean_oco_angle_deg": df["oco_angle_deg"].mean(),
            }
        )

    return pd.DataFrame(rows)


def load_tddft_transitions() -> pd.DataFrame:
    paths = sorted(RESULTS_DIR.glob("tddft/*/*/transitions.csv"))
    if not paths:
        raise FileNotFoundError("No TDDFT transition files found in results/tddft/*/*/")

    rows = []

    for path in paths:
        material = path.parent.parent.name
        site = normalize_site(path.parent.name)
        df = pd.read_csv(path)

        cols = ["transition_index", "energy_ev", "oscillator_strength"]
        require_columns(df, cols, f"TDDFT transitions {path}")
        df = numeric(df, cols)

        rows.append(
            pd.DataFrame(
                {
                    "material": [material] * len(df),
                    "site": [site] * len(df),
                    "feature_tddft_transition_energy_ev": df["energy_ev"].to_numpy(),
                    "feature_tddft_transition_oscillator_strength": df["oscillator_strength"].to_numpy(),
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def load_md_windows(n_windows: int) -> pd.DataFrame:
    paths = sorted(RESULTS_DIR.glob("md/*/*/md_summary_*.csv"))
    if not paths:
        raise FileNotFoundError("No MD summary files found in results/md/*/*/")

    rows = []

    for path in paths:
        material = path.parent.parent.name
        site = normalize_site(path.parent.name)
        df = pd.read_csv(path)

        cols = ["time_ps", "c_surface_distance_ang", "oco_angle_deg"]
        require_columns(df, cols, f"MD summary {path}")
        df = numeric(df, cols).sort_values("time_ps").reset_index(drop=True)

        for window_id, idx in enumerate(np.array_split(np.arange(len(df)), n_windows)):
            if len(idx) == 0:
                continue

            window = df.iloc[idx]
            distance_std = window["c_surface_distance_ang"].std(ddof=1)
            if pd.isna(distance_std):
                distance_std = 0.0

            rows.append(
                {
                    "material": material,
                    "site": site,
                    "md_window_id": int(window_id),
                    "feature_md_window_c_surface_distance_mean_a": window["c_surface_distance_ang"].mean(),
                    "feature_md_window_c_surface_distance_std_a": distance_std,
                    "feature_md_window_mean_oco_angle_deg": window["oco_angle_deg"].mean(),
                }
            )

    return pd.DataFrame(rows)


def build_summary_dataset() -> pd.DataFrame:
    dataset = load_tddft_summary()

    for table in [load_dft_features(), load_co2rr_features(), load_md_summary_features()]:
        dataset = dataset.merge(table, on=["material", "site"], how="inner")

    return add_site_labels(dataset)


def build_expanded_dataset(summary: pd.DataFrame) -> pd.DataFrame:
    transitions = load_tddft_transitions()
    md_windows = load_md_windows(MD_WINDOWS_PER_SITE)

    data = transitions.merge(summary, on=["material", "site"], how="inner")
    data = data.merge(md_windows, on=["material", "site"], how="inner")

    data["feature_tddft_transition_delta_from_peak_ev"] = (
        data["feature_tddft_transition_energy_ev"]
        - data["feature_tddft_peak_energy_ev"]
    )

    data = add_site_labels(data)
    return data


def build_unique_md_window_dataset() -> pd.DataFrame:
    md_windows = load_md_windows(MD_WINDOWS_PER_SITE)
    md_windows = add_site_labels(md_windows)
    return md_windows.drop_duplicates(subset=MD_UNIQUE_COLUMNS).sort_values(
        ["site", "md_window_id"]
    )


def main() -> None:
    summary = build_summary_dataset()
    expanded = build_expanded_dataset(summary)
    md_unique = build_unique_md_window_dataset()

    save_dataset(
        summary,
        OUTPUT_DIR / "multitech_selected_features_dataset.csv",
        SUMMARY_COLUMNS,
    )

    save_dataset(
        expanded,
        OUTPUT_DIR / "multitech_expanded_ml_dataset_clean.csv",
        EXPANDED_COLUMNS,
    )

    save_dataset(
        md_unique,
        OUTPUT_DIR / "md_window_unique_dataset.csv",
        MD_UNIQUE_COLUMNS,
    )


if __name__ == "__main__":
    main()