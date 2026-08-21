from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SITE_LABELS = {
    "top_metal": 0,
    "top_oxygen": 1,
    "bridge": 2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tddft-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["summary", "transitions", "both"], required=True)
    parser.add_argument("--summary-output", type=Path, default=Path("tddft_summary_dataset.csv"))
    parser.add_argument("--transitions-output", type=Path, default=Path("tddft_transitions_dataset.csv"))
    return parser.parse_args()


def read_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df[df["tddft_status"] == "ok"].copy()

    df["label_site"] = df["site"]
    df["label_id"] = df["site"].map(SITE_LABELS)

    if df["label_id"].isna().any():
        raise ValueError("Unknown site found in summary.")

    df["label_id"] = df["label_id"].astype(int)

    return df


def build_summary_dataset(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()

    data["feature_1_total_oscillator_strength"] = data["tddft_total_oscillator_strength"]
    data["feature_2_onset_ev"] = data["tddft_onset_ev"]
    data["feature_3_peak_energy_ev"] = data["tddft_peak_energy_ev"]

    cols = [
        "material",
        "site",
        "label_site",
        "label_id",
        "feature_1_total_oscillator_strength",
        "feature_2_onset_ev",
        "feature_3_peak_energy_ev",
    ]

    return data[cols].sort_values(["material", "site"])


def read_transitions(tddft_dir: Path) -> pd.DataFrame:
    rows = []

    for path in sorted(tddft_dir.glob("*/*/transitions.csv")):
        material = path.parent.parent.name
        site = path.parent.name

        df = pd.read_csv(path)
        df["material"] = material
        df["site"] = site

        rows.append(df)

    if not rows:
        raise FileNotFoundError("No transitions.csv files found.")

    return pd.concat(rows, ignore_index=True)


def build_transitions_dataset(summary: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        "material",
        "site",
        "tddft_peak_energy_ev",
    ]

    data = transitions.merge(
        summary[summary_cols],
        on=["material", "site"],
        how="inner",
    )

    data["label_site"] = data["site"]
    data["label_id"] = data["site"].map(SITE_LABELS)

    if data["label_id"].isna().any():
        raise ValueError("Unknown site found in transitions.")

    data["label_id"] = data["label_id"].astype(int)

    data["feature_1_energy_ev"] = data["energy_ev"]
    data["feature_2_oscillator_strength"] = data["oscillator_strength"]
    data["feature_3_delta_from_peak_ev"] = (data["energy_ev"] - data["tddft_peak_energy_ev"])

    cols = [
        "material",
        "site",
        "label_site",
        "label_id",
        "transition_index",
        "feature_1_energy_ev",
        "feature_2_oscillator_strength",
        "feature_3_delta_from_peak_ev",
    ]

    return data[cols].sort_values(["material", "site", "transition_index"])


def save_dataset(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"\nSaved: {path}")
    print(f"Samples: {len(df)}")
    print("Classes:")
    print(df["label_site"].value_counts())


def main():
    args = parse_args()

    summary = read_summary(args.summary)

    if args.mode in ["summary", "both"]:
        summary_dataset = build_summary_dataset(summary)
        save_dataset(summary_dataset, args.summary_output)

    if args.mode in ["transitions", "both"]:
        transitions = read_transitions(args.tddft_dir)
        transitions_dataset = build_transitions_dataset(summary, transitions)
        save_dataset(transitions_dataset, args.transitions_output)

if __name__ == "__main__":
    main()