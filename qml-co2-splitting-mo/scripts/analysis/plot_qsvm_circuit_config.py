from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ENTANGLEMENT_INPUT = (
    ROOT
    / "results/benchmarks/20260904_171734/benchmark_summary.csv"
)

REPETITIONS_INPUT = (
    ROOT
    / "results/benchmarks/20260904_171959/benchmark_summary.csv"
)

OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SEEDS = [42, 7, 21, 84, 123]
EXPECTED_ENTANGLEMENTS = ["linear", "circular", "full"]
EXPECTED_REPS = [1, 2]


def validate_common_config(df, source_name):
    if len(df) == 0:
        raise ValueError(f"No rows found in {source_name}")

    if set(df["feature_set"].astype(str)) != {"md3"}:
        raise ValueError(
            f"Unexpected feature set in {source_name}: "
            f"{set(df['feature_set'].astype(str))}"
        )

    if set(df["n_qubits"].astype(int)) != {6}:
        raise ValueError(
            f"Unexpected qubit count in {source_name}: "
            f"{set(df['n_qubits'].astype(int))}"
        )

    if set(df["scaler"].astype(str)) != {"minmax"}:
        raise ValueError(
            f"Unexpected scaler in {source_name}: "
            f"{set(df['scaler'].astype(str))}"
        )

    if set(df["C"].astype(float)) != {1.0}:
        raise ValueError(
            f"Unexpected C in {source_name}: "
            f"{set(df['C'].astype(float))}"
        )

    if set(df["max_samples"].astype(int)) != {30}:
        raise ValueError(
            f"Unexpected sample count in {source_name}: "
            f"{set(df['max_samples'].astype(int))}"
        )


def summarize_groups(df, group_column, group_values):
    records = []

    for value in group_values:
        subset = df[df[group_column] == value].copy()

        seeds = sorted(
            subset["random_state"].astype(int).tolist()
        )

        if seeds != sorted(EXPECTED_SEEDS):
            raise ValueError(
                f"Unexpected seeds for {group_column}={value}: {seeds}"
            )

        values = subset[
            "cv_balanced_accuracy_mean"
        ].astype(float).to_numpy()

        records.append(
            {
                "setting": value,
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
                "values": values,
            }
        )

    return records


def main():
    if not ENTANGLEMENT_INPUT.exists():
        raise FileNotFoundError(
            f"Missing benchmark file: {ENTANGLEMENT_INPUT}"
        )

    if not REPETITIONS_INPUT.exists():
        raise FileNotFoundError(
            f"Missing benchmark file: {REPETITIONS_INPUT}"
        )

    ent_df = pd.read_csv(ENTANGLEMENT_INPUT)
    rep_df = pd.read_csv(REPETITIONS_INPUT)

    ent_df = ent_df[
        ent_df["model"].astype(str).isin(["QSVM_IDEAL", "QSVM"])
    ].copy()

    rep_df = rep_df[
        rep_df["model"].astype(str).isin(["QSVM_IDEAL", "QSVM"])
    ].copy()

    validate_common_config(ent_df, "entanglement sweep")
    validate_common_config(rep_df, "repetition sweep")

    ent_df["entanglement"] = ent_df["entanglement"].astype(str)
    rep_df["reps"] = rep_df["reps"].astype(int)

    observed_entanglements = set(ent_df["entanglement"])

    if observed_entanglements != set(EXPECTED_ENTANGLEMENTS):
        raise ValueError(
            "Unexpected entanglement settings: "
            f"{observed_entanglements}"
        )

    observed_reps = set(rep_df["reps"])

    if observed_reps != set(EXPECTED_REPS):
        raise ValueError(
            f"Unexpected repetition settings: {observed_reps}"
        )

    ent_records = summarize_groups(
        ent_df,
        "entanglement",
        EXPECTED_ENTANGLEMENTS,
    )

    rep_records = summarize_groups(
        rep_df,
        "reps",
        EXPECTED_REPS,
    )

    summary_rows = []

    for record in ent_records:
        summary_rows.append(
            {
                "experiment": "entanglement",
                "setting": str(record["setting"]),
                "mean_cv_balanced_accuracy": record["mean"],
                "std_cv_balanced_accuracy": record["std"],
            }
        )

    for record in rep_records:
        summary_rows.append(
            {
                "experiment": "repetitions",
                "setting": str(record["setting"]),
                "mean_cv_balanced_accuracy": record["mean"],
                "std_cv_balanced_accuracy": record["std"],
            }
        )

    summary = pd.DataFrame(summary_rows)

    print("\nCircuit configuration summary:")
    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "ml_03_qsvm_circuit_configuration.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 4.2),
        sharey=True,
    )

    # Panel (a): entanglement
    ax = axes[0]

    x_ent = np.arange(len(EXPECTED_ENTANGLEMENTS))

    ent_means = np.array(
        [record["mean"] for record in ent_records]
    )

    ent_stds = np.array(
        [record["std"] for record in ent_records]
    )

    ax.errorbar(
        x_ent,
        ent_means,
        yerr=ent_stds,
        fmt="o",
        linestyle="none",
        capsize=5,
        markersize=7,
        linewidth=1.5,
        label="Mean ± SD",
    )

    jitter = np.linspace(
        -0.06,
        0.06,
        len(EXPECTED_SEEDS),
    )

    for i, record in enumerate(ent_records):
        ax.scatter(
            np.full(len(EXPECTED_SEEDS), x_ent[i]) + jitter,
            record["values"],
            s=24,
            alpha=0.55,
        )

    ax.set_xticks(x_ent)
    ax.set_xticklabels(
        ["Linear", "Circular", "Full"]
    )

    ax.set_xlabel("Entanglement")
    ax.set_ylabel("CV balanced accuracy")
    ax.set_title("(a) Entanglement topology")

    # Panel (b): repetitions
    ax = axes[1]

    x_rep = np.arange(len(EXPECTED_REPS))

    rep_means = np.array(
        [record["mean"] for record in rep_records]
    )

    rep_stds = np.array(
        [record["std"] for record in rep_records]
    )

    ax.errorbar(
        x_rep,
        rep_means,
        yerr=rep_stds,
        fmt="o",
        linestyle="none",
        capsize=5,
        markersize=7,
        linewidth=1.5,
        label="Mean ± SD",
    )

    for i, record in enumerate(rep_records):
        ax.scatter(
            np.full(len(EXPECTED_SEEDS), x_rep[i]) + jitter,
            record["values"],
            s=24,
            alpha=0.55,
        )

    ax.set_xticks(x_rep)
    ax.set_xticklabels(["1", "2"])

    ax.set_xlabel("Feature-map repetitions")
    ax.set_title("(b) Circuit repetitions")

    for ax in axes:
        ax.set_ylim(0.60, 0.92)

    axes[1].legend(
        frameon=False,
        loc="lower left",
    )

    fig.tight_layout()

    png_path = (
        OUTPUT_DIR
        / "ml_03_qsvm_circuit_configuration.png"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"\nSaved: {png_path}")

    plt.show()


if __name__ == "__main__":
    main()