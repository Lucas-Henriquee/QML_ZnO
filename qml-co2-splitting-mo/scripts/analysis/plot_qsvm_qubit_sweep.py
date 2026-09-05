from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "results/benchmarks/20260904_171605/benchmark_summary.csv"
)

OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_QUBITS = [3, 6, 9, 12]
EXPECTED_SEEDS = [42, 7, 21, 84, 123]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing benchmark file: {INPUT}")

    df = pd.read_csv(INPUT)

    df = df[df["model"].astype(str).isin(["QSVM_IDEAL", "QSVM"])].copy()

    if len(df) != 20:
        raise ValueError(
            f"Expected 20 QSVM runs, found {len(df)}"
        )

    observed_qubits = sorted(df["n_qubits"].astype(int).unique().tolist())

    if observed_qubits != EXPECTED_QUBITS:
        raise ValueError(
            f"Unexpected qubit values: {observed_qubits}"
        )

    records = []

    for n_qubits in EXPECTED_QUBITS:
        subset = df[df["n_qubits"].astype(int) == n_qubits].copy()

        seeds = sorted(subset["random_state"].astype(int).tolist())

        if seeds != sorted(EXPECTED_SEEDS):
            raise ValueError(
                f"Unexpected seeds for {n_qubits} qubits: {seeds}"
            )

        values = subset["cv_balanced_accuracy_mean"].astype(float).to_numpy()

        records.append(
            {
                "n_qubits": n_qubits,
                "mean_cv_balanced_accuracy": np.mean(values),
                "std_cv_balanced_accuracy": np.std(values, ddof=1),
                "values": values,
            }
        )

    summary = pd.DataFrame(
        [
            {
                "n_qubits": r["n_qubits"],
                "mean_cv_balanced_accuracy": r[
                    "mean_cv_balanced_accuracy"
                ],
                "std_cv_balanced_accuracy": r[
                    "std_cv_balanced_accuracy"
                ],
            }
            for r in records
        ]
    )

    print("\nQubit sweep summary:")
    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "ml_02_qsvm_qubit_sweep.csv",
        index=False,
    )

    x = np.array(EXPECTED_QUBITS, dtype=float)

    means = np.array(
        [r["mean_cv_balanced_accuracy"] for r in records]
    )

    stds = np.array(
        [r["std_cv_balanced_accuracy"] for r in records]
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.5))

    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        linestyle="none",
        capsize=5,
        markersize=7,
        linewidth=1.5,
        label="Mean ± SD across five splits",
    )

    jitter = np.linspace(-0.12, 0.12, len(EXPECTED_SEEDS))

    for i, record in enumerate(records):
        ax.scatter(
            np.full(len(EXPECTED_SEEDS), x[i]) + jitter,
            record["values"],
            s=24,
            alpha=0.55,
        )

    ax.set_xticks(EXPECTED_QUBITS)

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("CV balanced accuracy")

    ax.set_ylim(0.65, 0.90)

    ax.legend(frameon=False)

    fig.tight_layout()

    png_path = OUTPUT_DIR / "ml_02_qsvm_qubit_sweep.png"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"\nSaved: {png_path}")

    plt.show()


if __name__ == "__main__":
    main()