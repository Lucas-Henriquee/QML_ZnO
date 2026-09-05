from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 21, 84, 123]

RUNS = {
    ("Expanded", "SVM"): (
        ROOT
        / "results/benchmarks/20260904_171120/benchmark_summary.csv"
    ),
    ("Expanded", "QSVM"): (
        ROOT
        / "results/benchmarks/20260904_171129/benchmark_summary.csv"
    ),
    ("Unique AIMD", "SVM"): (
        ROOT
        / "results/benchmarks/20260904_172124/benchmark_summary.csv"
    ),
    ("Unique AIMD", "QSVM"): (
        ROOT
        / "results/benchmarks/20260904_171409/benchmark_summary.csv"
    ),
}


def load_test_balanced_accuracy(path, expected_model):
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {path}")

    df = pd.read_csv(path)

    if len(df) != 5:
        raise ValueError(
            f"{path} contains {len(df)} rows; expected 5 seeds."
        )

    observed_seeds = sorted(df["random_state"].astype(int).tolist())
    expected_seeds = sorted(SEEDS)

    if observed_seeds != expected_seeds:
        raise ValueError(
            f"Unexpected seeds in {path}: {observed_seeds}"
        )

    models = set(df["model"].astype(str))

    if expected_model == "SVM":
        valid_models = {"SVM"}
    else:
        valid_models = {"QSVM_IDEAL", "QSVM"}

    if not models.issubset(valid_models):
        raise ValueError(
            f"Unexpected model values in {path}: {models}"
        )

    values = df["test_balanced_accuracy"].astype(float).to_numpy()

    return values


def main():
    records = []

    for (dataset, model), path in RUNS.items():
        values = load_test_balanced_accuracy(path, model)

        records.append(
            {
                "dataset": dataset,
                "model": model,
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
                "values": values,
            }
        )

    summary = pd.DataFrame(
        [
            {
                "dataset": r["dataset"],
                "model": r["model"],
                "mean_test_balanced_accuracy": r["mean"],
                "std_test_balanced_accuracy": r["std"],
            }
            for r in records
        ]
    )

    print("\nSummary:")
    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "ml_01_generalization_comparison.csv",
        index=False,
    )

    datasets = ["Expanded", "Unique AIMD"]
    models = ["SVM", "QSVM"]

    x = np.arange(len(datasets), dtype=float)
    offsets = {"SVM": -0.10, "QSVM": 0.10}
    markers = {"SVM": "o", "QSVM": "s"}

    fig, ax = plt.subplots(figsize=(6.4, 4.5))

    for model in models:
        means = []
        stds = []
        all_values = []

        for dataset in datasets:
            record = next(
                r
                for r in records
                if r["dataset"] == dataset and r["model"] == model
            )

            means.append(record["mean"])
            stds.append(record["std"])
            all_values.append(record["values"])

        xpos = x + offsets[model]

        ax.errorbar(
            xpos,
            means,
            yerr=stds,
            fmt=markers[model],
            capsize=5,
            markersize=8,
            linewidth=1.5,
            label=model,
        )

        jitter = np.linspace(-0.025, 0.025, len(SEEDS))

        for i, values in enumerate(all_values):
            ax.scatter(
                np.full(len(values), xpos[i]) + jitter,
                values,
                s=24,
                alpha=0.55,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Expanded\n($N=90$)",
            "Unique AIMD\n($N=30$)",
        ]
    )

    ax.set_ylabel("Test balanced accuracy")
    ax.set_ylim(0.25, 1.05)

    ax.legend(frameon=False)

    fig.tight_layout()

    png_path = OUTPUT_DIR / "ml_01_generalization_comparison.png"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"\nSaved: {png_path}")

    plt.show()


if __name__ == "__main__":
    main()