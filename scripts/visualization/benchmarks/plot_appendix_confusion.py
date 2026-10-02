"""Plot the published unique-window comparison from saved seed-42 predictions."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUNS = {
    "svm": ROOT / "results/benchmarks/svm_unique_windows/svm/svm_md3_rbf_power_seed42",
    "qsvm": ROOT / "results/benchmarks/qsvm_unique_windows/qsvm/qsvm_md3_6q_linear_r1_minmax_reference_seed42",
}
LABELS = ["Top metal", "Top oxygen", "Bridge"]


def main():
    output = ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    reference = None
    for model, run in RUNS.items():
        with (run / "predictions.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        split = sorted((int(row["row_id"]), int(row["true_label"])) for row in rows)
        if reference is not None and split != reference:
            raise ValueError("Models do not use the same test samples.")
        reference = split
        matrix = np.zeros((3, 3), dtype=int)
        for row in rows:
            matrix[int(row["true_label"]), int(row["pred_label"])] += 1
        with (run / "confusion_matrix.csv").open(newline="") as handle:
            saved = list(csv.reader(handle))
        expected = np.array([[int(value) for value in row[1:]] for row in saved[1:]])
        if not np.array_equal(matrix, expected) or matrix.sum() != 8:
            raise ValueError("Predictions disagree with the saved eight-sample matrix.")

        fig, ax = plt.subplots(figsize=(4.3, 3.8), constrained_layout=True)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=3)
        ax.set_xticks(range(3), LABELS, rotation=25, ha="right")
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted site")
        ax.set_ylabel("True site")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=13,
                        color="white" if matrix[i, j] >= 2 else "black")
        for extension in ("png", "pdf"):
            path = output / f"appendix_confusion_{model}_unique_seed42.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(path)
        plt.close(fig)
        balanced_accuracy = np.mean(matrix.diagonal() / matrix.sum(axis=1))
        print(f"{model}: test samples={matrix.sum()}, balanced accuracy={balanced_accuracy:.6f}")


if __name__ == "__main__":
    main()
