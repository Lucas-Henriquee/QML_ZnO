from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.analysis.analyze_model_results import split_status


MODEL_COLORS = {
    "Classical SVM": "#4472C4",
    "Local QSVM": "#ED7D31",
    "IBM QSVM": "#70AD47",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a model comparison and its confusion matrices."
    )
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--model-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def save_figure(fig, output_dir: Path, name: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf", "svg"]:
        path = output_dir / f"{name}.{extension}"
        kwargs = {"dpi": dpi} if extension == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"[model-figures] wrote {path}")
    plt.close(fig)


def plot_metrics(comparison: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    names = comparison["model"].astype(str).tolist()
    x = np.arange(len(names))
    width = 0.34
    accuracy = comparison["test_balanced_accuracy"].astype(float).to_numpy()
    f1 = comparison["test_f1_macro"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    bars_accuracy = ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Balanced accuracy",
        color="#4472C4",
    )
    bars_f1 = ax.bar(
        x + width / 2,
        f1,
        width,
        label="Macro F1",
        color="#ED7D31",
    )
    for bars in [bars_accuracy, bars_f1]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x, names)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    split_label = "Shared Test Split" if split_status(comparison) == "yes" else "Recorded Test Splits"
    ax.set_title(f"Model Performance on {split_label}")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, output_dir, "model_metrics_comparison", dpi)


def confusion_label(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "qsvm_ibm" in parts or "ibm" in parts:
        return "IBM QSVM"
    if "qsvm" in parts:
        return "Local QSVM"
    if "svm" in parts:
        return "Classical SVM"
    return path.parent.name


def plot_confusion_matrices(model_run_dir: Path, output_dir: Path, dpi: int, comparison: pd.DataFrame) -> None:
    paths = sorted(model_run_dir.rglob("confusion_matrix.csv"))
    if "requested_model" in comparison:
        requested = set(comparison["requested_model"].astype(str))
        paths = [path for path in paths if requested.intersection(path.relative_to(model_run_dir).parts)]
    if not paths:
        raise SystemExit(f"No confusion matrices found below: {model_run_dir}")
    matrices = [(confusion_label(path), pd.read_csv(path, index_col=0)) for path in paths]
    recorded_hashes = []
    for path, (_, frame) in zip(paths, matrices):
        values = frame.to_numpy(dtype=float)
        if frame.empty or values.shape[0] != values.shape[1] or not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise SystemExit(f"Confusion matrix must contain finite nonnegative integer counts in a square table: {path}")
        metadata_path = path.with_name("metrics.json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        recorded_hashes.append(metadata.get("split_sha256"))
    fig, axes = plt.subplots(
        1,
        len(matrices),
        figsize=(5.0 * len(matrices), 4.4),
        constrained_layout=True,
        squeeze=False,
    )
    max_value = max(float(frame.to_numpy().max()) for _, frame in matrices)
    image = None
    for ax, (label, frame) in zip(axes[0], matrices):
        values = frame.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=max_value)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                ax.text(
                    column,
                    row,
                    f"{values[row, column]:.0f}",
                    ha="center",
                    va="center",
                    color="white" if values[row, column] > max_value / 2 else "black",
                )
        ax.set_xticks(range(len(frame.columns)), [str(value) for value in frame.columns])
        ax.set_yticks(range(len(frame.index)), [str(value) for value in frame.index])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(label)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Samples", shrink=0.82)
    matrix_status = split_status(pd.DataFrame({"split_sha256": recorded_hashes}))
    verified_shared = matrix_status == "yes" and split_status(comparison) == "yes" and recorded_hashes[0] == comparison["split_sha256"].iloc[0]
    title = "Shared Test Split" if verified_shared else "Recorded Test Splits"
    fig.suptitle(f"Confusion Matrices on {title}", fontsize=13)
    save_figure(fig, output_dir, "model_confusion_matrices", dpi)


def main() -> None:
    args = parse_args()
    comparison_path = args.comparison.expanduser().resolve()
    model_run_dir = args.model_run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not comparison_path.is_file():
        raise SystemExit(f"Model comparison not found: {comparison_path}")
    if not model_run_dir.is_dir():
        raise SystemExit(f"Model run directory not found: {model_run_dir}")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive.")
    comparison = pd.read_csv(comparison_path)
    required = {"model", "test_balanced_accuracy", "test_f1_macro"}
    missing = sorted(required.difference(comparison.columns))
    if missing:
        raise SystemExit(f"Model comparison is missing columns: {missing}")
    if comparison.empty:
        raise SystemExit("Model comparison contains no completed runs.")
    for column in ("test_balanced_accuracy", "test_f1_macro"):
        values = pd.to_numeric(comparison[column], errors="coerce")
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise SystemExit(f"Model comparison contains invalid {column} values.")
    plot_metrics(comparison, output_dir, args.dpi)
    plot_confusion_matrices(model_run_dir, output_dir, args.dpi, comparison)


if __name__ == "__main__":
    main()
