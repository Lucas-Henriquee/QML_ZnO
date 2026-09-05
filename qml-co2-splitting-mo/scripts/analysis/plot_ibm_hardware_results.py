from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSICAL_COLOR = "tab:blue"
QUANTUM_COLOR = "tab:orange"

CLASSICAL_MARKER = "o"   
QUANTUM_MARKER = "s"     

ROOT = Path(__file__).resolve().parents[2]

QUBIT_SUMMARY = (
    ROOT
    / "results/benchmarks/ibm_qubits/20260822_095337/benchmark_summary.csv"
)

SHOT_SUMMARY = (
    ROOT
    / "results/benchmarks/ibm_shots/20260822_095836/benchmark_summary.csv"
)

SHOT_SENSITIVITY = (
    ROOT
    / "results/benchmarks/ibm_shots/20260822_095836/shot_sensitivity.csv"
)

SVM_METRICS = (
    ROOT
    / "results/benchmarks/final_matched_n12/"
    / "svm_md3_rbf_power_C10_seed42/metrics.json"
)

OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SPLIT = (
    "68d8173a67ebaf84012f375992e537ba44f32cbbcc0d19a881325ee1285d9d8e"
)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def main():
    for path in [
        QUBIT_SUMMARY,
        SHOT_SUMMARY,
        SHOT_SENSITIVITY,
        SVM_METRICS,
    ]:
        require_file(path)

    qubit_df = pd.read_csv(QUBIT_SUMMARY)
    shot_df = pd.read_csv(SHOT_SUMMARY)
    shot_sensitivity = pd.read_csv(SHOT_SENSITIVITY)

    with open(SVM_METRICS, "r", encoding="utf-8") as handle:
        svm_metrics = json.load(handle)

    # Validate IBM qubit sweep

    hw_qubits = qubit_df[
        qubit_df["model"].astype(str) == "QSVM_IBM"
    ].copy()

    hw_qubits["n_qubits"] = hw_qubits["n_qubits"].astype(int)

    hw_qubits = hw_qubits.sort_values("n_qubits")

    expected_qubits = [3, 6, 9, 12]

    observed_qubits = hw_qubits["n_qubits"].tolist()

    if observed_qubits != expected_qubits:
        raise ValueError(
            f"Unexpected hardware qubit sweep: {observed_qubits}"
        )

    if set(hw_qubits["split_sha256"].astype(str)) != {EXPECTED_SPLIT}:
        raise ValueError("Qubit sweep does not use the expected split.")

    if set(hw_qubits["shots"].astype(int)) != {512}:
        raise ValueError("Qubit sweep is not consistently using 512 shots.")

    # Validate IBM shot sweep

    hw_shots = shot_df[
        shot_df["model"].astype(str) == "QSVM_IBM"
    ].copy()

    hw_shots["shots"] = hw_shots["shots"].astype(int)
    hw_shots = hw_shots.sort_values("shots")

    expected_shots = [256, 512, 1024]

    if hw_shots["shots"].tolist() != expected_shots:
        raise ValueError(
            f"Unexpected shot sweep: {hw_shots['shots'].tolist()}"
        )

    if set(hw_shots["n_qubits"].astype(int)) != {3}:
        raise ValueError("Shot sweep is not consistently 3 qubits.")

    if set(hw_shots["split_sha256"].astype(str)) != {EXPECTED_SPLIT}:
        raise ValueError("Shot sweep does not use the expected split.")

    # Ideal QSVM for final matched comparison

    ideal = shot_df[
        shot_df["model"].astype(str) == "QSVM_IDEAL"
    ].copy()

    if len(ideal) != 1:
        raise ValueError(
            f"Expected one ideal QSVM reference, found {len(ideal)}."
        )

    ideal = ideal.iloc[0]

    if int(ideal["n_qubits"]) != 3:
        raise ValueError("Ideal matched reference is not 3 qubits.")

    if str(ideal["split_sha256"]) != EXPECTED_SPLIT:
        raise ValueError("Ideal QSVM split does not match IBM split.")

    # Classical SVM for final matched comparison

    if svm_metrics["model"] != "SVM":
        raise ValueError("Unexpected model in SVM metrics file.")

    if svm_metrics["feature_set"] != "md3":
        raise ValueError("Matched SVM is not using md3.")

    if svm_metrics["kernel"] != "rbf":
        raise ValueError("Matched SVM is not using RBF.")

    if svm_metrics["scaler"] != "power":
        raise ValueError("Matched SVM is not using power scaling.")

    if float(svm_metrics["C"]) != 10.0:
        raise ValueError("Matched SVM is not using C=10.")

    if svm_metrics["split_sha256"] != EXPECTED_SPLIT:
        raise ValueError("SVM split does not match IBM split.")

    # Nominal IBM comparison: 512 shots, 3 qubits.
    ibm_512 = hw_shots[
        hw_shots["shots"] == 512
    ].iloc[0]

    # Export consolidated values used by the figure

    final_comparison = pd.DataFrame(
        {
            "model": [
                "SVM",
                "Ideal QSVM (3q)",
                "IBM QSVM (3q)",
            ],
            "test_balanced_accuracy": [
                float(
                    svm_metrics["test_metrics"]["balanced_accuracy"]
                ),
                float(ideal["test_balanced_accuracy"]),
                float(ibm_512["test_balanced_accuracy"]),
            ],
            "test_f1_macro": [
                float(svm_metrics["test_metrics"]["f1_macro"]),
                float(ideal["test_f1_macro"]),
                float(ibm_512["test_f1_macro"]),
            ],
        }
    )

    final_comparison.to_csv(
        OUTPUT_DIR / "ml_04_ibm_final_comparison.csv",
        index=False,
    )

    print("\nHardware qubit sweep:")
    print(
        hw_qubits[
            [
                "n_qubits",
                "test_balanced_accuracy",
                "train_transpiled_depth_mean",
                "train_transpiled_2q_mean",
            ]
        ].to_string(index=False)
    )

    print("\nShot sweep:")
    print(
        hw_shots[
            [
                "shots",
                "test_balanced_accuracy",
                "test_f1_macro",
                "quantum_time_s",
            ]
        ].to_string(index=False)
    )

    print("\nFinal matched comparison:")
    print(final_comparison.to_string(index=False))

    # Figure

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 7.5),
    )

    # (a) Hardware classification vs qubit count

    ax = axes[0, 0]

    qubits = hw_qubits["n_qubits"].to_numpy()
    qubit_ba = hw_qubits[
        "test_balanced_accuracy"
    ].to_numpy()

    ax.scatter(
        qubits,
        qubit_ba,
        s=55,
        color=QUANTUM_COLOR,
        marker=QUANTUM_MARKER,
        label="IBM QSVM",
    )
    ax.legend(frameon=False, loc="upper left")

    ax.set_xticks(qubits)
    ax.set_ylim(0.25, 0.80)

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Test balanced accuracy")
    ax.set_title("(a) Hardware classification")

    # (b) Circuit complexity

    ax = axes[0, 1]

    depth = hw_qubits[
        "train_transpiled_depth_mean"
    ].to_numpy()

    gates_2q = hw_qubits[
        "train_transpiled_2q_mean"
    ].to_numpy()

    ax.scatter(
        qubits,
        depth,
        marker=CLASSICAL_MARKER,
        color=CLASSICAL_COLOR,
        s=55,
        label="Mean transpiled depth",
    )

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Mean transpiled depth")
    ax.set_xticks(qubits)

    ax2 = ax.twinx()

    ax2.scatter(
        qubits,
        gates_2q,
        marker=QUANTUM_MARKER,
        color=QUANTUM_COLOR,
        s=50,
        label="Mean two-qubit gates",
    )

    ax2.set_ylabel("Mean two-qubit gates")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        loc="upper left",
    )

    ax.set_title("(b) Circuit complexity")

    # (c) Shot sensitivity

    ax = axes[1, 0]

    shots = hw_shots["shots"].to_numpy()

    shot_ba = hw_shots[
        "test_balanced_accuracy"
    ].to_numpy()

    quantum_time = hw_shots[
        "quantum_time_s"
    ].to_numpy()

    ax.scatter(
        shots,
        shot_ba,
        marker=CLASSICAL_MARKER,
        color=CLASSICAL_COLOR,
        s=55,
        label="Balanced accuracy",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(shots)
    ax.set_xticklabels(
        [str(value) for value in shots]
    )

    ax.set_ylim(0.25, 0.80)

    ax.set_xlabel("Shots")
    ax.set_ylabel("Test balanced accuracy")

    ax2 = ax.twinx()

    ax2.scatter(
        shots,
        quantum_time,
        marker=QUANTUM_MARKER,
        color=QUANTUM_COLOR,
        s=50,
        label="Quantum execution time",
    )

    ax2.set_ylabel("Reported quantum execution time (s)")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        loc="upper left",
    )

    ax.set_title("(c) Shot sensitivity")

    # (d) Final matched N=12 comparison

    ax = axes[1, 1]

    model_labels = [
        "SVM",
        "Ideal\nQSVM",
        "IBM\nQSVM",
    ]

    final_ba = final_comparison[
        "test_balanced_accuracy"
    ].to_numpy()

    x_models = np.arange(len(model_labels))

    # SVM
    ax.scatter(
        [x_models[0]],
        [final_ba[0]],
        s=70,
        color=CLASSICAL_COLOR,
        marker=CLASSICAL_MARKER,
        label="SVM",
    )

    # QSVM ideal + QSVM IBM
    ax.scatter(
        [x_models[1], x_models[2]],
        [final_ba[1], final_ba[2]],
        s=70,
        color=QUANTUM_COLOR,
        marker=QUANTUM_MARKER,
        label="QSVM",
    )

    ax.legend(frameon=False, loc="upper left")

    ax.set_xticks(x_models)
    ax.set_xticklabels(model_labels)

    ax.set_ylim(0.20, 0.80)

    ax.set_ylabel("Test balanced accuracy")
    ax.set_title("(d) Matched comparison ($N_{test}=3$)")

    for x_value, y_value in zip(x_models, final_ba):
        ax.text(
            x_value,
            y_value + 0.025,
            f"{y_value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    png_path = (
        OUTPUT_DIR / "ml_04_ibm_hardware_results.png"
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