"""Render archived logical feature maps locally, without hardware access or fitting."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
    "hardware_3q": "results/benchmarks/ibm_shot_comparison/qsvm_ibm/ibm_md3_3q_linear_r1_minmax_512shots_seed42_trial1/run_metadata.json",
    "ideal_6q": "results/benchmarks/qsvm_unique_windows/qsvm/qsvm_md3_6q_linear_r1_minmax_reference_seed42/metrics.json",
}


def load_model():
    path = ROOT / "scripts/models/qml/qsvm_baseline.py"
    spec = importlib.util.spec_from_file_location("paper_circuit_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(circuit, stem):
    """Draw gates from the actual instruction list; abbreviate phase expressions only."""
    n = circuit.num_qubits
    available = [0] * n
    gates = []
    for item in circuit.data:
        name = item.operation.name
        wires = [circuit.find_bit(bit).index for bit in item.qubits]
        if name not in {"h", "p", "cx"}:
            raise ValueError(f"Unexpected gate: {name}")
        span = list(range(min(wires), max(wires) + 1))
        column = max(available[q] for q in span) + 1
        for q in span:
            available[q] = column
        label = "H"
        if name == "p":
            indices = sorted(int(str(p).split("[")[1].split("]")[0]) for p in item.operation.params[0].parameters)
            label = (rf"$P(2x_{indices[0]})$" if len(indices) == 1
                     else rf"$P(\phi_{{{indices[0]}{indices[1]}}})$")
        gates.append((column, name, wires, label))
    width = max(available)
    fig, ax = plt.subplots(figsize=(max(9, width * .72), n * .65 + .5))
    for q in range(n):
        y = n - 1 - q
        ax.plot([.35, width + .65], [y, y], color="#444444", lw=1)
        ax.text(.18, y, rf"$q_{q}$", ha="right", va="center", fontsize=11)
    for column, name, wires, label in gates:
        y = n - 1 - wires[-1]
        if name == "cx":
            control = n - 1 - wires[0]
            ax.plot([column, column], [y, control], color="black", lw=1.5)
            ax.plot(column, control, "o", color="black", ms=5)
            ax.plot(column, y, "o", mfc="white", mec="black", ms=14)
            ax.plot([column-.09, column+.09], [y, y], color="black")
            ax.plot([column, column], [y-.12, y+.12], color="black")
        else:
            ax.add_patch(Rectangle((column-.43, y-.24), .86, .48, facecolor="white", edgecolor="black", zorder=3))
            ax.text(column, y, label, ha="center", va="center", fontsize=10, zorder=4)
    ax.set(xlim=(-.6, width+1), ylim=(-.4, n-.6))
    ax.axis("off")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def fidelity_schematic(stem):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for q in range(3):
        ax.plot([0, 6], [q, q], color="#444444")
        ax.text(-.2, q, rf"$|0\rangle_{{q_{2-q}}}$", ha="right", va="center")
        ax.text(5.5, q, "M", ha="center", va="center", bbox=dict(facecolor="white", edgecolor="black"))
    for x, label in [(1, r"$U(x)$"), (3, r"$U^{\dagger}(y)$")]:
        ax.add_patch(Rectangle((x, -.3), 1.4, 2.6, facecolor="white", edgecolor="black", zorder=3))
        ax.text(x+.7, 1, label, ha="center", va="center", fontsize=17, zorder=4)
    ax.set(xlim=(-.8, 6.2), ylim=(-.8, 2.6))
    ax.axis("off")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(stem.with_suffix("."+suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    model = load_model()
    reconstructed = []
    for name, relative in SOURCES.items():
        source = ROOT / relative
        metadata = json.loads(source.read_text())
        circuit, encoding = model.build_feature_map(metadata["n_features"], metadata["n_qubits"], metadata["reps"], metadata["entanglement"])
        fingerprint = model.feature_map_fingerprint(circuit)
        if fingerprint != metadata["feature_map_fingerprint"]:
            raise ValueError(f"Archived fingerprint mismatch: {name}")
        archived = (source.parent / "feature_map.txt").read_text().strip()
        if str(circuit.draw(output="text", fold=-1)).strip() != archived:
            raise ValueError(f"Archived drawing mismatch: {name}")
        reconstructed.append((name, circuit, metadata, fingerprint, relative))
    output.mkdir(parents=True)
    provenance = []
    for name, circuit, metadata, fingerprint, relative in reconstructed:
        render(circuit, output / ("qsvm_feature_map_" + name))
        (output / (name + ".txt")).write_text(str(circuit.draw(output="text", fold=-1))+"\n")
        provenance.append(dict(source=relative, fingerprint=fingerprint, verified_archived_text=True, qubits=circuit.num_qubits, depth=circuit.depth(), gates=circuit.size(), features=metadata["features"], scaler=metadata["scaler"], reps=metadata["reps"], entanglement=metadata["entanglement"]))
    fidelity_schematic(output / "qsvm_compute_uncompute_schematic")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"Verified both archived fingerprints and text drawings. Figures saved in {output}")


if __name__ == "__main__":
    main()
