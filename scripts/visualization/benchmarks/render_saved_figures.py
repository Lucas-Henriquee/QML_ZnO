"""Rebuild the manuscript benchmark figures from saved runs in a new directory."""

import argparse
import importlib
from pathlib import Path

PLOTS = {
    "generalization": "plot_ml_results",
    "qubits": "plot_qsvm_qubit_sweep",
    "configuration": "plot_qsvm_circuit_config",
    "ibm": "plot_ibm_hardware_results",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; existing directories are not overwritten.")
    parser.add_argument("--plots", nargs="+", choices=list(PLOTS), default=list(PLOTS))
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in dict.fromkeys(args.plots):
        module = importlib.import_module("scripts.visualization.benchmarks." + PLOTS[name])
        module.OUTPUT_DIR = output
        module.main()
    print(f"Saved benchmark figures in {output}")


if __name__ == "__main__":
    main()
