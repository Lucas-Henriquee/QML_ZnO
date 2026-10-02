# ZnO–CO₂ adsorption: simulations, ML, and QML

Code and results for comparing CO₂ adsorption at the top-metal, top-oxygen,
and bridge sites on ZnO. Physical calculations include DFT, TDDFT, CO₂RR,
AIMD, and NEB. Classical SVM and quantum-kernel SVM models classify adsorption
sites using AIMD-window descriptors; selected models were also tested on IBM
Quantum hardware.

## Setup

From the repository root:

```bash
./scripts/environment/setup_environment.sh all
./scripts/environment/check_environment.sh simulation --mpi
./scripts/environment/check_environment.sh modeling
```

See [environment setup](environments/README.md) for details.

## Run

Preview each stage before execution:

```bash
./run_simulations.sh all 
./run_datasets.sh all 
./run_models.sh local 
./run_analysis.sh all 
```

The launchers use the profiles in `configs/`. Use `--help` for stage options,
`--config PATH` for another profile, or `--run-id NAME` to name an output run.
See [reproduction](docs/reproduction.md) for the complete `run_all.sh` workflow.

IBM hardware runs require an account and explicit submission. An account-free
preview is available with:

```bash
./run_models.sh ibm-plan --config configs/models/zno_ibm_plan.yaml
```

See [model instructions](docs/models.md) before submitting hardware jobs.

## Repository guide

`configs/` holds run settings; `scripts/` holds simulations, dataset builders,
models, and analysis; `data/` holds inputs and datasets; `results/` holds saved
outputs. Figures are in `figures/`, and detailed instructions are in `docs/`.

## Tests

```bash
conda run -n qml-zno-simulation python -m unittest discover -s tests -v
```

Tests check the code and workflows, not scientific convergence.
