# CO₂ adsorption on ZnO

Simulation, dataset preparation, SVM/QSVM classification, and analysis of
top-metal, top-oxygen, and bridge adsorption configurations. The simulation
code also supports TiO₂ and CeO₂; the saved studies focus on ZnO.

## Setup

Run these commands from the repository root:

```bash
./scripts/environment/setup_environment.sh all
./scripts/environment/check_environment.sh simulation --mpi
./scripts/environment/check_environment.sh modeling
```

The launchers use `qml-zno-simulation` for simulations and
`qml-zno-modeling` for datasets, models, and analysis.
See [environment setup](environments/README.md) for installation details.

## Run a stage

| Stage | Preview command | Configuration |
| --- | --- | --- |
| Simulations | `./run_simulations.sh all ` | `configs/simulations/zno_production.yaml` |
| Datasets | `./run_datasets.sh all ` | `configs/datasets/zno_existing_results.yaml` |
| Local models | `./run_models.sh local ` | `configs/models/zno_local.yaml` |
| Analysis | `./run_analysis.sh all ` | `configs/analysis/zno_current.yaml` |

Use `--run-id NAME` for a new output
directory and `--config PATH` to select settings. Each launcher has `--help`.
When no run name is supplied, outputs use numbered names such as
`simulation_01`, `dataset_01`, `local_01`, and `analysis_01`.
The next available number is selected; existing results are never replaced.
Input paths in the supplied profiles refer to existing local campaigns;
use `--dataset`, `--source-results`, or `--model-run` to select other inputs.

[Run the complete local workflow](docs/reproduction.md) with `run_all.sh`.
It takes explicit configurations and prints a plan until `--execute` is supplied.

IBM planning is available without an account:

```bash
./run_models.sh ibm-plan --config configs/models/zno_ibm_plan.yaml
```

Hardware execution requires `ibm --submit-ibm --backend NAME`.
See [models](docs/models.md).

## Layout

| Directory | Contents |
| --- | --- |
| `configs/` | Execution parameters and input paths |
| `environments/` | Conda environment definitions |
| `scripts/simulations/` | DFT, TDDFT, MD, NEB, CO₂RR, and geometry builders |
| `scripts/datasets/` | Dataset builders |
| `scripts/models/` | Classical SVM, ideal QSVM, IBM, and benchmarks |
| `scripts/analysis/` | Tables and numerical summaries |
| `scripts/visualization/` | Plotting and circuit diagrams |
| `scripts/workflows/` | Stage coordination |
| `scripts/common/` | Shared paths and utilities |
| `scripts/legacy/` | Earlier campaign-specific plotting implementations |
| `data/` | Geometries and datasets |
| `results/` | Saved campaigns and new run outputs |
| `figures/` | Selected manuscript figures |
| `notebooks/` | ZnO notebook and its launchers |
| `docs/` | Usage and testing guides |
| `tests/` | Preservation and workflow checks |

New runs write to `results/simulations/ID`, `data/generated/ID`,
`results/models/ID`, and `results/analyses/ID`. Their manifests record inputs,
commands, configurations, and status. Saved campaigns use descriptive names.

## Tests

```bash
conda run -n qml-zno-simulation python -m unittest discover -s tests -v
```

Tests cover syntax, launcher arguments, dataset file discovery, and configuration
consistency. They do not establish convergence of the scientific calculations.
The [simulation guide](docs/simulations.md) lists the differences between the
notebook configuration and standalone defaults.

[Documentation index](docs/README.md) ·
[Reproduction](docs/reproduction.md) ·
[Dataset reference](docs/datasets_reference.md) ·
[Manual testing](docs/testing.md)
