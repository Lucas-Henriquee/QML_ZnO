# Run the local workflow

Use the existing simulation outputs to rebuild analysis tables, datasets,
local models, and model figures. The commands below run from the project root.
Install the [two environments](../environments/README.md) first.

## Preview

```bash
./run_all.sh --run-id local_reproduction_01 \
  --source-results results \
  --dataset-config configs/datasets/zno_existing_results.yaml \
  --model-config configs/models/zno_local.yaml \
  --analysis-config configs/analysis/zno_current.yaml \
  --model-dataset multitech/md_window_unique_dataset.csv \
  --dry-run
```

Replace `--dry-run` with `--execute` to run the plan.
Choose a new run ID for each execution. The coordinator stops on a failed stage.

| Stage | Input | Output |
| --- | --- | --- |
| Scientific analysis | Selected simulation results | `results/analyses/ID_science/science/` |
| Datasets | Same results and generated ranking | `data/generated/ID/` |
| Local SVM and QSVM | Selected generated dataset | `results/models/ID/` |
| Model analysis | That model run | `results/analyses/ID_models/models/` |

The coordinator honors the output roots in the selected YAML files.
Commands, configuration hashes, exit codes, and logs are recorded in
`results/workflows/ID/`. Stage manifests record input and output hashes.

## Include simulations

Replace `--source-results results` with:

```text
--simulate --simulation-config configs/simulations/zno_production.yaml
```

This starts new scientific calculations using the supplied settings.
The [simulation guide](simulations.md) describes dependencies and MPI.
The simulation configuration follows the notebook; standalone defaults differ.

## Reproduce a particular experiment

The supplied local profile uses its own settings: seed 42, a 30% test split,
30 samples, and a three-qubit QSVM with `power_minmax` preprocessing.
It is not the selected six-qubit, five-seed comparison in the manuscript.
Use the configuration, split, and metadata associated with the desired result.

The [figure guide](figures.md) identifies saved benchmark sources.
The [dataset reference](datasets_reference.md) explains the input columns.
Archive the source commit, configurations, environment versions, and selected
inputs with a published result. `requirements.txt` is not an environment lockfile.

## Scope

Rebuilding tables and figures from saved outputs does not repeat the underlying
simulations or hardware measurements. A fresh simulation campaign requires
convergence checks, and a new hardware run can differ because of sampling noise
and device calibration.

IBM execution is separate through [the model launcher](models.md); it is never
included in `run_all.sh`.
