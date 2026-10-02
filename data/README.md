# Data

- `geometries/`: structural inputs used by standalone scripts.
- `generated/<run-id>/`: datasets produced by the dataset launcher.
- `qml_datasets/tddft/`: saved TDDFT tables.

Simulation-launcher runs keep their geometries inside the run directory.
Dataset manifests record source files and hashes. Historical datasets also
remain under `results/datasets/`, where existing experiment records reference them.

See [dataset generation](../docs/datasets.md) and
[feature definitions](../docs/datasets_reference.md).
