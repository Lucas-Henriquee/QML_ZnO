# Dataset Generation

See [dataset and feature definitions](datasets_reference.md), including `static6`,
and [reproduction](reproduction.md) for input/output routing across stages.

Use the dedicated dataset runner from the project root:

```bash
./run_datasets.sh all --dry-run
./run_datasets.sh all
./run_datasets.sh tddft multitech --run-id zno_descriptors_v1
./run_datasets.sh qml_features --run-id zno_qml_features_v1
```

The three methods are independent:

- `tddft` creates summary-level and transition-level TDDFT datasets.
- `multitech` combines DFT, TDDFT, CO2RR, and MD descriptors.
- `qml_features` normalizes a ranked candidate table for QML models.

Every run is written below `data/generated/<run-id>/`. The manifest records the
SHA-256 hash of every consumed input, the exact commands, package versions, and
method status. Existing datasets and simulation results are read-only inputs.

The expanded multitech table pairs TDDFT transitions with MD windows from the
same material/site. The unique-window dataset keeps one row per window.
Windows from a single trajectory may remain correlated; the supplied model
configuration uses its recorded stratified split and does not claim independent
trajectory validation.

MD collection accepts both `md_summary.csv` and historical `md_summary_*.csv`
names. There must be one matching file per material/site directory. Ambiguous
inputs are reported instead of combining or choosing trajectories implicitly.

Use `--source-results` to consume a specific isolated simulation run and
`--ranking` to select its ranked candidate table. A run ID cannot be reused.
