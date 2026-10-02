# Results

| Location | Contents |
| --- | --- |
| `simulations/<run-id>/` | New simulation campaigns |
| `models/<run-id>/` | New model runs |
| `analyses/<run-id>/` | New tables and plots |
| `workflows/<run-id>/` | Combined-workflow manifests and logs |
| `dft/`, `tddft/`, `md/`, `neb*/`, `pathways/` | Earlier simulation campaigns |
| `svm*/`, `qsvm*/`, `benchmarks/` | Earlier model experiments |
| `datasets/`, `tables/`, `figures/`, `my_analysis/` | Derived data and analyses |

Existing paths are retained because notebooks, scripts, and experiment metadata
refer to them. New runs use distinct IDs and record their inputs and settings.

`simulations/zno_wurtzite_production/` is an interrupted campaign with
different input geometries. Do not use it to reproduce the manuscript results.

The [figure guide](../docs/figures.md) lists the saved manuscript benchmarks.
