# Repository map

`run_all.sh` coordinates the four stage-specific public entry points.
Their Python implementations live in `scripts/workflows/`.

```text
QML_ZnO/
├── run_all.sh
├── run_simulations.sh
├── run_datasets.sh
├── run_models.sh
├── run_analysis.sh
├── configs/
│   ├── simulations/
│   ├── datasets/
│   ├── models/
│   └── analysis/
├── environments/
├── scripts/
│   ├── environment/
│   ├── simulations/       # dft, tddft, md, neb, co2rr
│   ├── datasets/
│   ├── models/            # ml, qml, qml_ibm, benchmarks
│   ├── analysis/
│   ├── visualization/
│   ├── common/
│   ├── legacy/
│   └── workflows/
├── data/
│   ├── geometries/
│   ├── generated/
│   └── qml_datasets/
├── results/
│   ├── simulations/
│   ├── models/
│   ├── analyses/
│   └── benchmarks/       # saved SVM, QSVM, and IBM comparisons
├── figures/
├── notebooks/
├── docs/
└── tests/
```

The active notebook and its launchers are in `notebooks/`.
Keep Python module calls rooted here, for example
`python -m scripts.simulations.dft.run_dft --help`.

## Existing campaigns

Saved campaigns have descriptive names based on their method and comparison.
Repeated runs use numbered suffixes. Scientific settings are preserved;
path references use the current locations.
Generated run directories are ignored by Git.

The older `scripts/workflows/run_pipeline.py` operates on shared historical
paths. Use the root launchers for new work. Specialized plotting scripts such as
the archived `analysis_plot2.py`, `plot_ml_results.py`, and the benchmark figures describe
specific earlier campaigns; they are not the generic analysis entry point.

Usage guides are centralized here. Former workflow shell paths are compatibility
wrappers, so existing terminal commands continue to work.

See [reproduction](reproduction.md) for the active route and
[historical code](../scripts/legacy/README.md) for the moved implementations.
