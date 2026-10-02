# Model Runs

Each subdirectory is an isolated model workflow created by
`scripts/workflows/run_models.sh`.

```text
<run-id>/
├── config.yaml
├── run_manifest.json
├── logs/models.log
└── benchmark/<suite-timestamp>/
    ├── splits/
    ├── svm/
    ├── qsvm/
    ├── qsvm_ibm/
    ├── benchmark_summary.csv
    └── engineering_comparison.csv
```

Some model directories appear only when selected. Generated runs are excluded
from Git, while this README preserves the output convention.
