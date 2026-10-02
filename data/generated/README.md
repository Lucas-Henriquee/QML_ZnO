# Generated Dataset Runs

Each subdirectory is an isolated dataset generation run created by
`scripts/workflows/run_datasets.sh`.

```text
<run-id>/
├── config.yaml
├── run_manifest.json
├── logs/
├── tddft/
├── multitech/
└── qml_features/
```

Generated run directories are excluded from Git. The manifest provides the
lineage needed to reproduce a dataset from its original scientific inputs.
