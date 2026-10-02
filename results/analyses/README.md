# Isolated Analysis Runs

Each subdirectory is produced by `scripts/workflows/run_analysis.sh`.

```text
<run-id>/
├── config.yaml
├── run_manifest.json
├── logs/
├── science/
│   ├── screening/
│   ├── co2_reduction/
│   └── figures/
└── models/
    ├── tables/
    └── figures/
```

Generated analysis directories are excluded from Git. This README documents the
layout while existing historical analyses remain unchanged.

See [the verification report](../../docs/validation.md) for tested runs.
Older numbered runs are retained with their original manifests.
