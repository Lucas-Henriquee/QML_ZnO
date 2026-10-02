# Dataset Configuration

Dataset generation is configured independently from simulations and models.
`zno_existing_results.yaml` reads the preserved ZnO outputs and never writes
back into `results/`.

Relative input paths are resolved below `input.results_dir`. This makes the same
configuration usable with a new isolated simulation run:

```bash
./run_datasets.sh all \
  --source-results results/simulations/zno_run_01 \
  --ranking results/simulations/zno_run_01/ranked_candidates.csv \
  --run-id zno_run_01_datasets
```

The ranking file is required only for `qml_features`. It may be supplied later,
after the analysis stage has produced a ranked candidate table.
