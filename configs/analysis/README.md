# Analysis Configuration

`zno_current.yaml` connects the preserved scientific results and the current
isolated local model run. Analysis output is always written to a new directory
below `results/analyses/`.

Input roots can be overridden without editing YAML:

```bash
./run_analysis.sh all \
  --scientific-results results/simulations/zno_run_01 \
  --model-run results/models/another_local_run \
  --run-id another_analysis
```

The scientific and model analyses remain separate. A missing model run does not
prevent a science-only execution, and model analysis does not modify simulation
outputs.
