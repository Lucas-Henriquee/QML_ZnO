# Analysis and Figures

Use the dedicated analysis workflow from the project root:

```bash
./run_analysis.sh all --dry-run
./run_analysis.sh all
./run_analysis.sh science --run-id zno_science_v1
./run_analysis.sh models --run-id zno_models_analysis_v1
```

Available stages are `screening`, `co2_reduction`, `science_figures`, and
`model_analysis`. The `science` alias selects the first three; `models` selects
the model comparison; `all` selects every stage. Figure generation automatically
adds its CO2-reduction table dependency.

Every run is isolated below `results/analyses/<run-id>/` and contains its
configuration snapshot, input and output hashes, logs, tables, reports, and
figures. Preserved scientific and model results are read-only inputs.

Model figures are descriptive. The current single-split, single-trajectory
comparison must not be interpreted as quantum advantage or as generalization to
unseen materials and trajectories.
