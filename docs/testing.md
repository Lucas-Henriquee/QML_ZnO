# Manual testing

Run from the repository root. These commands use the supplied configurations
without changing parameters. Use a new run ID or output directory for each
execution; existing outputs are not overwritten.

## 1. Environments and regression tests

```bash
cd /home/lucas/Documents/GitHub_Repository/QML_ZnO
./scripts/environment/check_environment.sh simulation --mpi
./scripts/environment/check_environment.sh modeling
conda run --no-capture-output -n qml-zno-simulation python -m unittest discover -s tests -v
```

Environment checks should finish without errors; the test suite should report
`OK`. The MPI check initializes GPAW in dry-run mode, not a production calculation.
If an environment is missing, install it with
`./scripts/environment/setup_environment.sh all`, then repeat the checks.

## 2. Validate commands and inputs

```bash
./run_simulations.sh all --dry-run
./run_datasets.sh all --dry-run
./run_models.sh local --dry-run
./run_models.sh ibm-plan --config configs/models/zno_ibm_plan.yaml --dry-run
./run_analysis.sh all --dry-run
```

Every command should report successful validation. These checks do not execute
simulations, train models, or submit hardware jobs.

## 3. Execute the local workflow using saved simulation outputs

Preview the combined workflow:

```bash
./run_all.sh --run-id manual_local_01 \
  --source-results results \
  --dataset-config configs/datasets/zno_existing_results.yaml \
  --model-config configs/models/zno_local.yaml \
  --analysis-config configs/analysis/zno_current.yaml \
  --model-dataset multitech/md_window_unique_dataset.csv \
  --dry-run
```

Then execute the same workflow:

```bash
./run_all.sh --run-id manual_local_01 \
  --source-results results \
  --dataset-config configs/datasets/zno_existing_results.yaml \
  --model-config configs/models/zno_local.yaml \
  --analysis-config configs/analysis/zno_current.yaml \
  --model-dataset multitech/md_window_unique_dataset.csv \
  --execute
```

This generates scientific tables, datasets, SVM/QSVM results, and model figures.
It does not repeat physical simulations or IBM measurements. Check that
`results/workflows/manual_local_01/run_manifest.json` reports `completed`.
Consult the stage logs if it reports `failed`.

## 4. Test figure and circuit generation from saved experiments

```bash
conda run --no-capture-output -n qml-zno-modeling python -m scripts.visualization.benchmarks.render_saved_figures \
  --output-dir results/analyses/manual_benchmark_figures_01
conda run --no-capture-output -n qml-zno-modeling python -m scripts.visualization.benchmarks.export_paper_circuits \
  --output-dir results/analyses/manual_circuits_01
./run_models.sh ibm-plan --config configs/models/zno_ibm_plan.yaml --run-id manual_ibm_plan_01
```

Inspect the generated images. IBM planning requires no account or submission;
it does not validate hardware access. Real IBM execution is described separately
in [the model guide](models.md).

## 5. Run physical simulations

This is the expensive test: it launches all configured methods at their original
settings. It can take substantially longer than the checks above.

```bash
./run_simulations.sh all --config configs/simulations/zno_production.yaml --run-id manual_simulation_01
```

Check `results/simulations/manual_simulation_01/run_manifest.json` and the method
logs. A successful process is not proof of electronic or optimizer convergence.
Review convergence messages, MD outputs, and NEB paths before using new results.

To test downstream processing with that new campaign, repeat the command in
section 3 with `--source-results results/simulations/manual_simulation_01` and
a new run ID, such as `manual_new_campaign_01`.

The local model profile is not identical to every selected manuscript experiment.
These tests verify the configured workflow, not full reproduction of all published
comparisons. See [reproduction](reproduction.md).
