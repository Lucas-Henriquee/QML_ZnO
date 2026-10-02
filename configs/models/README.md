# Model Configuration

- `zno_local.yaml` is the default local SVM/QSVM comparison. It uses the 30-row
  unique MD-window dataset and a shared stratified split.
- `zno_ibm_plan.yaml` limits the matched local/IBM study to 12 balanced samples
  so its circuit workload can be reviewed before submission.

IBM credentials are never stored in these files. A real hardware execution
requires the `ibm` method, the explicit `--submit-ibm` flag, and a backend name.
An `ibm-plan` only calculates and records the command and expected workload; it
does not contact IBM Quantum.

The dataset can be replaced without editing YAML:

```bash
./run_models.sh local \
  --dataset data/generated/another_run/multitech/md_window_unique_dataset.csv \
  --run-id another_local_comparison
```
