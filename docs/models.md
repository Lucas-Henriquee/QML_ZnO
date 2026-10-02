# Model Execution

The model workflow keeps classical ML, local QML, IBM planning, and real IBM
submission distinct:

```bash
./run_models.sh local --dry-run
./run_models.sh local
./run_models.sh svm --run-id zno_md3_svm_v1
./run_models.sh qsvm --run-id zno_md3_qsvm_v1
```

`local` and `all` both mean a matched SVM and local QSVM comparison using the
same dataset rows and split. The default configuration uses the unique MD-window
dataset, not the 960-row Cartesian expansion.

Review an IBM plan without credentials, network access, or submission:

```bash
./run_models.sh ibm-plan \
  --config configs/models/zno_ibm_plan.yaml \
  --run-id zno_ibm_plan_v1
```

Hardware execution requires the `ibm` method, an explicit backend, and
`--submit-ibm`:

```bash
./run_models.sh ibm \
  --config configs/models/zno_ibm_plan.yaml \
  --backend BACKEND_NAME \
  --submit-ibm \
  --run-id zno_ibm_hardware_v1
```

Do not place tokens, API keys, or IBM instance credentials in YAML. Use a saved
Qiskit Runtime account. Every run is isolated under `results/models/<run-id>/`
and records its dataset hash, configuration, command, versions, and output hashes.

The current dataset contains windows from one MD trajectory per adsorption site.
Results therefore measure within-campaign classification and must not be reported
as generalization to unseen trajectories or materials.
