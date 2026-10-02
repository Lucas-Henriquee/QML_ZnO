# Isolated Simulation Runs

Each directory here is created by `scripts/workflows/run_simulations.sh` and is
identified by a unique run ID.

```text
<run-id>/
├── config.yaml             # configuration snapshot used by the run
├── run_manifest.json       # commands, versions, paths, timestamps, and status
├── logs/                   # one combined log per method
├── data/geometries/        # run-specific structural inputs
├── dft/
├── tddft/
├── pathways/
├── md/
└── neb/
```

Some method directories appear only when that method is selected. Generated run
directories are intentionally excluded from Git because GPAW outputs can be very
large. This README remains tracked so the output convention is documented.

Never manually combine files from runs with different configuration hashes.
Use the manifest to associate outputs with their exact inputs and commands.

