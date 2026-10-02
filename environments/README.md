# Project Environments

The project uses two separate Conda environments so that compiled GPAW/MPI
dependencies do not conflict with the QML stack.

| Profile | Conda environment | Purpose |
|---|---|---|
| `simulation` | `qml-zno-simulation` | Geometry generation, DFT, TDDFT, MD, NEB, CO2RR, and the undergraduate simulation notebook |
| `modeling` | `qml-zno-modeling` | Dataset preparation, analysis, plotting, classical ML, local QML, and IBM Runtime execution |

The simulation versions follow the most recent production environment recorded
in this repository. The Qiskit versions follow the metadata stored with the
existing benchmark results. These definitions record the intended software
versions; the actual versions used by each run are stored in its manifest.

## Install

From the repository root:

```bash
./scripts/environment/setup_environment.sh simulation
./scripts/environment/setup_environment.sh modeling
```

Create both environments in sequence:

```bash
./scripts/environment/setup_environment.sh all
```

The command creates a missing environment but does not change an existing one.
To deliberately apply a newer environment file to an existing environment, use:

```bash
./scripts/environment/setup_environment.sh simulation --update
```

Updates never use Conda's `--prune` option, so unrelated packages are not
removed. Use `--dry-run` to inspect creation without installing packages. The
environment files use only `conda-forge`; implicit default channels are disabled
to avoid mixing binary packages from different ecosystems.

## Verify

The checks can run without activating an environment:

```bash
./scripts/environment/check_environment.sh simulation
./scripts/environment/check_environment.sh simulation --mpi
./scripts/environment/check_environment.sh modeling
```

The simulation check validates Python imports, compiled GPAW features, PAW
datasets for the project elements, a one-process dry run, and optionally a
two-process MPI dry run. It does not start a scientific calculation or write to
`results/`.

## Activate

```bash
conda activate qml-zno-simulation
export OMP_NUM_THREADS=1
```

The launcher uses the thread count recorded in the simulation configuration.
The `gpaw-data` package supplies PAW datasets inside the simulation environment.
An existing `GPAW_SETUP_PATH` can select another installation; the environment
check prints the resolved files. Record those paths when reproducing a campaign.

IBM credentials are intentionally not stored in either environment file or in
the repository. Configure them through Qiskit IBM Runtime's credential storage
before running hardware jobs.

The notebook commands in `notebooks/` use the same simulation
profile. The notebook runner refuses to overwrite an existing executed notebook;
pass a new output filename as its second argument for each run.
