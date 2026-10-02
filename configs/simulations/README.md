# Simulation settings

`zno_production.yaml` contains the parameters from the supplied ZnO notebook.
The standalone Python scripts have separate defaults; see
[simulation guide](../../docs/simulations.md).

Shared materials, sites, MPI settings, and output paths are defined once.
Each method has its own section. NEB uses `idpp` interpolation and five internal
images (seven including endpoints), with ten MPI processes.

Preview the configured commands from the project directory:

```bash
./run_simulations.sh all --dry-run
```

The filename identifies a configuration, not evidence of convergence.
New outputs go to `results/simulations/<run-id>/`.
