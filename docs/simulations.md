# Simulation Scripts

Method implementations live in `scripts/simulations/`. Use the workflow runner instead
of calling these modules directly for normal project runs:

```bash
./run_simulations.sh all --dry-run
./run_simulations.sh dft --run-id zno_dft_01
./run_simulations.sh dft tddft --run-id zno_optical_01
./run_simulations.sh md neb --run-id zno_dynamics_paths_01
```

Prerequisites are added automatically. For example, selecting `tddft` resolves
to `geometry -> dft -> tddft`. Use `--no-dependencies` only when intentionally
continuing from compatible inputs in the same isolated run.

Useful options:

```text
--config PATH           Select the notebook-derived YAML profile or an explicit profile.
--run-id NAME           Assign a readable, unique output directory name.
--resume                Continue a run and skip completed methods.
--dry-run               Validate and display commands without creating files.
--materials ZnO,TiO2    Override the configured material list.
--sites bridge          Override the configured adsorption sites.
--mpi-processes 10      Override the configured process count.
```

Every non-dry run writes to `results/simulations/<run-id>/`, saves an immutable
configuration snapshot, and records commands, parameters, status, paths, and
timestamps in `run_manifest.json`. A run directory is never reused implicitly.
Resume mode verifies the configuration hash and execution context, then refuses
to rerun failed or interrupted methods, protecting partial scientific outputs;
use a new run ID instead.

## Configuration

The YAML follows the notebook's parameter values. Standalone scripts have
different defaults:

| Setting | Standalone default | Notebook / YAML |
| --- | --- | --- |
| DFT steps | 160 | 180 |
| CO₂RR force threshold / steps | 0.06 / 220 | 0.05 / 260 |
| MD steps / log interval | 500 / 10 | 2000 / 20 |
| MD cutoff / k-points | 350 eV / 2×2×1 | 500 eV / 3×3×1 |
| NEB internal images / steps | 3 / 120 | 5 / 300 |
| NEB cutoff / k-points | 350 eV / 2×2×1 | 500 eV / 3×3×1 |

Select the settings associated with the experiment being reproduced. Process
completion alone does not establish optimizer or electronic convergence; inspect
the calculation logs before using the outputs.
