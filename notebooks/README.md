# ZnO simulation notebook

`zno-co2-lucas.ipynb` contains the notebook workflow and its experiment settings.
The Python implementations are in [the repository scripts](../README.md).

## Setup

From this directory:

```bash
./setup_lucas_env.sh
```

This creates the shared `qml-zno-simulation` Conda environment if needed.
It leaves an existing environment unchanged. See the
[environment guide](../environments/README.md).

## Execute

```bash
./run_zno_notebook.sh zno-co2-lucas.ipynb zno-co2-lucas.executed.ipynb
```

This executes the notebook, including its scientific calculations.
Choose a new output filename for subsequent runs; existing executed notebooks
are not overwritten. Input and output filenames are relative to this directory.

For interactive use:

```bash
conda activate qml-zno-simulation
jupyter lab zno-co2-lucas.ipynb
```

`QML_ZNO_SIM_ENV` selects a different environment.
`NB_TIMEOUT` controls the per-cell execution timeout in seconds.
