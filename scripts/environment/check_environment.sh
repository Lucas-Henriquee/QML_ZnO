#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROFILE=""
ENV_OVERRIDE=""
CHECK_MPI=0

usage() {
  cat <<'EOF'
Usage: check_environment.sh PROFILE [OPTIONS]

Validate a project Conda environment without running a scientific simulation.

Profiles:
  simulation    Validate GPAW, ASE, PAW datasets, and compiled features
  modeling      Validate dataset, ML, QML, and IBM Runtime imports

Options:
  --env NAME    Check a non-default Conda environment
  --mpi         Add a two-process GPAW MPI dry run (simulation only)
  -h, --help    Show this help message
EOF
}

fail() {
  echo "[error] $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    simulation|modeling)
      [[ -z "${PROFILE}" ]] || fail "Only one profile may be selected."
      PROFILE="$1"
      shift
      ;;
    --env)
      [[ $# -ge 2 ]] || fail "--env requires a value."
      ENV_OVERRIDE="$2"
      shift 2
      ;;
    --mpi)
      CHECK_MPI=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

[[ -n "${PROFILE}" ]] || fail "Select simulation or modeling."
command -v conda >/dev/null 2>&1 || fail "Conda was not found in PATH."

case "${PROFILE}" in
  simulation) DEFAULT_ENV="qml-zno-simulation" ;;
  modeling) DEFAULT_ENV="qml-zno-modeling" ;;
esac
ENVIRONMENT_NAME="${ENV_OVERRIDE:-${DEFAULT_ENV}}"
CACHE_ROOT="${TMPDIR:-/tmp}/qml-zno-environment-check"
mkdir -p "${CACHE_ROOT}/matplotlib"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"

if [[ "${PROFILE}" != "simulation" && "${CHECK_MPI}" == "1" ]]; then
  fail "--mpi is only valid for the simulation profile."
fi

if ! conda run -n "${ENVIRONMENT_NAME}" python --version >/dev/null 2>&1; then
  fail "Conda environment '${ENVIRONMENT_NAME}' does not exist or is not usable."
fi

run_in_environment() {
  conda run --no-capture-output -n "${ENVIRONMENT_NAME}" "$@"
}

echo "[check] Profile: ${PROFILE}"
echo "[check] Conda environment: ${ENVIRONMENT_NAME}"

if [[ "${PROFILE}" == "simulation" ]]; then
  echo "[check] Python packages and versions"
  run_in_environment python - <<'PY'
import importlib

modules = [
    "numpy",
    "scipy",
    "ase",
    "gpaw",
    "mpi4py",
    "yaml",
    "matplotlib",
    "pandas",
    "nbconvert",
    "nbclient",
    "ipykernel",
]
for name in modules:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "available")
    print(f"  {name}: {version}")
PY

  echo "[check] GPAW compiled features"
  run_in_environment gpaw info

  echo "[check] PAW datasets"
  run_in_environment python - <<'PY'
from gpaw.setup_data import search_for_file

for element in ["H", "C", "O", "Zn", "Ti", "Ce"]:
    path, _ = search_for_file(f"{element}.PBE")
    print(f"  {element}: {path}")
PY

  echo "[check] One-process GPAW dry run"
  run_in_environment gpaw python --dry-run=1 -c \
    'from ase import Atoms; from gpaw import GPAW; atoms = Atoms("H", cell=[6, 6, 6], pbc=False); atoms.center(); atoms.calc = GPAW(mode="fd", xc="PBE", txt=None); atoms.get_potential_energy()'

  if [[ "${CHECK_MPI}" == "1" ]]; then
    echo "[check] Two-process GPAW MPI dry run"
    OMP_NUM_THREADS=1 run_in_environment mpiexec -n 2 gpaw python --dry-run=2 -c \
      'from ase import Atoms; from gpaw import GPAW; atoms = Atoms("H", cell=[6, 6, 6], pbc=False); atoms.center(); atoms.calc = GPAW(mode="fd", xc="PBE", txt=None); atoms.get_potential_energy()'
  fi
else
  echo "[check] Python packages and versions"
  run_in_environment python - <<'PY'
import importlib

modules = [
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "matplotlib",
    "yaml",
    "qiskit",
    "qiskit_aer",
    "qiskit_machine_learning",
    "qiskit_ibm_runtime",
    "nbconvert",
    "nbclient",
    "ipykernel",
]
for name in modules:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "available")
    print(f"  {name}: {version}")
PY
fi

[[ -d "${PROJECT_ROOT}/results" ]] || fail "Missing results directory: ${PROJECT_ROOT}/results"
[[ -w "${PROJECT_ROOT}/results" ]] || fail "Results directory is not writable: ${PROJECT_ROOT}/results"

echo "[ok] Environment '${ENVIRONMENT_NAME}' passed the ${PROFILE} checks."
