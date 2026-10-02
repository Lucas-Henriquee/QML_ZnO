#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECK_SCRIPT="${PROJECT_DIR}/scripts/environment/check_environment.sh"
ENVIRONMENT_NAME="${QML_ZNO_SIM_ENV:-qml-zno-simulation}"

INPUT_NOTEBOOK="${1:-zno-co2-lucas.ipynb}"
OUTPUT_NOTEBOOK="${2:-zno-co2-lucas.executed.ipynb}"
NB_TIMEOUT="${NB_TIMEOUT:-86400}"

if [[ ! -f "${SCRIPT_DIR}/${INPUT_NOTEBOOK}" ]]; then
  echo "[error] Input notebook does not exist: ${SCRIPT_DIR}/${INPUT_NOTEBOOK}" >&2
  exit 1
fi

if [[ -e "${SCRIPT_DIR}/${OUTPUT_NOTEBOOK}" ]]; then
  echo "[error] Refusing to overwrite the existing notebook: ${SCRIPT_DIR}/${OUTPUT_NOTEBOOK}" >&2
  echo "Choose a different output filename as the second argument." >&2
  exit 1
fi

command -v conda >/dev/null 2>&1 || {
  echo "[error] Conda was not found in PATH." >&2
  exit 1
}

if ! conda run -n "${ENVIRONMENT_NAME}" python --version >/dev/null 2>&1; then
  echo "[error] Conda environment '${ENVIRONMENT_NAME}' is not available." >&2
  echo "Run ./setup_lucas_env.sh before executing the notebook." >&2
  exit 1
fi

if [[ ! -x "${CHECK_SCRIPT}" ]]; then
  echo "[error] Missing environment check script: ${CHECK_SCRIPT}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"

"${CHECK_SCRIPT}" simulation --env "${ENVIRONMENT_NAME}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec conda run --no-capture-output -n "${ENVIRONMENT_NAME}" python -m nbconvert \
  --to notebook \
  --execute "${INPUT_NOTEBOOK}" \
  --output "${OUTPUT_NOTEBOOK}" \
  --ExecutePreprocessor.timeout="${NB_TIMEOUT}"
