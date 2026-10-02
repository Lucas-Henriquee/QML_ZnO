#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
ENVIRONMENT_NAME="${QML_ZNO_MODEL_ENV:-qml-zno-modeling}"

command -v conda >/dev/null 2>&1 || {
  echo "[error] Conda was not found in PATH." >&2
  exit 1
}

if ! conda run -n "${ENVIRONMENT_NAME}" python -c \
  'import matplotlib, numpy, pandas, yaml' >/dev/null 2>&1; then
  echo "[error] Conda environment '${ENVIRONMENT_NAME}' is not available." >&2
  echo "Run ./scripts/environment/setup_environment.sh modeling first," >&2
  echo "then ./scripts/environment/check_environment.sh modeling." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export QML_ZNO_ACTIVE_ENVIRONMENT="${ENVIRONMENT_NAME}"
exec conda run --no-capture-output -n "${ENVIRONMENT_NAME}" \
  python -m scripts.workflows.run_analysis "$@"
