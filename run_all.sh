#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
exec conda run --no-capture-output -n "${QML_ZNO_MODEL_ENV:-qml-zno-modeling}" \
  python -m scripts.workflows.run_all "$@"
