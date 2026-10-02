#!/usr/bin/env bash
set -euo pipefail
# Keep previously documented commands working.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../../run_analysis.sh" "$@"
