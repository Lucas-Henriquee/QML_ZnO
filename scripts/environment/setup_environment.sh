#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROFILE=""
UPDATE=0
DRY_RUN=0
ENV_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage: setup_environment.sh PROFILE [OPTIONS]

Create or update a project Conda environment.

Profiles:
  simulation    GPAW, ASE, MPI, and notebook tools
  modeling      Dataset, analysis, ML, QML, and IBM Runtime tools
  all           Install both profiles (cannot be combined with --name)

Options:
  --update      Update an existing environment without pruning packages
  --dry-run     Show the creation plan without installing packages
  --name NAME   Override the environment name for one profile
  -h, --help    Show this help message
EOF
}

fail() {
  echo "[error] $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    simulation|modeling|all)
      [[ -z "${PROFILE}" ]] || fail "Only one profile may be selected."
      PROFILE="$1"
      shift
      ;;
    --update)
      UPDATE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --name)
      [[ $# -ge 2 ]] || fail "--name requires a value."
      ENV_OVERRIDE="$2"
      shift 2
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

[[ -n "${PROFILE}" ]] || fail "Select simulation, modeling, or all."
command -v conda >/dev/null 2>&1 || fail "Conda was not found in PATH."

if [[ "${PROFILE}" == "all" && -n "${ENV_OVERRIDE}" ]]; then
  fail "--name can only be used with one profile."
fi

if [[ "${UPDATE}" == "1" && "${DRY_RUN}" == "1" ]]; then
  fail "Conda does not support a dry run for 'conda env update'. Choose one option."
fi

default_name() {
  case "$1" in
    simulation) echo "qml-zno-simulation" ;;
    modeling) echo "qml-zno-modeling" ;;
    *) fail "Unsupported profile: $1" ;;
  esac
}

environment_exists() {
  local environment_name="$1"
  conda env list | awk 'NF > 0 && $1 !~ /^#/ {print $1}' | grep -Fxq "${environment_name}"
}

install_profile() {
  local profile="$1"
  local environment_file="${PROJECT_ROOT}/environments/${profile}.yaml"
  local environment_name
  environment_name="${ENV_OVERRIDE:-$(default_name "${profile}")}"

  [[ -f "${environment_file}" ]] || fail "Missing environment file: ${environment_file}"

  if environment_exists "${environment_name}"; then
    if [[ "${UPDATE}" == "1" ]]; then
      echo "[update] ${environment_name} from ${environment_file}"
      conda env update --name "${environment_name}" --file "${environment_file}" --solver libmamba
    else
      echo "[skip] Environment '${environment_name}' already exists."
      echo "       Use --update to apply the current environment file."
    fi
  else
    echo "[create] ${environment_name} from ${environment_file}"
    local create_command=(conda env create --name "${environment_name}" --file "${environment_file}" --solver libmamba --yes)
    if [[ "${DRY_RUN}" == "1" ]]; then
      create_command+=(--dry-run)
    fi
    "${create_command[@]}"
  fi

  if [[ "${DRY_RUN}" == "0" ]]; then
    "${SCRIPT_DIR}/check_environment.sh" "${profile}" --env "${environment_name}"
  fi
}

if [[ "${PROFILE}" == "all" ]]; then
  install_profile simulation
  install_profile modeling
else
  install_profile "${PROFILE}"
fi

