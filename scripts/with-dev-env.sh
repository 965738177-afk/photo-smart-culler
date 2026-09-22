#!/usr/bin/env bash
# Run PhotoSort commands with the workspace-local native EXIF libraries.
#
# Usage:
#   scripts/with-dev-env.sh                         # launch PhotoSort
#   scripts/with-dev-env.sh .venv/bin/python -m pytest -q
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(cd "${project_dir}/.." && pwd)"
library_root="${workspace_dir}/work/homebrew-libs"

required_dirs=(
  "${library_root}/brotli/1.2.0/lib"
  "${library_root}/gettext/1.0/lib"
  "${library_root}/inih/62/lib"
)

native_library_path=""
for required_dir in "${required_dirs[@]}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Missing native development library: ${required_dir}" >&2
    exit 1
  fi
  native_library_path="${native_library_path:+${native_library_path}:}${required_dir}"
done

export DYLD_LIBRARY_PATH="${native_library_path}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

if [[ "$#" -eq 0 ]]; then
  exec "${project_dir}/.venv/bin/python" -m src.main
fi

exec "$@"
