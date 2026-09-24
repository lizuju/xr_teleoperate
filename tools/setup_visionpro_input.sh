#!/usr/bin/env bash
set -euo pipefail
tools_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dev_root="$(cd -- "${tools_dir}/../.." && pwd)"
/usr/bin/python3 -m venv "${dev_root}/.venv-visionpro"
"${dev_root}/.venv-visionpro/bin/python" -m pip install -r "${tools_dir}/requirements-visionpro.txt"
echo "Vision Pro receiver ready: ${dev_root}/.venv-visionpro/bin/python"
