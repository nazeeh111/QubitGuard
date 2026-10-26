#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m qubitguard.cli serve --root "$PWD" --port "${QUBITGUARD_PORT:-8917}"
fi
exec python3 -m qubitguard.cli serve --root "$PWD" --port "${QUBITGUARD_PORT:-8917}"
