#!/usr/bin/env bash
# One-command rerun of the AF RR-source mechanism experiment.
# Usage:
#   ./run.sh                 # full PhysioNet download + eval
#   ./run.sh --smoke         # small subset
#   ./run.sh --skip-download # reuse data/
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

python3 -m pip install --user -q -r requirements.txt
bash host/build.sh
python3 -m pytest -q tests
python3 python/run_experiment.py --root "$ROOT" "$@"
