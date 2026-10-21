#!/bin/bash
# One-shot VF-rule evaluation.
#   ECG_DATA_ROOT=/path/to/cache ./run_eval.sh
#   ./run_eval.sh --data-root /path/to/cache
# Databases (vfdb, cudb, mitdb, and nstdb when present) are read from that
# root, or from ./data, and downloaded from PhysioNet when a required set
# is missing. Waveforms stay in the data directory and are gitignored.
set -euo pipefail
cd "$(dirname "$0")"
python3 -c "import numpy, scipy, matplotlib, wfdb"
./host/build.sh
python3 test_consistency.py
python3 eval_vf_rules.py "$@"
