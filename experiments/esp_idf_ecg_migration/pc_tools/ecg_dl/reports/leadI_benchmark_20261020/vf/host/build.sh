#!/bin/bash
# Build the host replay of vf_detect.cpp + heartrate.cpp + filter.cpp.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
d="$HERE"
FW=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
  d="$(dirname "$d")"
  if [[ -f "$d/experiments/esp_idf_ecg_migration/components/ecg_core/src/vf_detect.cpp" ]]; then
    FW="$d/experiments/esp_idf_ecg_migration/components/ecg_core"
    break
  fi
done
if [[ -z "$FW" ]]; then
  echo "ecg_core not found above $HERE" >&2
  exit 1
fi
BUILD="$HERE/build"
mkdir -p "$BUILD"
cat "$FW/src/vf_detect.cpp" "$HERE/vf_export.cpp" > "$BUILD/vf_detect_host.cpp"
CXX="${CXX:-g++}"
"$CXX" -O2 -std=c++17 -ffunction-sections -fdata-sections \
  -I"$HERE/stubs" -I"$FW/include" \
  "$BUILD/vf_detect_host.cpp" \
  "$FW/src/heartrate.cpp" \
  "$FW/src/filter.cpp" \
  "$HERE/fw_replay_host.cpp" \
  -lm -Wl,--gc-sections -o "$HERE/fw_replay_host"
echo "built $HERE/fw_replay_host"
