#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
FW="$REPO/experiments/esp_idf_ecg_migration/components/ecg_core"
cd "$HERE"

gcc -O2 -I"$FW/include" -c "$FW/src/ai_qrs.c" -o ai_qrs.o

g++ -O2 -std=c++17 -I. -I"$FW/include" \
    hr_host.cpp \
    "$FW/src/heartrate.cpp" \
    "$FW/src/filter.cpp" \
    ai_qrs.o \
    -o hr_host

g++ -O2 -std=c++17 -I. -I"$FW/include" \
    af_host.cpp \
    "$FW/src/af_detect.cpp" \
    -o af_host

ls -la hr_host af_host
