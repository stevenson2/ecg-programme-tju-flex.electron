#!/usr/bin/env bash
# task.sh — thin task wrappers for the two-repo ECG project (R17 P5).
#
# Usage:
#   scripts/task.sh train    "<command...>"   # run in WSL (GPU env preloaded)
#   scripts/task.sh build                     # ESP-IDF firmware build (Windows)
#   scripts/task.sh flash                     # flash via COM port (Windows)
#   scripts/task.sh serial                    # serial monitor (Windows)
#   scripts/task.sh check                     # hardcoded-path audit (both repos)
#   scripts/task.sh push-research             # meeti snapshot -> push research tree
#
# Shells are resolved through scripts/project_paths.json via cross_shell.py;
# no task-specific path is hardcoded here. Long jobs: start and poll in the
# same shell (AGENTS.md §1).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

PYTHON=python3
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

target="${1:-}"
shift || true

case "$target" in
  train)
    cmd="$*"
    [ -n "$cmd" ] || { echo "usage: task.sh train \"<command...>\"" >&2; exit 2; }
    # GPU preamble (AGENTS.md §6); CUDA_VISIBLE_DEVICES= for CPU-only runs.
    gpu_ld="$(find /usr/local/lib/python3.12/dist-packages/nvidia -maxdepth 2 -type d -name lib -printf '%p:' 2>/dev/null || true)"
    exec bash -lc "export LD_LIBRARY_PATH=${gpu_ld%%:}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; source scripts/project_paths.sh; $cmd"
    ;;
  build|flash|serial)
    case "$target" in
      build)  cmd='idf.py -B build_n16r8 -D SDKCONFIG=C:/esp/esp_idf_ecg_migration/sdkconfig.n16r8 build' ;;
      flash)  cmd='idf.py -B build_n16r8 -D SDKCONFIG=C:/esp/esp_idf_ecg_migration/sdkconfig.n16r8 -p COM3 flash' ;;
      serial) cmd='idf.py -B build_n16r8 -D SDKCONFIG=C:/esp/esp_idf_ecg_migration/sdkconfig.n16r8 -p COM3 monitor' ;;
    esac
    exec "$PYTHON" scripts/cross_shell.py win --cmd "$cmd"
    ;;
  check)
    exec "$PYTHON" scripts/check_paths.py "$@"
    ;;
  push-research)
    echo "task.sh: push-research needs the research-push flow; see README '公开与私密边界' and AGENTS.md §4 (tar snapshot -> %TEMP%\\meeti_push -> push research branch from Windows)."
    exit 3
    ;;
  *)
    echo "usage: task.sh {train|build|flash|serial|check|push-research} [args]" >&2
    exit 2
    ;;
esac
