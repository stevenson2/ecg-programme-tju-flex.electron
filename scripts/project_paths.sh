#!/usr/bin/env bash
# Source this file to export project paths in WSL bash:
#   source scripts/project_paths.sh
# Variables: MEETI_ROOT ELECTRON_ROOT ECG_DATA AFDB LTAFDB CINC2017 AFPDB AFTDB
#            PTBXL_FOLD10 PTBXL_COPY WIN_TEMP PUBLIC_CLONE ESP_SHORT ESP_IDF
#            IDF_PYTHON DEVICE_052 DEVICE_REC_LATEST COM_PORT PIO
set -a
eval "$(python3 - <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path('scripts/project_paths.json').read_text(encoding='utf-8'))
w = d['wsl']; win = d['windows']
m = {
 'MEETI_ROOT': w['meeti_root'], 'ELECTRON_ROOT': w['electron_root'],
 'ECG_DATA': w['ecg_data'], 'AFDB': w['afdb'], 'LTAFDB': w['ltafdb'],
 'CINC2017': w['cinc2017'], 'AFPDB': w['afpdb'], 'AFTDB': w['aftdb'],
 'PTBXL_FOLD10': w['ptbxl_fold10'], 'PTBXL_COPY': w['ptbxl_copy'],
 'WIN_TEMP': w['windows_temp'], 'PUBLIC_CLONE': w['public_clone'],
 'ESP_SHORT': w['esp_short'], 'ESP_IDF': w['esp_idf'],
 'IDF_PYTHON': w['idf_python'], 'DEVICE_052': w['device_052'],
 'DEVICE_REC_LATEST': w['device_rec_latest'],
 'COM_PORT': win['com_port'], 'PIO': win['pio'],
}
for k, v in m.items():
    print(f"export {k}='{v}'")
PY
)"
set +a
