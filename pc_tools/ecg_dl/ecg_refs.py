#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference-ECG-database locations for the pc_tools pipeline (R17 P4).

The big reference databases no longer live inside the repo: they moved to the
unified library root configured in scripts/project_paths.json (`ecg_refs_*`
keys; Windows `C:\\ecg_data`, WSL via the WSL-flavor keys of the same JSON).
Import this module instead of hardcoding those paths:

    from ecg_refs import PTBXL_DIR, ECG_DATABASE_DIR, INCART_DIR

Works from either Windows or WSL because project_paths resolves the flavor
from the running interpreter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from project_paths import paths_for  # noqa: E402

_P = paths_for()

PTBXL_DIR = Path(_P['ecg_refs_ptbxl'])
ECG_DATABASE_DIR = Path(_P['ecg_refs_ecg_database'])
INCART_DIR = Path(_P['ecg_refs_incart']) / 'files'
FIGS_DIR = Path(_P['ecg_refs_figs'])
RECORDS_FILE = ECG_DATABASE_DIR / 'RECORDS'
LUDB_DIR = (ECG_DATABASE_DIR
            / 'lobachevsky-university-electrocardiography-database-1.0.1'
            / 'lobachevsky-university-electrocardiography-database-1.0.1'
            / 'data')
