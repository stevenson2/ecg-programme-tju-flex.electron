#!/bin/bash
cd '/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl' || exit 1
START=$(date +%s)
python3 retest_ai_rec_latest.py
RC=$?
END=$(date +%s)
echo "EXIT=$RC ELAPSED=$((END-START))s" > /dev/null
exit $RC
