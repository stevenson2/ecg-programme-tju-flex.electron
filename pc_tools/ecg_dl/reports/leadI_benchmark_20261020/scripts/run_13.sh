#!/bin/bash
cd "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/benchmark_leadI_20261019"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2
nice -n 19 python3 -u scripts/13_event_sensitivity_fph.py > logs/13_event_sensitivity_fph.log 2>&1
echo EXIT=$? >> logs/13_event_sensitivity_fph.log