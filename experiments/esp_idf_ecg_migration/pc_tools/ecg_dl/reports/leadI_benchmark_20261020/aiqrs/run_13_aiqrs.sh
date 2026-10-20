#!/bin/bash
cd "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/benchmark_leadI_20261019"
bash tools/hr_host/build.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2
export DET=aiqrs MODELS_SUBSET=student_hard,distill_v2,exp5_int8
nice -n 19 python3 -u scripts/13_event_sensitivity_fph.py > logs/13_event_sensitivity_fph_aiqrs.log 2>&1
echo EXIT=$? >> logs/13_event_sensitivity_fph_aiqrs.log