set -e
cd "$(dirname "$0")"
FW="/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/experiments/esp_idf_ecg_migration/components/ecg_core"
nice -n 19 g++ -O2 -std=c++17 -I. -I"$FW/include" -include esp_timer.h hr_host.cpp "$FW/src/heartrate.cpp" "$FW/src/filter.cpp" -o hr_host 2>&1 | grep -v -i warning | head -30
ls -la hr_host
