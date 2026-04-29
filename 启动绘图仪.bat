@echo off
title ESP32-ECG Plotter
chcp 936 >nul

echo ========================================
echo   ESP32-ECG 3-Channel Plotter
echo   Auto-detecting COM4...
echo ========================================
echo.

cd /d "%~dp0"

python pc_tools\ecg_plotter.py --port COM4

pause
