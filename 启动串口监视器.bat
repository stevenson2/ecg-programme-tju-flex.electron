@echo off
title ESP32-ECG Monitor
chcp 936 >nul

echo ========================================
echo   ESP32-ECG Serial Monitor
echo   COM4 @ 115200
echo ========================================
echo.
echo Data format: clean,noisy,filtered
echo Press Ctrl+C to exit
echo.

cd /d "%~dp0"

"C:\Users\cai\.platformio\penv\Scripts\pio.exe" device monitor -p COM4 -b 115200

pause
