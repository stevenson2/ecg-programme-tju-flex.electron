@echo off
title ESP32-ECG Monitor
chcp 936 >nul

cd /d "%~dp0"

REM Auto-detect COM port
for /f "usebackq" %%p in (`python pc_tools\find_port.py`) do set COMPORT=%%p
if "%COMPORT%"=="" set COMPORT=COM4

echo ========================================
echo   ESP32-ECG Serial Monitor
echo   Port: %COMPORT% @ 115200
echo ========================================
echo.
echo Data format: clean,noisy,filtered
echo Press Ctrl+C to exit
echo.

"C:\Users\cai\.platformio\penv\Scripts\pio.exe" device monitor -p %COMPORT% -b 115200

pause
