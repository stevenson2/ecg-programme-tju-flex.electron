@echo off
title ESP32-ECG 串口监视器
chcp 65001 >nul

echo ========================================
echo   ESP32-ECG 串口监视器
echo   自动连接 COM4 @ 115200
echo ========================================
echo.
echo 提示：串口数据格式为 clean,noisy,filtered
echo 按 Ctrl+C 退出
echo.

cd /d "%~dp0"

"C:\Users\cai\.platformio\penv\Scripts\pio.exe" device monitor -p COM4 -b 115200

pause
