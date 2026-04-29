@echo off
title ESP32-ECG 绘图仪
chcp 65001 >nul

echo ========================================
echo   ESP32-ECG 三通道心电绘图仪
echo   自动检测 COM 口，双击即用
echo ========================================
echo.

cd /d "%~dp0"

python pc_tools\ecg_plotter.py --port COM4

pause
