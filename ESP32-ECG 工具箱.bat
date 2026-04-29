@echo off
title ESP32-ECG 工具箱
chcp 65001 >nul

:menu
cls
echo ========================================
echo      ESP32-ECG 心电采集系统 - 工具箱
echo ========================================
echo.
echo  当前项目: %~dp0
echo.
echo  [1] 启动 PC 绘图仪 (三通道波形)
echo  [2] 启动串口监视器 (原始数据)
echo  [3] 编译并烧录固件
echo  [4] 打开项目文件夹
echo  [5] 退出
echo.
echo ========================================
set /p choice="请选择 [1-5]: "

if "%choice%"=="1" goto plotter
if "%choice%"=="2" goto monitor
if "%choice%"=="3" goto upload
if "%choice%"=="4" goto explorer
if "%choice%"=="5" goto exit
goto menu

:plotter
cls
echo.
echo [启动绘图仪] 自动检测 COM 口...
echo.
cd /d "%~dp0"
start "绘图仪" cmd /c "python pc_tools\ecg_plotter.py & pause"
goto menu

:monitor
cls
echo.
echo [启动串口监视器] COM4 @ 115200...
echo.
cd /d "%~dp0"
start "串口" cmd /c ""C:\Users\cai\.platformio\penv\Scripts\pio.exe" device monitor -p COM4 -b 115200"
goto menu

:upload
cls
echo.
echo [编译烧录] 开始编译...
echo.
cd /d "%~dp0"
"C:\Users\cai\.platformio\penv\Scripts\pio.exe" run --target upload
if %errorlevel% equ 0 (
    echo.
    echo 烧录成功！
) else (
    echo.
    echo 烧录失败，请检查连接。
)
pause
goto menu

:explorer
start "" "%~dp0"
goto menu

:exit
exit /b
