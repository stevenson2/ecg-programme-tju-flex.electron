# ESP32-ECG 工具箱 - PowerShell 脚本
# 右键 → 使用 PowerShell 运行，或直接在终端执行

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PioExe = "C:\Users\cai\.platformio\penv\Scripts\pio.exe"
$Python = "python"

function Show-Menu {
    Clear-Host
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "      ESP32-ECG  Toolbox" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Project: $ProjectRoot"
    Write-Host ""
    Write-Host "  [1] PC Plotter (3-channel waveform)"
    Write-Host "  [2] Serial Monitor (raw data)"
    Write-Host "  [3] Compile & Upload firmware"
    Write-Host "  [4] Open project folder"
    Write-Host "  [5] Exit"
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
}

function Start-Plotter {
    Clear-Host
    Write-Host "[Plotter] Launching 3-channel waveform viewer..."
    Write-Host ""
    Start-Process "python" -ArgumentList "pc_tools\ecg_plotter.py --port COM4" -WindowStyle Normal -WorkingDirectory $ProjectRoot
    Start-Sleep 1
}

function Start-Monitor {
    Clear-Host
    Write-Host "[Monitor] Opening serial monitor on COM4 @ 115200..."
    Write-Host ""
    Start-Process -FilePath "$PioExe" -ArgumentList "device monitor -p COM4 -b 115200" -WindowStyle Normal -WorkingDirectory $ProjectRoot
    Start-Sleep 1
}

function Start-Upload {
    Clear-Host
    Write-Host "[Build] Compiling and uploading..."
    Write-Host ""
    Set-Location $ProjectRoot
    & $PioExe run --target upload
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "SUCCESS: Upload complete!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "FAILED: Check USB connection." -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Press any key to continue..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

function Open-Explorer {
    Start-Process "explorer" -ArgumentList $ProjectRoot
}

# Main loop
do {
    Show-Menu
    $choice = Read-Host "`nSelect [1-5]"
    
    switch ($choice) {
        "1" { Start-Plotter }
        "2" { Start-Monitor }
        "3" { Start-Upload }
        "4" { Open-Explorer }
        "5" { 
            Write-Host "Goodbye!" -ForegroundColor Green
            break 
        }
        default {
            Write-Host "Invalid option, try again." -ForegroundColor Yellow
            Start-Sleep 1
        }
    }
} while ($choice -ne "5")
