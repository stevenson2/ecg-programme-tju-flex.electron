# Dot-source this file in PowerShell to set project paths:
#   . .\scripts\project_paths.ps1
# Sets $env:MEETI_ROOT, $env:ELECTRON_ROOT, $env:ESP_SHORT, $env:COM_PORT, ...
$__here = $PSScriptRoot
if (-not $__here) { $__here = Split-Path -Parent $MyInvocation.MyCommand.Path }
$__pathsJson = Join-Path $__here 'project_paths.json'
$__d = Get-Content $__pathsJson -Raw -Encoding UTF8 | ConvertFrom-Json
$__w = $__d.windows
$env:MEETI_ROOT        = $__w.meeti_root
$env:ELECTRON_ROOT     = $__w.electron_root
$env:ECG_DATA          = $__w.ecg_data
$env:WIN_TEMP          = $__w.windows_temp
$env:PUBLIC_CLONE      = $__w.public_clone
$env:ESP_SHORT         = $__w.esp_short
$env:ESP_IDF           = $__w.esp_idf
$env:IDF_PYTHON        = $__w.idf_python
$env:PIO               = $__w.pio
$env:COM_PORT          = $__w.com_port
$env:DEVICE_052        = $__w.device_052
$env:DEVICE_REC_LATEST = $__w.device_rec_latest
Write-Host "[project_paths] COM_PORT=$env:COM_PORT ESP_SHORT=$env:ESP_SHORT"
