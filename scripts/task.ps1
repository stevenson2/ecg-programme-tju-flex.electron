# task.ps1 - thin task wrappers for the two-repo ECG project (R17 P5).
#
# Usage (from the electron repo root, PowerShell):
#   .\scripts\task.ps1 build              # ESP-IDF firmware build (short path)
#   .\scripts\task.ps1 flash              # flash via COM port
#   .\scripts\task.ps1 serial             # serial monitor
#   .\scripts\task.ps1 train "<cmd>"      # run a WSL training command
#   .\scripts\task.ps1 check              # hardcoded-path audit (both repos)
#
# Paths resolve through scripts\project_paths.json; nothing task-specific
# is hardcoded here except the ESP-IDF env vars documented in AGENTS.md §2.

param(
  [Parameter(Position = 0, Mandatory = $true)]
  [ValidateSet('build', 'flash', 'serial', 'train', 'check')]
  [string]$Target,

  [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

. (Join-Path $PSScriptRoot 'project_paths.ps1')

# IDF env vars (IDF_TOOLS_PATH / IDF_PYTHON_ENV_PATH / ESP_IDF) come from
# project_paths.ps1; IDF_PATH mirrors ESP_IDF for the idf.py toolchain.
$env:IDF_PATH = $env:ESP_IDF
$env:ESP_IDF_VERSION = '6.0'

switch ($Target) {
  'build' {
    & (Join-Path $env:IDF_PATH 'export.bat') | Out-Null
    Set-Location $env:ESP_SHORT
    idf.py -B build_n16r8 -D "SDKCONFIG=C:/esp/esp_idf_ecg_migration/sdkconfig.n16r8" build
  }
  'flash' {
    & (Join-Path $env:IDF_PATH 'export.bat') | Out-Null
    Set-Location $env:ESP_SHORT
    idf.py -B build_n16r8 -D "SDKCONFIG=C:/esp/esp_idf_ecg_migration/sdkconfig.n16r8" -p $env:COM_PORT flash
  }
  'serial' {
    & (Join-Path $env:IDF_PATH 'export.bat') | Out-Null
    Set-Location $env:ESP_SHORT
    idf.py -B build_n16r8 -D "SDKCONFIG=C:/esp/esp_idf_ecg_migration/sdkconfig.n16r8" -p $env:COM_PORT monitor
  }
  'train' {
    if (-not $Args -or -not $Args[0]) { throw 'usage: task.ps1 train "<command>"' }
    python (Join-Path $PSScriptRoot 'cross_shell.py') wsl --cmd "$($Args -join ' ')"
  }
  'check' {
    python (Join-Path $PSScriptRoot 'check_paths.py') @Args
  }
}
