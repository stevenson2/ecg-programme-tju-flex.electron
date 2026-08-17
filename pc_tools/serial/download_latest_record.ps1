# download_latest_record.ps1 — 从 ESP32-ECG AP 下载录制记录 (meta+data)
# 用法: 先让电脑连上 AP "ESP32-ECG-XXXX" (密码 12345678), 再运行:
#   .\download_latest_record.ps1            # 自动选 id 最大的记录
#   .\download_latest_record.ps1 -Id 123    # 下载指定 id (rec_collect 的 [NEWREC] id)
param([int]$Id = 0)

$ErrorActionPreference = 'Stop'
$base = 'http://192.168.4.1'

$list = Invoke-RestMethod "$base/api/records" -TimeoutSec 10
$list | ConvertTo-Json -Depth 5 | Out-File -FilePath 'api_records.json' -Encoding utf8

$ids = @($list.records | ForEach-Object { [int]$_.id })
if ($ids.Count -eq 0) { Write-Output 'no records'; exit 1 }
if ($Id -eq 0) {
    $Id = ($ids | Measure-Object -Maximum).Maximum
    Write-Output "using max id=$Id (list count=$($ids.Count))"
} else {
    if ($ids -notcontains $Id) { Write-Warning "id $Id not in list; ids: $($ids -join ',')"; exit 2 }
}

$meta = Invoke-RestMethod "$base/api/records/$Id/meta" -TimeoutSec 10
$meta | ConvertTo-Json -Depth 5 | Out-File -FilePath 'api_record_meta.json' -Encoding utf8
Invoke-WebRequest "$base/api/records/$Id/data" -OutFile 'rec_latest.ecgr' -TimeoutSec 120

Write-Output "meta: $($meta | ConvertTo-Json -Compress)"
Write-Output "saved: api_records.json, api_record_meta.json, rec_latest.ecgr"
