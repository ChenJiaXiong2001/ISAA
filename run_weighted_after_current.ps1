param(
    [int]$ProcessId = 0,
    [string]$Match = "train_best_no_pool.py",
    [string[]]$TrainingArguments = @()
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-ProcessAlive([int]$Id) {
    return $null -ne (Get-Process -Id $Id -ErrorAction SilentlyContinue)
}

if ($ProcessId -gt 0) {
    Write-Host "等待训练进程 PID=$ProcessId 完成..."
    while (Test-ProcessAlive $ProcessId) {
        Start-Sleep -Seconds 30
    }
} else {
    Write-Host "查找当前训练进程: $Match"
    while ($true) {
        $running = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -match "^(python|python3|torchrun)(\.exe)?$" -and
            $_.CommandLine -like "*$Match*" -and
            $_.CommandLine -notlike "*run_weighted_after_current.ps1*" -and
            $_.CommandLine -notlike "*train_best_no_pool_weighted_100.py*"
        })
        if ($running.Count -eq 0) {
            break
        }
        $pids = ($running | ForEach-Object { $_.ProcessId }) -join ", "
        Write-Host "仍在训练，PID=$pids；30秒后重试..."
        Start-Sleep -Seconds 30
    }
}

Write-Host "当前训练已结束，启动100帧置信度加权压缩训练。"
Push-Location $projectRoot
try {
    & python train_best_no_pool_weighted_100.py @TrainingArguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
