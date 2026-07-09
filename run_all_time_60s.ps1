param(
    [string]$Python = "C:/Users/bench/AppData/Local/Python/pythoncore-3.14-64/python.exe"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Runs = @(
    @{ Name = "Easy - empty-8-8"; Manifest = "experiment_manifest_easy_time_60s.json" },
    @{ Name = "Medium - maze-32-32-2"; Manifest = "experiment_manifest_medium_time_60s.json" },
    @{ Name = "Hard - warehouse-10-20-10-2-1"; Manifest = "experiment_manifest_hard_time_60s.json" }
)

New-Item -ItemType Directory -Force "results" | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path "results" "all_time_60s_$Timestamp.log"

Start-Transcript -Path $LogFile
try {
    $OverallStart = Get-Date

    foreach ($Run in $Runs) {
        Write-Host ""
        Write-Host "============================================================"
        Write-Host "Starting: $($Run.Name)"
        Write-Host "Manifest: $($Run.Manifest)"
        Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host "============================================================"

        if (-not (Test-Path $Run.Manifest)) {
            throw "Manifest not found: $($Run.Manifest)"
        }

        $RunStart = Get-Date

        & $Python "src/run_isolated_matrix.py" $Run.Manifest "--overwrite" "--fail-fast"

        if ($LASTEXITCODE -ne 0) {
            throw "Run failed: $($Run.Name), exit code: $LASTEXITCODE"
        }

        $RunDuration = (Get-Date) - $RunStart
        Write-Host "Finished: $($Run.Name)"
        Write-Host "Duration: $($RunDuration.ToString())"
    }

    $OverallDuration = (Get-Date) - $OverallStart
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "ALL 60-SECOND EXPERIMENTS FINISHED"
    Write-Host "Total duration: $($OverallDuration.ToString())"
    Write-Host "Log file: $LogFile"
    Write-Host "============================================================"
}
finally {
    Stop-Transcript | Out-Null
}
