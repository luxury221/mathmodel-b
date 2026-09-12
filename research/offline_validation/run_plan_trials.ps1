param([switch]$SkipReplay)
$ErrorActionPreference = 'Stop'
$ValidationRoot = $PSScriptRoot
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $ValidationRoot '..\..'))
. (Join-Path $ProjectRoot 'scripts\env.ps1')
Push-Location -LiteralPath $ProjectRoot
try {
    $OutputDirectory = Join-Path $ProjectRoot 'reports\plan_trials_v2'
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    & $PythonExe -m unittest discover -s $ValidationRoot -p 'test_plan_components.py' -v 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputDirectory 'component_tests.log')
    if ($LASTEXITCODE -ne 0) { throw 'Plan component tests failed.' }
    & $PythonExe -m ruff check $ValidationRoot
    if ($LASTEXITCODE -ne 0) { throw 'Static checks failed.' }
    & $PythonExe -u (Join-Path $ValidationRoot 'plan_experiments.py') --phase all 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputDirectory 'all_phases.log')
    if ($LASTEXITCODE -ne 0) { throw 'Offline experiment failed.' }
    if (-not $SkipReplay) {
        & $PythonExe -u (Join-Path $ValidationRoot 'plan_experiments.py') --phase replay 2>&1 |
            Tee-Object -FilePath (Join-Path $OutputDirectory 'deterministic_replay.log')
        if ($LASTEXITCODE -ne 0) { throw 'Deterministic replay failed.' }
    }
    & $PythonExe (Join-Path $ValidationRoot 'plan_report.py')
    if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
    & $PythonExe -m compileall -q $ValidationRoot
    if ($LASTEXITCODE -ne 0) { throw 'Compilation check failed.' }
    Write-Output "Offline plan tests complete: $OutputDirectory"
} finally {
    Pop-Location
}
