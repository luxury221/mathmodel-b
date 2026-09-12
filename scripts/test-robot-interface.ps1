param([switch]$UnitOnly)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
$RunName = 'run_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$OutputDirectory = Join-Path $ProjectRoot ('reports\interface_validation_v1\' + $RunName)
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$PreviousRunRoot = $env:B2026_INTERFACE_RUN_ROOT
$env:B2026_INTERFACE_RUN_ROOT = $OutputDirectory
Push-Location -LiteralPath $ProjectRoot
try {
    & $PythonExe -m unittest discover -s research/interface_validation -p test_interface.py -v 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputDirectory 'unit_tests.log')
    if ($LASTEXITCODE -ne 0) { throw 'Interface unit tests failed; evidence preserved.' }
    & $PythonExe -m unittest discover -s research/offline_validation -p test_plan_components.py -v 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputDirectory 'frozen_component_tests.log')
    if ($LASTEXITCODE -ne 0) { throw 'Frozen geometry component tests failed.' }
    & $PythonExe -m ruff check src/b2026_robot src/run_robot.py research/interface_validation 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputDirectory 'ruff.log')
    if ($LASTEXITCODE -ne 0) { throw 'Interface static checks failed.' }
    & $PythonExe -m compileall -q src/b2026_robot src/run_robot.py research/interface_validation
    if ($LASTEXITCODE -ne 0) { throw 'Compilation check failed.' }
    if (-not $UnitOnly) {
        & $PythonExe -u research/interface_validation/run_trials.py --output $OutputDirectory 2>&1 |
            Tee-Object -FilePath (Join-Path $OutputDirectory 'integration.log')
        if ($LASTEXITCODE -ne 0) { throw 'Local HTTP integration failed; evidence preserved.' }
    }
    [ordered]@{
        unit_tests = 'passed'
        frozen_geometry_tests = 'passed'
        ruff = 'passed'
        compileall = 'passed'
        integration = $(if ($UnitOnly) { 'not_run' } else { 'passed' })
        official_calls = 0
        output_directory = $OutputDirectory
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'verification.json') -Encoding utf8
    if (-not $UnitOnly) {
        Set-Content -LiteralPath (Join-Path $ProjectRoot 'reports\interface_validation_v1\LATEST_SUCCESS.txt') `
            -Value $OutputDirectory -Encoding utf8
    }
    Write-Output "Local-only interface validation: $OutputDirectory"
} finally {
    Pop-Location
    $env:B2026_INTERFACE_RUN_ROOT = $PreviousRunRoot
}
