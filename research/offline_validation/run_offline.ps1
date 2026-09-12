$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
. (Join-Path $ProjectRoot 'scripts\env.ps1')
Push-Location -LiteralPath $ProjectRoot
try {
    $OutputDirectory = Join-Path $ProjectRoot 'reports\offline_review'
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    foreach ($ScriptName in @('audit_geometry', 'offline_benchmark', 'replay_verify', 'make_figures')) {
        $ScriptPath = Join-Path $PSScriptRoot "$ScriptName.py"
        $LogPath = Join-Path $OutputDirectory "$ScriptName`_run.txt"
        & $PythonExe -u $ScriptPath 2>&1 | Tee-Object -FilePath $LogPath
        if ($LASTEXITCODE -ne 0) {
            throw "Offline validation failed: $ScriptName"
        }
    }
    & $PythonExe -m ruff check $PSScriptRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Offline validation lint failed.'
    }
    & $PythonExe -m compileall -q $PSScriptRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Offline validation compilation failed.'
    }
    Write-Output "Offline-only validation complete. Results: $OutputDirectory"
} finally {
    Pop-Location
}
