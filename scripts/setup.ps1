param([switch]$RefreshLock)
. (Join-Path $PSScriptRoot 'env.ps1')
if (-not (Test-Path -LiteralPath $UvExe)) {
    throw 'The verified uv executable is missing from .runtime\uv.'
}
$PythonVersion = (Get-Content -LiteralPath (Join-Path $ProjectRoot '.python-version') -Raw).Trim()
& $UvExe python install $PythonVersion --no-bin
if ($LASTEXITCODE -ne 0) { throw 'Managed Python installation failed.' }
if (-not (Test-Path -LiteralPath $PythonExe)) {
    & $UvExe venv --python $PythonVersion --managed-python (Join-Path $ProjectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$LockPath = Join-Path $ProjectRoot 'requirements.lock.txt'
if ($RefreshLock -or -not (Test-Path -LiteralPath $LockPath)) {
    & $UvExe pip compile (Join-Path $ProjectRoot 'requirements.in') --python $PythonExe --generate-hashes --output-file $LockPath | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Dependency resolution failed.' }
}
& $UvExe pip sync $LockPath --python $PythonExe --require-hashes
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
$KernelArguments = @('-m', 'ipykernel', 'install', '--prefix', $env:VIRTUAL_ENV, '--name', 'b2026', '--display-name', 'Python (B2026, D drive)')
foreach ($EnvironmentName in @($EnvironmentDirectories.Keys) + @('PYTHONUTF8', 'PYTHONIOENCODING', 'PYTHONNOUSERSITE', 'B2026_ROOT')) {
    $KernelArguments += @('--env', $EnvironmentName, [Environment]::GetEnvironmentVariable($EnvironmentName, 'Process'))
}
& $PythonExe @KernelArguments
if ($LASTEXITCODE -ne 0) { throw 'Jupyter kernel registration failed.' }
& $UvExe pip check --python $PythonExe
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
Write-Host 'Environment installed. Run scripts\check-environment.ps1 to validate it.'
