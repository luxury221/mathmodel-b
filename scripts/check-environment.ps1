. (Join-Path $PSScriptRoot 'env.ps1')
if (-not (Test-Path -LiteralPath $PythonExe)) { throw 'Run scripts\setup.ps1 first.' }
& $UvExe pip check --python $PythonExe
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
& $PythonExe (Join-Path $PSScriptRoot 'check_environment.py')
exit $LASTEXITCODE
