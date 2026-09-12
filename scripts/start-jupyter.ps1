param([ValidateRange(1024, 65535)][int]$Port = 8888)
. (Join-Path $PSScriptRoot 'env.ps1')
if (-not (Test-Path -LiteralPath $PythonExe)) { throw 'Run scripts\setup.ps1 first.' }
Set-Location -LiteralPath $ProjectRoot
& $PythonExe -m jupyterlab --no-browser --ServerApp.ip=127.0.0.1 --ServerApp.port=$Port --ServerApp.port_retries=0 --ServerApp.root_dir=$ProjectRoot
exit $LASTEXITCODE
