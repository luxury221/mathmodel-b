param(
    [ValidateSet('q3', 'q4')][string]$Problem = 'q3',
    [string]$RobotId,
    [string]$BaseUrl = 'http://127.0.0.1:2026',
    [ValidateSet('dual21', 'grid25')][string]$Q4Network = 'dual21',
    [switch]$ConfirmPracticeReady
)
$ErrorActionPreference = 'Stop'
if (-not $ConfirmPracticeReady -or [string]::IsNullOrWhiteSpace($RobotId)) {
    throw 'No connection made. Explicit -ConfirmPracticeReady and your logged-in team RobotId are required.'
}
. (Join-Path $PSScriptRoot 'env.ps1')
& $PythonExe (Join-Path $ProjectRoot 'src\run_robot.py') --problem $Problem --robot-id $RobotId `
    --base-url $BaseUrl --q4-network $Q4Network --confirm-practice-ready
if ($LASTEXITCODE -ne 0) { throw 'Robot stopped with needs_attention; inspect the D-drive request journal.' }
