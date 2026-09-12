param(
    [string]$RobotId,
    [string]$BaseUrl = 'http://127.0.0.1:2026',
    [switch]$ConfirmPracticeReady
)
$ErrorActionPreference = 'Stop'
if (-not $ConfirmPracticeReady -or [string]::IsNullOrWhiteSpace($RobotId)) {
    throw 'No connection made. Confirm the visible q3 PRACTICE page is ready and supply the logged-in RobotId.'
}
. (Join-Path $PSScriptRoot 'env.ps1')
& $PythonExe (Join-Path $ProjectRoot 'src\run_candidate_practice.py') --robot-id $RobotId `
    --base-url $BaseUrl --confirm-practice-ready
if ($LASTEXITCODE -ne 0) { throw 'Candidate stopped with needs_attention; preserve and inspect the D-drive journal.' }
