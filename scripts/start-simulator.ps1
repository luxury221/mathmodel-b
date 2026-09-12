param(
    [string]$ExecutablePath,
    [ValidateSet('Normal', 'Hidden')][string]$WindowStyle = 'Normal',
    [switch]$PassThru
)
. (Join-Path $PSScriptRoot 'env.ps1')
$SimulatorRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot 'simulator'))
if (-not $ExecutablePath) {
    $ManifestPath = Join-Path $SimulatorRoot 'installation.json'
    if (-not (Test-Path -LiteralPath $ManifestPath)) {
        throw 'The official simulator is not installed yet. See README_ENV.md.'
    }
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    $ExecutablePath = Join-Path $SimulatorRoot $Manifest.executable
}
$ExecutablePath = [System.IO.Path]::GetFullPath($ExecutablePath)
if (-not $ExecutablePath.StartsWith($SimulatorRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'The simulator executable must be located inside the project simulator directory.'
}
if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) { throw 'Simulator executable not found.' }
$ProcessName = [System.IO.Path]::GetFileNameWithoutExtension($ExecutablePath)
$ExistingProcess = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue | Where-Object { $_.Path -ieq $ExecutablePath }
if ($ExistingProcess) { throw 'This simulator is already running. Use its existing window.' }
$EnvironmentOverrides = @{
    APPDATA = Join-Path $ProjectRoot '.cache\simulator\roaming'
    LOCALAPPDATA = Join-Path $ProjectRoot '.cache\simulator\local'
    WEBVIEW2_USER_DATA_FOLDER = Join-Path $ProjectRoot '.cache\simulator\webview2'
}
$PreviousEnvironment = @{}
try {
    foreach ($EnvironmentName in $EnvironmentOverrides.Keys) {
        $PreviousEnvironment[$EnvironmentName] = [Environment]::GetEnvironmentVariable($EnvironmentName, 'Process')
        New-Item -ItemType Directory -Force -Path $EnvironmentOverrides[$EnvironmentName] | Out-Null
        [Environment]::SetEnvironmentVariable($EnvironmentName, $EnvironmentOverrides[$EnvironmentName], 'Process')
    }
    $SimulatorProcess = Start-Process -FilePath $ExecutablePath -WorkingDirectory (Split-Path -Parent $ExecutablePath) -WindowStyle $WindowStyle -PassThru
    if ($PassThru) { $SimulatorProcess }
}
finally {
    foreach ($EnvironmentName in $PreviousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($EnvironmentName, $PreviousEnvironment[$EnvironmentName], 'Process')
    }
}
