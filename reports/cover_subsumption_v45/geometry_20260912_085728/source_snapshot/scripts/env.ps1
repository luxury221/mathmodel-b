$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ([System.IO.Path]::GetPathRoot($ProjectRoot) -ine 'D:\') {
    throw 'The experiment workspace must remain on drive D.'
}
$EnvironmentDirectories = [ordered]@{
    TEMP = '.cache\tmp'
    TMP = '.cache\tmp'
    PIP_CACHE_DIR = '.cache\pip'
    UV_CACHE_DIR = '.cache\uv'
    UV_PYTHON_INSTALL_DIR = '.runtime\python'
    UV_PYTHON_BIN_DIR = '.runtime\python-bin'
    UV_TOOL_DIR = '.runtime\uv-tools'
    UV_TOOL_BIN_DIR = '.runtime\bin'
    JUPYTER_CONFIG_DIR = '.jupyter\config'
    JUPYTER_DATA_DIR = '.jupyter\data'
    JUPYTER_RUNTIME_DIR = '.jupyter\runtime'
    IPYTHONDIR = '.ipython'
    MPLCONFIGDIR = '.cache\matplotlib'
    NUMBA_CACHE_DIR = '.cache\numba'
    JOBLIB_TEMP_FOLDER = '.cache\joblib'
    XDG_CACHE_HOME = '.cache\xdg'
    XDG_DATA_HOME = '.cache\xdg-data'
    PYTHONPYCACHEPREFIX = '.cache\pycache'
    PYTEST_DEBUG_TEMPROOT = '.cache\pytest'
    RUFF_CACHE_DIR = '.cache\ruff'
}
foreach ($EnvironmentName in $EnvironmentDirectories.Keys) {
    $DirectoryPath = Join-Path $ProjectRoot $EnvironmentDirectories[$EnvironmentName]
    New-Item -ItemType Directory -Force -Path $DirectoryPath | Out-Null
    [Environment]::SetEnvironmentVariable($EnvironmentName, $DirectoryPath, 'Process')
}
foreach ($RelativePath in @('notebooks', 'src', 'logs\practice', 'logs\formal', 'outputs', 'reports', 'simulator', '.runtime\downloads')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot $RelativePath) | Out-Null
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONNOUSERSITE = '1'
$env:UV_NO_MODIFY_PATH = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:VIRTUAL_ENV = Join-Path $ProjectRoot '.venv'
$env:PYTHONPATH = $null
$env:PYTHONHOME = $null
$env:B2026_ROOT = $ProjectRoot
$UvExe = Join-Path $ProjectRoot '.runtime\uv\uv.exe'
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$EnvironmentPathEntries = @((Join-Path $ProjectRoot '.venv\Scripts'), (Join-Path $ProjectRoot '.runtime\uv'))
$RemainingPathEntries = @($env:PATH -split ';' | Where-Object { $_ -and $_ -notin $EnvironmentPathEntries })
$env:PATH = ($EnvironmentPathEntries + $RemainingPathEntries) -join ';'
