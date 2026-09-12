param(
    [Parameter(Mandatory = $true)][string]$ArchivePath,
    [string]$SevenZipPath = 'D:\7-Zip\7z.exe'
)
. (Join-Path $PSScriptRoot 'env.ps1')
$ArchivePath = (Resolve-Path -LiteralPath $ArchivePath).Path
if ([System.IO.Path]::GetPathRoot($ArchivePath) -ine 'D:\') {
    throw 'Place the downloaded official archive on drive D before importing it.'
}
if (-not (Test-Path -LiteralPath $SevenZipPath -PathType Leaf)) { throw '7-Zip executable not found.' }
$AllowedNames = @('Jammers-simulator-full-win64.7z', 'Jammers-simulator-win64.7z')
if ([System.IO.Path]::GetFileName($ArchivePath) -notin $AllowedNames) {
    throw 'Use the original official simulator archive name.'
}
$SimulatorRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot 'simulator'))
$InstallRoot = [System.IO.Path]::GetFullPath((Join-Path $SimulatorRoot 'official'))
if (-not $InstallRoot.StartsWith($SimulatorRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Installation path is outside the simulator directory.'
}
if ((Test-Path -LiteralPath $InstallRoot) -and @(Get-ChildItem -LiteralPath $InstallRoot -Force).Count -gt 0) {
    throw 'The installation directory is not empty. Existing simulator data must not be overwritten.'
}
$Listing = & $SevenZipPath l -slt -sccUTF-8 $ArchivePath
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect the archive.' }
$EntrySection = $false
foreach ($ListingLine in $Listing) {
    if ($ListingLine -eq '----------') { $EntrySection = $true; continue }
    if (-not $EntrySection) { continue }
    if ($ListingLine -match '^(Symbolic Link|Hard Link|Reparse Point) =') {
        throw 'Links inside the archive are not allowed.'
    }
    if ($ListingLine.StartsWith('Path = ')) {
        $EntryName = $ListingLine.Substring(7)
        if ([System.IO.Path]::IsPathRooted($EntryName) -or $EntryName.Contains(':')) {
            throw 'The archive contains an absolute path or alternate data stream.'
        }
        $EntryPath = [System.IO.Path]::GetFullPath((Join-Path $InstallRoot $EntryName))
        if (-not $EntryPath.StartsWith($InstallRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'The archive contains an unsafe entry path.'
        }
    }
}
& $SevenZipPath t -sccUTF-8 $ArchivePath
if ($LASTEXITCODE -ne 0) { throw 'Archive integrity check failed.' }
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
& $SevenZipPath x -sccUTF-8 $ArchivePath ('-o' + $InstallRoot) -y
if ($LASTEXITCODE -ne 0) { throw 'Simulator extraction failed.' }
$Candidates = @(Get-ChildItem -LiteralPath $InstallRoot -Recurse -File -Filter '*.exe' | Where-Object {
    $_.BaseName -match '(?i)(jammers|simulator)' -and $_.BaseName -notmatch '(?i)(unins|setup|crash|helper)'
})
if ($Candidates.Count -ne 1) {
    Get-ChildItem -LiteralPath $InstallRoot -Recurse -File -Filter '*.exe' | Select-Object FullName
    throw 'Files were extracted, but the main executable needs manual confirmation. No program was launched.'
}
$Manifest = [ordered]@{
    executable = $Candidates[0].FullName.Substring($SimulatorRoot.Length + 1)
    archive_name = [System.IO.Path]::GetFileName($ArchivePath)
    archive_sha256 = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    source = 'https://pan.baidu.com/s/1P1yfVjY0RufU93XOdzhOLw?pwd=2026'
    imported_at = (Get-Date).ToString('o')
    login_verified = $false
    practice_test_started = $false
    formal_test_started = $false
}
$Manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $SimulatorRoot 'installation.json') -Encoding UTF8
Write-Host 'Official archive imported. No login or competition test was started.'
Write-Host 'Launch interactively with scripts\start-simulator.ps1 when ready.'
