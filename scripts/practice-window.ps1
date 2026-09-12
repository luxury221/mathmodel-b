param(
    [ValidateSet('Capture', 'Click')][string]$Action = 'Capture',
    [int]$RelativeX = -1,
    [int]$RelativeY = -1,
    [int]$ExpectedWidth = 1280,
    [int]$ExpectedHeight = 800,
    [ValidateRange(0, 10000)][int]$DelayMilliseconds = 400,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$Label = 'practice'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'env.ps1')
Add-Type -AssemblyName System.Drawing
if (-not ('PracticePublicWindow' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class PracticePublicWindow {
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left; public int Top; public int Right; public int Bottom; }
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window, out Rect rectangle);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr window);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr window, int command);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int horizontal, int vertical);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint horizontal, uint vertical, uint data, UIntPtr extra);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr window, IntPtr dc, uint flags);
    [DllImport("user32.dll")] public static extern IntPtr SetThreadDpiAwarenessContext(IntPtr context);
}
'@
}
$SimulatorRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot 'simulator'))
$Simulator = Get-Process -Name 'jammers-simulator' -ErrorAction Stop |
    Where-Object { $_.MainWindowHandle -ne 0 -and $_.Path.StartsWith($SimulatorRoot + '\', [StringComparison]::OrdinalIgnoreCase) } |
    Select-Object -First 1
if (-not $Simulator) { throw 'Workspace simulator window not found; no action taken.' }
$Handle = $Simulator.MainWindowHandle
$OldDpi = [PracticePublicWindow]::SetThreadDpiAwarenessContext([IntPtr](-4))
try {
    if ([PracticePublicWindow]::IsIconic($Handle)) {
        [PracticePublicWindow]::ShowWindow($Handle, 9) | Out-Null
        Start-Sleep -Milliseconds 300
    }
    $Rectangle = New-Object PracticePublicWindow+Rect
    if (-not [PracticePublicWindow]::GetWindowRect($Handle, [ref]$Rectangle)) { throw 'Cannot inspect window bounds.' }
    if ($Action -eq 'Click') {
        if (($Rectangle.Right - $Rectangle.Left) -ne $ExpectedWidth -or ($Rectangle.Bottom - $Rectangle.Top) -ne $ExpectedHeight) {
            throw 'Window dimensions changed. Capture and inspect again before clicking.'
        }
        if ($RelativeX -lt 220 -or $RelativeX -ge $ExpectedWidth -or $RelativeY -lt 180 -or $RelativeY -ge $ExpectedHeight) {
            throw 'Only an observed control in the main content area may be clicked; navigation/sidebar clicks are blocked.'
        }
        [PracticePublicWindow]::SetForegroundWindow($Handle) | Out-Null
        Start-Sleep -Milliseconds 100
        if ([PracticePublicWindow]::GetForegroundWindow() -ne $Handle) { throw 'Simulator is not foreground; no click sent.' }
        if (-not [PracticePublicWindow]::SetCursorPos($Rectangle.Left + $RelativeX, $Rectangle.Top + $RelativeY)) { throw 'Cursor positioning failed.' }
        [PracticePublicWindow]::mouse_event(2, 0, 0, 0, [UIntPtr]::Zero)
        [PracticePublicWindow]::mouse_event(4, 0, 0, 0, [UIntPtr]::Zero)
        Start-Sleep -Milliseconds $DelayMilliseconds
    }
    [PracticePublicWindow]::GetWindowRect($Handle, [ref]$Rectangle) | Out-Null
    $Width = $Rectangle.Right - $Rectangle.Left
    $Height = $Rectangle.Bottom - $Rectangle.Top
    if ($Width -le 0 -or $Height -le 0 -or $Width -gt 8000 -or $Height -gt 8000) { throw 'Invalid capture bounds.' }
    $Directory = Join-Path $ProjectRoot 'outputs\practice_baseline_v1'
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
    $Path = Join-Path $Directory ($Label + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.png')
    $Bitmap = New-Object System.Drawing.Bitmap($Width, $Height)
    $Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
    $Dc = $Graphics.GetHdc()
    try { $Printed = [PracticePublicWindow]::PrintWindow($Handle, $Dc, 2) }
    finally { $Graphics.ReleaseHdc($Dc); $Graphics.Dispose() }
    try {
        if (-not $Printed) { throw 'Public window capture failed.' }
        $Bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally { $Bitmap.Dispose() }
    [pscustomobject]@{ Path = $Path; Width = $Width; Height = $Height; Action = $Action } | ConvertTo-Json
} finally {
    [PracticePublicWindow]::SetThreadDpiAwarenessContext($OldDpi) | Out-Null
}
