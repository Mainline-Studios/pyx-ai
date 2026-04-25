# Pyx 1.5 - Windows installer builder (PowerShell).
# Produces:
#   dist\Pyx\Pyx.exe                  (portable bundle)
#   dist\Pyx-<version>-setup.exe      (Inno Setup installer)
#
# Requirements:
#   - Python 3.11+ on PATH
#   - Inno Setup 6+: https://jrsoftware.org/isinfo.php  (iscc on PATH)
#
# Usage (from repo root):
#   pwsh .\packaging\build-windows.ps1
#   pwsh .\packaging\build-windows.ps1 -Version 1.5.1

param(
  [string]$Version = "1.5.0"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

Write-Host "==> cleaning"
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue

Write-Host "==> venv + deps"
if (-not (Test-Path .venv)) { python -m venv .venv }
$py = Join-Path $Root ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip | Out-Null
& $py -m pip install -r requirements.txt
& $py -m pip install -r packaging/requirements-desktop.txt
& $py -m pip install "pyinstaller>=6.3"

Write-Host "==> PyInstaller"
& $py -m PyInstaller packaging\pyx.spec --noconfirm --log-level WARN

$iscc = (Get-Command iscc -ErrorAction SilentlyContinue)
if ($null -eq $iscc) {
  Write-Warning "Inno Setup 'iscc' not on PATH. Portable build ready in dist\Pyx\. Skipping installer."
} else {
  Write-Host "==> Inno Setup installer"
  & iscc "/DPyxVersion=$Version" "packaging\windows\pyx.iss"
}

Write-Host ""
Write-Host "Artifacts:"
Get-ChildItem dist -File | Format-Table Name, Length, LastWriteTime -AutoSize
