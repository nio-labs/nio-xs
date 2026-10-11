# ==============================================================================
# NioXS Windows PowerShell Installer
# Run via: irm https://raw.githubusercontent.com/nio-labs/nio-xs/main/install.ps1 | iex
# ==============================================================================

$ErrorActionPreference = "Stop"

$Repo = "nio-labs/nio-xs"
$Version = if ($env:NIOXS_VERSION) { $env:NIOXS_VERSION } else { "latest" }

Write-Host "==> Detecting Windows Architecture..." -ForegroundColor Green
$Arch = if ([System.Environment]::Is64BitOperatingSystem) { "amd64" } else { "x86" }
if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
    $Arch = "arm64"
}

$InstallDir = Join-Path $env:USERPROFILE ".nioxs\bin"
if (!(Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

$TargetExe = Join-Path $InstallDir "nioxs.exe"

# Check if local release build exists in current dir
$LocalRelease = "target\release\nioxs.exe"
if (Test-Path $LocalRelease) {
    Write-Host "==> Found local release binary in workspace. Copying..." -ForegroundColor Green
    Copy-Item $LocalRelease $TargetExe -Force
} else {
    $DownloadUrl = if ($Version -eq "latest") {
        "https://github.com/$Repo/releases/latest/download/nioxs-windows-$Arch.exe"
    } else {
        "https://github.com/$Repo/releases/download/$Version/nioxs-windows-$Arch.exe"
    }

    Write-Host "==> Downloading NioXS from $DownloadUrl..." -ForegroundColor Green
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $TargetExe -UseBasicParsing
    } catch {
        Write-Warning "Could not download remote binary ($_.Exception.Message)."
        if (Get-Command cargo -ErrorAction SilentlyContinue) {
            Write-Host "==> Compiling locally via Cargo..." -ForegroundColor Yellow
            cargo install --path crates/provider --bin nioxs --root (Join-Path $env:USERPROFILE ".nioxs")
        } else {
            Write-Error "Please install Rust (cargo) or ensure GitHub releases are published."
        }
    }
}

# Bundle and install NioDB for state persistence
$NioDbExe = Join-Path $InstallDir "niodb.exe"
if (!(Test-Path $NioDbExe) -and !(Get-Command niodb -ErrorAction SilentlyContinue)) {
    Write-Host "==> Bundling NioDB for Windows ($Arch)..." -ForegroundColor Green
    $DbArch = if ($Arch -eq "amd64") { "x64" } else { $Arch }
    $NioDbUrl = "https://github.com/nio-labs/nio-db/releases/latest/download/niodb-win32-$DbArch.exe"
    try {
        Invoke-WebRequest -Uri $NioDbUrl -OutFile $NioDbExe -UseBasicParsing
        Write-Host "==> NioDB bundled successfully at $NioDbExe" -ForegroundColor Green
    } catch {
        Write-Warning "Could not auto-download NioDB directly ($_.Exception.Message)."
    }
}

# Add to user PATH if not present
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    Write-Host "==> Adding $InstallDir to User PATH..." -ForegroundColor Cyan
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    $env:Path += ";$InstallDir"
}

Write-Host "==> Verifying Installation..." -ForegroundColor Green
& $TargetExe --help | Select-Object -First 5

Write-Host "`n✔ NioXS is ready on Windows! Run 'nioxs' to start." -ForegroundColor Green
