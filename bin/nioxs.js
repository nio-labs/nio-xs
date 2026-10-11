#!/usr/bin/env node

/**
 * NioXS NPX Runner
 * Automatically detects platform (Linux, macOS, Windows, Termux),
 * downloads or locates the native 5.5MB nioxs standalone binary,
 * and launches it directly.
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const https = require('https');

const REPO = 'nio-labs/nio-xs';
const VERSION = process.env.NIOXS_VERSION || 'v0.2.0';

// 1. Detect Operating System and Architecture
function getPlatformInfo() {
  const nodePlatform = process.platform;
  const nodeArch = process.arch;

  let osType = '';
  let isTermux = false;
  let isWindows = false;

  if (process.env.TERMUX_VERSION || fs.existsSync('/data/data/com.termux')) {
    isTermux = true;
    osType = 'termux';
  } else if (nodePlatform === 'win32') {
    isWindows = true;
    osType = 'windows';
  } else if (nodePlatform === 'darwin') {
    osType = 'darwin';
  } else {
    osType = 'linux';
  }

  let arch = '';
  if (nodeArch === 'x64') {
    arch = 'amd64';
  } else if (nodeArch === 'arm64') {
    arch = 'arm64';
  } else {
    arch = nodeArch;
  }

  return { osType, arch, isWindows, isTermux };
}

// 2. Download helper with redirect handling
function downloadBinary(url, dest) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return downloadBinary(res.headers.location, dest).then(resolve).catch(reject);
      }
      if (res.statusCode !== 200) {
        return reject(new Error(`Failed to download binary from ${url} (HTTP ${res.statusCode})`));
      }
      const fileStream = fs.createWriteStream(dest);
      res.pipe(fileStream);
      fileStream.on('finish', () => {
        fileStream.close();
        resolve();
      });
      fileStream.on('error', reject);
    }).on('error', reject);
  });
}

async function findOrDownloadBinary() {
  const { osType, arch, isWindows, isTermux } = getPlatformInfo();
  const binaryName = isWindows ? 'nioxs.exe' : 'nioxs';

  // Check 1: In local workspace target/release
  const localBuild = path.resolve(__dirname, '..', 'target', 'release', binaryName);
  if (fs.existsSync(localBuild)) {
    return localBuild;
  }

  // Check 2: Global PATH
  const whichCmd = isWindows ? 'where' : 'which';
  const whichResult = spawnSync(whichCmd, ['nioxs'], { stdio: 'pipe' });
  if (whichResult.status === 0) {
    const foundPath = whichResult.stdout.toString().split('\n')[0].trim();
    if (foundPath && fs.existsSync(foundPath)) {
      return foundPath;
    }
  }

  // Check 3: Termux $PREFIX/bin/nioxs
  if (isTermux) {
    const prefixBin = path.join(process.env.PREFIX || '/data/data/com.termux/files/usr', 'bin', binaryName);
    if (fs.existsSync(prefixBin)) {
      return prefixBin;
    }
  }

  // Check 4: User cache directory ~/.nioxs/bin/nioxs
  const userHome = os.homedir();
  const cacheDir = path.join(userHome, '.nioxs', 'bin');
  if (!fs.existsSync(cacheDir)) {
    fs.mkdirSync(cacheDir, { recursive: true });
  }

  const cachedBinary = path.join(cacheDir, binaryName);
  if (fs.existsSync(cachedBinary)) {
    return cachedBinary;
  }

  // Download binary from GitHub Releases
  console.log(`[nioxs] Initializing NioNano companion binary for ${osType}-${arch}...`);
  const assetName = isWindows
    ? `nioxs-${osType}-${arch}.exe`
    : `nioxs-${osType}-${arch}`;
  
  const downloadUrl = `https://github.com/${REPO}/releases/download/${VERSION}/${assetName}`;

  try {
    console.log(`[nioxs] Fetching from ${downloadUrl}...`);
    await downloadBinary(downloadUrl, cachedBinary);
    if (!isWindows) {
      fs.chmodSync(cachedBinary, 0o755);
    }
    console.log(`[nioxs] Successfully installed to ${cachedBinary}`);
    return cachedBinary;
  } catch (err) {
    // If remote release asset is not ready, check if cargo is available to build locally
    console.warn(`[nioxs] Remote binary download failed: ${err.message}`);
    console.log(`[nioxs] Checking for local Rust/Cargo installation...`);
    const cargoCheck = spawnSync('cargo', ['--version'], { stdio: 'pipe' });
    if (cargoCheck.status === 0) {
      console.log(`[nioxs] Compiling locally with cargo build --release...`);
      const repoRoot = path.resolve(__dirname, '..');
      const buildResult = spawnSync('cargo', ['build', '--release', '--bin', 'nioxs'], {
        cwd: repoRoot,
        stdio: 'inherit',
      });
      if (buildResult.status === 0 && fs.existsSync(localBuild)) {
        return localBuild;
      }
    }
    throw new Error(`Could not find or download nioxs binary. Please run 'cargo build --release' or install via install.sh.`);
  }
}

async function main() {
  try {
    const binaryPath = await findOrDownloadBinary();
    const args = process.argv.slice(2);

    const child = spawnSync(binaryPath, args, {
      stdio: 'inherit',
      env: process.env,
    });

    process.exit(child.status !== null ? child.status : 0);
  } catch (err) {
    console.error(`[nioxs error] ${err.message}`);
    process.exit(1);
  }
}

main();
