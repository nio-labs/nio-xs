use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Locates an existing GPU runner or automatically downloads and configures it in ~/.nioxs/bin/.
pub fn find_or_download_gpu_runner() -> Option<PathBuf> {
    if let Some(existing) = find_existing_runner() {
        return Some(existing);
    }
    eprintln!(
        "📦 GPU acceleration requested: downloading high-performance runner into ~/.nioxs/bin/..."
    );
    match download_and_install_runner() {
        Ok(path) => {
            eprintln!("✅ GPU runner installed successfully: {}", path.display());
            Some(path)
        }
        Err(e) => {
            eprintln!(
                "⚠️ Failed to auto-download GPU runner: {e}. Falling back to native CPU engine."
            );
            None
        }
    }
}

/// Locates an existing runner binary in ~/.nioxs/bin, env vars, or system PATH.
pub fn find_existing_runner() -> Option<PathBuf> {
    let runner_name = if cfg!(windows) {
        "llama-cli.exe"
    } else {
        "llama-cli"
    };

    if let Some(custom) =
        std::env::var_os("NIOXS_GPU_RUNNER").or_else(|| std::env::var_os("NIOXS_LLAMA_CLI"))
    {
        let p = PathBuf::from(custom);
        if p.is_file() {
            return Some(p);
        }
    }

    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        let nioxs_bin = PathBuf::from(home)
            .join(".nioxs")
            .join("bin")
            .join(runner_name);
        if nioxs_bin.is_file() {
            return Some(nioxs_bin);
        }
    }

    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|dir| dir.join(runner_name))
            .find(|path| path.is_file())
    })
}

fn download_and_install_runner() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| anyhow!("Cannot determine HOME directory"))?;
    let bin_dir = Path::new(&home).join(".nioxs").join("bin");
    std::fs::create_dir_all(&bin_dir)?;

    let runner_dir = bin_dir.join("llama-runner");
    std::fs::create_dir_all(&runner_dir)?;

    let is_windows = cfg!(target_os = "windows");
    let is_macos = cfg!(target_os = "macos");
    let is_arm = cfg!(target_arch = "aarch64");

    let tag = "b12554";
    let asset = if is_macos {
        if is_arm {
            format!("llama-{tag}-bin-macos-arm64.tar.gz")
        } else {
            format!("llama-{tag}-bin-macos-x64.tar.gz")
        }
    } else if is_windows {
        format!("llama-{tag}-bin-win-vulkan-x64.zip")
    } else if is_arm {
        format!("llama-{tag}-bin-ubuntu-vulkan-arm64.tar.gz")
    } else {
        format!("llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz")
    };

    let url = format!("https://github.com/ggerganov/llama.cpp/releases/download/{tag}/{asset}");

    if is_windows {
        let zip_path = runner_dir.join(&asset);
        let script = format!(
            "Invoke-WebRequest -Uri '{url}' -OutFile '{}'; Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
            zip_path.display(),
            zip_path.display(),
            runner_dir.display()
        );
        let ok = Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            return Err(anyhow!("Failed to download Windows GPU runner"));
        }
        let _ = std::fs::remove_file(zip_path);
        let target_exe = runner_dir.join("llama-cli.exe");
        if target_exe.is_file() {
            let sym_link = bin_dir.join("llama-cli.exe");
            let _ = std::fs::copy(&target_exe, &sym_link);
            return Ok(sym_link);
        }
    } else {
        let curl_cmd = format!(
            "curl -sL '{url}' | tar -xzf - --strip-components=1 -C '{}'",
            runner_dir.display()
        );
        let ok = Command::new("sh")
            .args(["-c", &curl_cmd])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            return Err(anyhow!("Failed to download and extract Unix GPU runner"));
        }

        // On Linux, inspect if libgomp is available; if not, attempt local apt-get download & extraction
        #[cfg(target_os = "linux")]
        {
            let libgomp_exists = Command::new("sh")
                .args(["-c", "ldconfig -p 2>/dev/null | grep -q libgomp.so.1"])
                .status()
                .map(|s| s.success())
                .unwrap_or(false);

            if !libgomp_exists {
                let deb_cmd = format!(
                    "cd '{}' && apt-get download libgomp1 libvulkan1 2>/dev/null && dpkg -x libgomp1*.deb . 2>/dev/null && dpkg -x libvulkan1*.deb . 2>/dev/null && rm -f *.deb",
                    runner_dir.display()
                );
                let _ = Command::new("sh").args(["-c", &deb_cmd]).status();
            }
        }

        // Create launcher script ~/.nioxs/bin/llama-cli
        let launcher_path = bin_dir.join("llama-cli");
        let launcher_content = r#"#!/bin/sh
DIR="$(cd "$(dirname "$0")/llama-runner" && pwd)"
export LD_LIBRARY_PATH="$DIR/usr/lib/x86_64-linux-gnu:$DIR/usr/lib/aarch64-linux-gnu:$DIR:$LD_LIBRARY_PATH"
exec "$DIR/llama-cli" "$@"
"#;
        std::fs::write(&launcher_path, launcher_content)
            .context("Failed to write llama-cli wrapper script")?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ =
                std::fs::set_permissions(&launcher_path, std::fs::Permissions::from_mode(0o755));
        }

        return Ok(launcher_path);
    }

    Err(anyhow!(
        "Completed installation but could not locate runner executable"
    ))
}
