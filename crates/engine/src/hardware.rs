//! Hardware Capability & Device Dispatcher
//!
//! Automatically probes the host machine for acceleration hardware (NVIDIA CUDA,
//! Apple Metal, Vulkan) while preserving seamless, guaranteed pure-CPU execution on
//! legacy/constrained edge nodes.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[test]
    fn adapter_inventory_handles_multiple_devices_and_bad_rows() {
        let devices = parse_host_adapters(
            "Intel Arc|2147483648\r\nNVIDIA GPU|4294967296\nmissing separator\nBad|unknown\n",
        );
        assert_eq!(
            devices,
            vec![("Intel Arc".into(), 2048), ("NVIDIA GPU".into(), 4096)]
        );
    }

    #[cfg(unix)]
    #[test]
    fn probe_collects_completed_output() {
        let result = probe_output(
            Command::new("sh").args(["-c", "printf ready"]),
            Duration::from_secs(1),
        )
        .unwrap();
        assert!(result.status.success());
        assert_eq!(result.stdout, b"ready");
    }

    #[cfg(unix)]
    #[test]
    fn stalled_probe_is_killed_and_reaped() {
        let started = Instant::now();
        assert!(
            probe_output(
                Command::new("sh").args(["-c", "exec sleep 5"]),
                Duration::from_millis(50)
            )
            .is_none()
        );
        assert!(started.elapsed() < Duration::from_secs(2));
    }
}

/// Bound tool discovery so a hung driver probe cannot block startup indefinitely.
fn probe_output(command: &mut Command, timeout: Duration) -> Option<Output> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return child.wait_with_output().ok(),
            Ok(None) if started.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(10))
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn parse_host_adapters(text: &str) -> Vec<(String, u64)> {
    text.lines()
        .filter_map(|line| {
            let (name, memory) = line.trim().rsplit_once('|')?;
            let bytes = memory.trim().parse::<u64>().ok()?;
            Some((name.trim().to_owned(), bytes / (1024 * 1024)))
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Gpu,
}

impl Default for DevicePreference {
    fn default() -> Self {
        Self::Auto
    }
}

impl DevicePreference {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "gpu" | "cuda" | "metal" | "vulkan" => Self::Gpu,
            "cpu" => Self::Cpu,
            _ => Self::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActiveDevice {
    Cpu,
    Cuda { vram_mb: Option<u64> },
    IntelArc { vram_mb: Option<u64> },
    Metal { unified_mem_mb: Option<u64> },
    Vulkan,
}

impl ActiveDevice {
    pub fn is_gpu(&self) -> bool {
        !matches!(self, Self::Cpu)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "Pure-CPU (AST-Micro AVX2)",
            Self::Cuda { .. } => "NVIDIA CUDA Acceleration",
            Self::IntelArc { .. } => "Intel Arc GPU (DirectX/Vulkan Acceleration)",
            Self::Metal { .. } => "Apple Metal Acceleration",
            Self::Vulkan => "Vulkan Acceleration",
        }
    }

    /// Automatically probes hardware capabilities based on user preference and environment.
    pub fn detect(pref: DevicePreference) -> Self {
        // Explicit environment override always takes priority
        let effective_pref = if let Ok(env_pref) = std::env::var("NIOXS_DEVICE") {
            DevicePreference::parse(&env_pref)
        } else {
            pref
        };

        if effective_pref == DevicePreference::Cpu {
            return Self::Cpu;
        }

        // 1. Probe for NVIDIA CUDA
        if let Some(cuda_dev) = Self::probe_cuda() {
            return cuda_dev;
        }

        // 2. Probe for Intel Arc / Intel GPU (DirectX / Level Zero / Vulkan)
        if let Some(intel_dev) = Self::probe_intel_arc() {
            return intel_dev;
        }

        // 3. Probe for Apple Silicon Metal (macOS aarch64)
        if let Some(metal_dev) = Self::probe_metal() {
            return metal_dev;
        }

        // 4. Probe for Vulkan
        if let Some(vulkan_dev) = Self::probe_vulkan() {
            return vulkan_dev;
        }

        // Fallback: If GPU was strictly requested but not found, log warning and use CPU
        if effective_pref == DevicePreference::Gpu {
            eprintln!(
                "[nioxs] Notice: GPU acceleration requested but no compatible GPU was detected. Falling back to pure CPU."
            );
        }

        Self::Cpu
    }

    fn probe_cuda() -> Option<Self> {
        // Fast path: Check for NVIDIA driver nodes in /dev or /proc on Linux
        #[cfg(target_os = "linux")]
        {
            if Path::new("/dev/nvidia0").exists()
                || Path::new("/dev/nvidiactl").exists()
                || Path::new("/proc/driver/nvidia/version").exists()
            {
                // Query nvidia-smi for VRAM if available
                let vram_mb = Self::query_nvidia_smi_vram();
                return Some(Self::Cuda { vram_mb });
            }

            // WSL exposes GPUs through /dev/dxg rather than /dev/nvidia*.
            // Ask Windows which adapter is present so an Intel CPU or the
            // generic libdxcore shim does not hide an NVIDIA GPU.
            if Path::new("/dev/dxg").exists() {
                if let Some(vram_mb) = Self::query_host_adapter_vram("nvidia") {
                    return Some(Self::Cuda {
                        vram_mb: Some(vram_mb),
                    });
                }
            }
        }

        // Windows path: Check for nvcuda.dll or nvidia-smi
        #[cfg(target_os = "windows")]
        {
            if let Ok(sys_root) = std::env::var("SystemRoot") {
                let nvcuda = Path::new(&sys_root).join("System32").join("nvcuda.dll");
                if nvcuda.exists() {
                    let vram_mb = Self::query_nvidia_smi_vram();
                    return Some(Self::Cuda { vram_mb });
                }
            }
        }

        None
    }

    fn probe_intel_arc() -> Option<Self> {
        // Fast path 1: WSL2 DirectX kernel passthrough (/dev/dxg)
        #[cfg(target_os = "linux")]
        {
            if Path::new("/dev/dxg").exists() {
                if let Some(vram_mb) = Self::query_host_adapter_vram("arc") {
                    return Some(Self::IntelArc {
                        vram_mb: Some(vram_mb),
                    });
                }
            }

            // Fast path 2: Native Linux DRI render node
            if Path::new("/dev/dri/renderD128").exists() {
                if let Ok(vendor) = std::fs::read_to_string("/sys/class/drm/card0/device/vendor") {
                    if vendor.trim().contains("0x8086") {
                        let vram_mb = Self::query_linux_unified_vram_mb();
                        return Some(Self::IntelArc { vram_mb });
                    }
                }
            }
        }

        // Windows native path
        #[cfg(target_os = "windows")]
        {
            if let Some(vram_mb) = Self::query_host_adapter_vram("arc") {
                return Some(Self::IntelArc {
                    vram_mb: Some(vram_mb),
                });
            }
        }

        None
    }

    #[cfg(target_os = "linux")]
    fn query_linux_unified_vram_mb() -> Option<u64> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = meminfo.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
        Some(kb / 2048)
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn query_host_adapter_vram(adapter_name: &str) -> Option<u64> {
        static ADAPTERS: std::sync::OnceLock<Vec<(String, u64)>> = std::sync::OnceLock::new();
        let adapters = ADAPTERS.get_or_init(|| {
            #[cfg(target_os = "linux")]
            let shell = "powershell.exe";
            #[cfg(target_os = "windows")]
            let shell = "powershell";
            probe_output(Command::new(shell).args([
                "-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { \"{0}|{1}\" -f $_.Name,$_.AdapterRAM }",
            ]), Duration::from_secs(3))
                .filter(|output| output.status.success())
                .map(|output| parse_host_adapters(&String::from_utf8_lossy(&output.stdout)))
                .unwrap_or_default()
        });
        adapters
            .iter()
            .find(|(name, _)| name.to_ascii_lowercase().contains(adapter_name))
            .map(|(_, memory)| *memory)
    }

    fn query_nvidia_smi_vram() -> Option<u64> {
        let output = probe_output(
            Command::new("nvidia-smi")
                .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"]),
            Duration::from_secs(2),
        )?;
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            if let Some(first_line) = text.lines().next() {
                return first_line.trim().parse::<u64>().ok();
            }
        }
        None
    }

    fn probe_metal() -> Option<Self> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // Apple Silicon unified memory
            let output = probe_output(
                Command::new("sysctl").arg("-n").arg("hw.memsize"),
                Duration::from_secs(2),
            )?;
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                if let Ok(bytes) = text.trim().parse::<u64>() {
                    let mem_mb = bytes / (1024 * 1024);
                    return Some(Self::Metal {
                        unified_mem_mb: Some(mem_mb),
                    });
                }
            }
            return Some(Self::Metal {
                unified_mem_mb: None,
            });
        }
        #[allow(unreachable_code)]
        None
    }

    fn probe_vulkan() -> Option<Self> {
        // Vulkan ICD check
        #[cfg(target_os = "linux")]
        {
            if Path::new("/usr/share/vulkan/icd.d").exists()
                || Path::new("/etc/vulkan/icd.d").exists()
            {
                // Verify vulkaninfo runs successfully
                if let Some(output) = probe_output(
                    Command::new("vulkaninfo").arg("--summary"),
                    Duration::from_secs(2),
                ) {
                    if output.status.success() {
                        return Some(Self::Vulkan);
                    }
                }
            }
        }
        None
    }

    /// Calculates the recommended number of layers to offload to GPU.
    pub fn recommended_gpu_layers(&self, total_layers: usize) -> usize {
        match self {
            Self::Cpu => 0,
            Self::Cuda { vram_mb } => {
                if let Some(vram) = vram_mb {
                    if *vram >= 2048 {
                        total_layers // Full offload (e.g. 28 layers for Beluga, 24 for Manta)
                    } else if *vram >= 1024 {
                        total_layers / 2 // Partial offload
                    } else {
                        0 // VRAM too constrained; keep on CPU
                    }
                } else {
                    total_layers
                }
            }
            Self::IntelArc { vram_mb } => {
                if let Some(vram) = vram_mb {
                    if *vram >= 2048 {
                        total_layers // Full offload (28 layers for Beluga, 24 for Manta on 2GB+ Intel Arc)
                    } else if *vram >= 1024 {
                        total_layers / 2 // Partial offload
                    } else {
                        0 // VRAM too constrained; keep on CPU
                    }
                } else {
                    total_layers
                }
            }
            Self::Metal { unified_mem_mb } => {
                if let Some(mem) = unified_mem_mb {
                    if *mem >= 8192 {
                        total_layers
                    } else {
                        total_layers / 2
                    }
                } else {
                    total_layers
                }
            }
            Self::Vulkan => total_layers,
        }
    }
}
