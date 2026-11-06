//! Hardware Capability & Device Dispatcher
//!
//! Automatically probes the host machine for acceleration hardware (NVIDIA CUDA,
//! Apple Metal, Vulkan) while preserving seamless, guaranteed pure-CPU execution on
//! legacy/constrained edge nodes.

use serde::{Deserialize, Serialize};
use std::path::Path;

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
                let is_intel = std::fs::read_to_string("/proc/cpuinfo")
                    .map(|c| c.contains("Intel"))
                    .unwrap_or(false);
                if is_intel || Path::new("/usr/lib/wsl/lib/libdxcore.so").exists() {
                    let vram_mb = Self::query_linux_unified_vram_mb();
                    return Some(Self::IntelArc { vram_mb });
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
            if let Ok(output) = std::process::Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
                ])
                .output()
            {
                if output.status.success() {
                    let text = String::from_utf8_lossy(&output.stdout);
                    if text.to_ascii_lowercase().contains("intel") {
                        let vram_mb = Self::query_wsl_intel_vram();
                        return Some(Self::IntelArc { vram_mb });
                    }
                }
            }
        }

        None
    }

    fn query_linux_unified_vram_mb() -> Option<u64> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = meminfo.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
        // Intel Arc iGPU shares up to 50% of unified system memory
        Some(kb / 2048)
    }

    #[allow(dead_code)]
    fn query_wsl_intel_vram() -> Option<u64> {
        let output = std::process::Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty AdapterRAM",
            ])
            .output()
            .ok()?;
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = text.lines().find(|l| !l.trim().is_empty()) {
                if let Ok(bytes) = line.trim().parse::<u64>() {
                    return Some(bytes / (1024 * 1024));
                }
            }
        }
        None
    }

    fn query_nvidia_smi_vram() -> Option<u64> {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;
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
            let output = std::process::Command::new("sysctl")
                .arg("-n")
                .arg("hw.memsize")
                .output()
                .ok()?;
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
                if let Ok(output) = std::process::Command::new("vulkaninfo")
                    .arg("--summary")
                    .output()
                {
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
