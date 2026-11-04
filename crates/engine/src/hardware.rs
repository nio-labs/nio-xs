//! Hardware Capability & Device Dispatcher
//!
//! Automatically probes the host machine for acceleration hardware (NVIDIA CUDA,
//! Apple Metal, Vulkan) while preserving seamless, guaranteed pure-CPU execution on
//! legacy/constrained edge nodes.

use std::path::Path;
use serde::{Deserialize, Serialize};

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
            Self::Metal { .. } => "Apple Metal Acceleration",
            Self::Vulkan => "Vulkan Acceleration",
        }
    }

    /// Automatically probes hardware capabilities based on user preference and environment.
    pub fn detect(pref: DevicePreference) -> Self {
        // Explicit environment override always takes priority
        if let Ok(env_pref) = std::env::var("NIOXS_DEVICE") {
            let parsed = DevicePreference::parse(&env_pref);
            if parsed == DevicePreference::Cpu {
                return Self::Cpu;
            }
        }

        if pref == DevicePreference::Cpu {
            return Self::Cpu;
        }

        // 1. Probe for NVIDIA CUDA
        if let Some(cuda_dev) = Self::probe_cuda() {
            return cuda_dev;
        }

        // 2. Probe for Apple Silicon Metal (macOS aarch64)
        if let Some(metal_dev) = Self::probe_metal() {
            return metal_dev;
        }

        // 3. Probe for Vulkan
        if let Some(vulkan_dev) = Self::probe_vulkan() {
            return vulkan_dev;
        }

        // Fallback: If GPU was strictly requested but not found, log warning and use CPU
        if pref == DevicePreference::Gpu {
            eprintln!("[nioxs] Notice: GPU acceleration requested but no compatible GPU was detected. Falling back to pure CPU.");
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
                    return Some(Self::Metal { unified_mem_mb: Some(mem_mb) });
                }
            }
            return Some(Self::Metal { unified_mem_mb: None });
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
                if let Ok(output) = std::process::Command::new("vulkaninfo").arg("--summary").output() {
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
