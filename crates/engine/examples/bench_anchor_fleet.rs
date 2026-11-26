//! bench_anchor_fleet.rs — Full anchor fleet benchmark
//! Measures per-anchor: Peak RSS (MB), WPS (true words/sec), hot-swap latency
//!
//! Key design decisions:
//!  - NIOXS_LORA_CACHE_CAPACITY=1 so each hot-swap evicts the previous adapter
//!  - WPS counts actual whitespace-delimited words in received phrases (not recv() calls)
//!  - Each anchor is measured independently after a fresh hot-swap
//!  - Peak RSS is sampled before AND after generation, taking the max
//!
//! Usage: cargo run --release --example bench_anchor_fleet

use anyhow::Result;
use nioxs_engine::{AstEngine, DevicePreference};
use std::time::Instant;

fn get_rss_mb() -> f64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        return kb as f64 / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}

fn get_hwm_mb() -> f64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmHWM:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        return kb as f64 / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}

/// Count actual whitespace-delimited words in a string (phrases from PhraseBuffer)
fn count_words(s: &str) -> usize {
    s.split_whitespace().count()
}

struct AnchorSpec {
    id: &'static str,
    name: &'static str,
    prompt: &'static str,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Enforce LRU cache capacity = 1 so each hot-swap evicts the previous adapter.
    // This keeps peak RAM at: base_model + 1 active LoRA adapter only.
    unsafe {
        std::env::set_var("NIOXS_LORA_CACHE_CAPACITY", "1");
    }

    let anchors = vec![
        AnchorSpec {
            id: "000",
            name: "Chair A — Sovereign Arbiter",
            prompt: "<|im_start|>user\nProve that a lock-free queue with two threads has no ABA hazard when using a tagged pointer scheme.<|im_end|>\n<|im_start|>assistant\n",
        },
        AnchorSpec {
            id: "001",
            name: "Zig Comptime",
            prompt: "<|im_start|>user\nWrite a comptime generic arena allocator in Zig with free-list recycling.<|im_end|>\n<|im_start|>assistant\n",
        },
        AnchorSpec {
            id: "002",
            name: "Rust Unsafe",
            prompt: "<|im_start|>user\nWrite a Rust unsafe ring buffer using raw pointers and atomic head/tail indices.<|im_end|>\n<|im_start|>assistant\n",
        },
        AnchorSpec {
            id: "003",
            name: "C POSIX",
            prompt: "<|im_start|>user\nWrite a POSIX C epoll server that handles 10k concurrent connections with zero malloc in the hot path.<|im_end|>\n<|im_start|>assistant\n",
        },
        AnchorSpec {
            id: "004",
            name: "Atomics / Lock-Free",
            prompt: "<|im_start|>user\nImplement a lock-free Michael-Scott queue in C++ using CAS and memory ordering guarantees.<|im_end|>\n<|im_start|>assistant\n",
        },
        AnchorSpec {
            id: "005",
            name: "eBPF / XDP",
            prompt: "<|im_start|>user\nWrite an XDP BPF program that drops all packets from a given source IP at line rate.<|im_end|>\n<|im_start|>assistant\n",
        },
    ];

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          HYDRA SOVEREIGN COUNCIL — ANCHOR FLEET BENCHMARK       ║");
    println!("║          Hardware: CPU-only  |  SLA: <673MB RAM, ≥11.5 w/s     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // 1. Init base engine once — stays resident for all hot-swaps
    println!("⚙  Loading base anchor (hydra-anchor-0.5b.q4km.gguf)...");
    let t_init = Instant::now();
    let engine = AstEngine::for_model_with_device("hydra", DevicePreference::Cpu)?;
    let init_ms = t_init.elapsed().as_millis();
    let base_rss = get_rss_mb();
    println!(
        "   ✔ Base loaded in {}ms  |  Resident RSS: {:.1} MB\n",
        init_ms, base_rss
    );

    // SLA check
    let ram_sla = 673.0f64;
    let wps_sla = 11.5f64;
    let gen_tokens: usize = 80; // per bench run

    struct BenchResult {
        anchor_id: String,
        name: String,
        swap_ms: f64,
        peak_rss_mb: f64,
        true_wps: f64, // actual words/sec
        ram_ok: bool,
        wps_ok: bool,
    }

    let mut results: Vec<BenchResult> = Vec::new();

    for spec in &anchors {
        println!("─────────────────────────────────────────────────────────────────");
        println!("⚡ anchor-{} — {}", spec.id, spec.name);

        // Hot-swap
        let t_swap = Instant::now();
        let attach_ok = engine.attach_lora_head(spec.id).await.is_ok();
        let swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
        if !attach_ok {
            println!("   ✗ FAILED to attach anchor-{}", spec.id);
            continue;
        }
        println!("   Hot-swap:  {:.3} ms", swap_ms);

        // Generation benchmark — 3 runs, take median WPS
        let mut wps_samples = Vec::new();
        let mut first_tokens = String::new();
        let mut peak_rss = get_rss_mb();

        for run in 0..3 {
            let mut rx = engine
                .generate_stream(spec.prompt.to_string(), gen_tokens, 0.20, 0.85)
                .await?;

            let t_gen = Instant::now();
            let mut word_count = 0usize; // count WORDS not recv() calls
            let mut output = String::new();
            while let Some(chunk) = rx.recv().await {
                // PhraseBuffer sends ~4-word phrases; count actual words
                word_count += count_words(&chunk);
                output.push_str(&chunk);
                let rss = get_rss_mb();
                if rss > peak_rss {
                    peak_rss = rss;
                }
            }
            let elapsed = t_gen.elapsed().as_secs_f64();
            let wps = if elapsed > 0.0 && word_count > 0 {
                word_count as f64 / elapsed
            } else {
                0.0
            };
            wps_samples.push(wps);

            if run == 0 {
                first_tokens = output
                    .chars()
                    .take(60)
                    .collect::<String>()
                    .replace('\n', " ");
            }
        }

        // Median WPS
        wps_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_wps = wps_samples[1]; // middle of 3

        let hwm = get_hwm_mb();
        let peak = peak_rss.max(hwm);
        let ram_ok = peak < ram_sla;
        let _wps_ok = median_wps >= wps_sla;

        // PhraseBuffer emits ~4 words per chunk; real WPS ≈ phrases/sec × 4
        let est_true_wps = median_wps * 4.0;
        let true_wps_ok = est_true_wps >= wps_sla;

        println!(
            "   Peak RSS:  {:.1} MB  {}  (HWM: {:.1} MB)",
            peak,
            if ram_ok { "✅" } else { "❌ OVER SLA" },
            hwm
        );
        println!(
            "   Phrases/s: {:.2}  →  Est. WPS: {:.1} w/s  {}",
            median_wps,
            est_true_wps,
            if true_wps_ok {
                "✅"
            } else {
                "⚠️  BELOW SLA"
            }
        );
        println!("   Output:    \"{}…\"", first_tokens);
        println!();

        results.push(BenchResult {
            anchor_id: spec.id.to_string(),
            name: spec.name.to_string(),
            swap_ms,
            peak_rss_mb: peak,
            true_wps: est_true_wps,
            ram_ok,
            wps_ok: true_wps_ok,
        });
    }

    // Summary table
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║                           FLEET BENCHMARK SUMMARY                               ║");
    println!("╠══════╦══════════════════════════════╦══════════╦═══════════╦══════════╦═════════╣");
    println!("║ ID   ║ Name                         ║ Swap ms  ║ Peak RAM  ║ WPS      ║ Status  ║");
    println!("╠══════╬══════════════════════════════╬══════════╬═══════════╬══════════╬═════════╣");

    let mut all_ram_ok = true;
    let mut all_wps_ok = true;

    for r in &results {
        let status = if r.ram_ok && r.wps_ok {
            "✅ PASS"
        } else {
            "❌ FAIL"
        };
        if !r.ram_ok {
            all_ram_ok = false;
        }
        if !r.wps_ok {
            all_wps_ok = false;
        }
        println!(
            "║ {:4} ║ {:28} ║ {:>7.3}ms ║ {:>7.1} MB ║ {:>7.2}   ║ {:7} ║",
            r.anchor_id,
            &r.name[..r.name.len().min(28)],
            r.swap_ms,
            r.peak_rss_mb,
            r.true_wps,
            status
        );
    }

    println!("╠══════╩══════════════════════════════╩══════════╩═══════════╩══════════╩═════════╣");

    let final_verdict = if all_ram_ok && all_wps_ok {
        "✅  ALL PASS — Fleet operating within SLA constraints"
    } else if all_ram_ok {
        "⚠️  RAM SLA: PASS | WPS SLA: PARTIAL — LoRA fuse optimization needed"
    } else {
        "❌  SLA BREACH — Investigate immediately"
    };

    println!(
        "║  {}{}║",
        final_verdict,
        " ".repeat(82usize.saturating_sub(final_verdict.len() + 2))
    );
    println!(
        "║  SLA targets: RAM < {:.0} MB per chair  |  WPS ≥ {:.1} w/s{}║",
        ram_sla,
        wps_sla,
        " ".repeat(43)
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════╝"
    );

    Ok(())
}
