// ============================================================================
// ABRASAX — Unified Xenoliving System
// PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
// CREATOR: Sir Charles Spikes (BASEDGOD)
// ============================================================================

#![allow(unused_imports)]
mod vram_manager;

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::ffi::CString;

use winapi::um::fileapi::CreateFileA;
use winapi::um::fileapi::OPEN_EXISTING;
use winapi::um::handleapi::CloseHandle;
use winapi::um::namedpipeapi::SetNamedPipeHandleState;
use winapi::um::winbase::PIPE_READMODE_MESSAGE;
use winapi::um::winnt::{GENERIC_READ, GENERIC_WRITE, FILE_SHARE_READ, FILE_SHARE_WRITE};

const PIPE_NAME: &str = "\\\\.\\pipe\\abrasax";
const PHI: f64 = 1.618033988749895;
const KAPPA: f64 = 0.6180339887498948;

#[derive(Parser)]
#[command(name = "abrasax_7d_core")]
#[command(version = "0.1.0")]
#[command(about = "Rust GPU compute engine for ABRASAX 7D")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    GpuInfo,
    Infer {
        prompt: String,
        #[arg(short, long, default_value = "256")]
        max_tokens: u32,
    },
    VramMonitor,
    RuneBench {
        #[arg(short, long, default_value = "512")]
        dim: usize,
        #[arg(short, long, default_value = "1000")]
        iterations: u32,
    },
}

#[derive(Serialize, Deserialize, Debug)]
struct PipeRequest {
    prompt: String,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize, Deserialize, Debug)]
struct PipeResponse {
    #[serde(default)]
    ok: bool,
    #[serde(default)]
    response: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::GpuInfo => cmd_gpu_info(),
        Commands::Infer { prompt, max_tokens } => cmd_infer(&prompt, max_tokens),
        Commands::VramMonitor => cmd_vram_monitor(),
        Commands::RuneBench { dim, iterations } => cmd_rune_bench(dim, iterations),
    }
}

fn cmd_gpu_info() {
    println!("ABRASAX 7D - GPU Enumeration");
    println!("  phi = {:.16}", PHI);
    println!("  K = {:.16}", KAPPA);
    println!("  VRAM: {}", vram_manager::VramInfo::new().summary());
}

fn cmd_infer(prompt: &str, max_tokens: u32) {
    println!("Connecting to pipe: {}", PIPE_NAME);
    let pipe_name = CString::new(PIPE_NAME).unwrap();
    unsafe {
        let handle = CreateFileA(
            pipe_name.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            std::ptr::null_mut(),
            OPEN_EXISTING,
            0,
            std::ptr::null_mut(),
        );
        if handle == winapi::um::handleapi::INVALID_HANDLE_VALUE {
            eprintln!("Failed to connect to pipe");
            return;
        }
        let mut mode: u32 = PIPE_READMODE_MESSAGE;
        SetNamedPipeHandleState(handle, &mut mode, std::ptr::null_mut(), std::ptr::null_mut());
        let request = PipeRequest { prompt: prompt.to_string(), max_tokens, temperature: 0.3 };
        let json = serde_json::to_string(&request).unwrap() + "\n";
        let mut written: u32 = 0;
        winapi::um::fileapi::WriteFile(handle, json.as_ptr() as *const _, json.len() as u32, &mut written, std::ptr::null_mut());
        let mut buf = vec![0u8; 65536];
        let mut read: u32 = 0;
        winapi::um::fileapi::ReadFile(handle, buf.as_mut_ptr() as *mut _, buf.len() as u32, &mut read, std::ptr::null_mut());
        CloseHandle(handle);
        let response_str = String::from_utf8_lossy(&buf[..read as usize]);
        match serde_json::from_str::<PipeResponse>(&response_str) {
            Ok(resp) => {
                if resp.ok { println!("Response: {}", resp.response.unwrap_or_default()); }
                else { eprintln!("Error: {}", resp.error.unwrap_or_default()); }
            }
            Err(e) => { eprintln!("Parse error: {:?}", e); }
        }
    }
}

fn cmd_vram_monitor() {
    println!("VRAM Monitor - ABRASAX 7D");
    let mut manager = vram_manager::VramManager::new();
    loop {
        let _ = manager.poll();
        println!("  VRAM: {}/{} MB used | {} MB free | {}C | {}% util",
            manager.info.used_mb, manager.info.total_mb,
            manager.info.free_mb, manager.info.temp_c, manager.info.util_pct);
        println!("  Optimal GPU layers: {} (8B model)", manager.suggest_gpu_layers(33));
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

fn cmd_rune_bench(dim: usize, iterations: u32) {
    println!("Rune Tensor Benchmark");
    println!("  Dim: {} | Iters: {} | phi = {:.16} | K = {:.16}", dim, iterations, PHI, KAPPA);
    let start = Instant::now();
    let mut sum: f64 = 0.0;
    for i in 0..iterations {
        let rune_val = ((i % 8) as f64 + 1.0) * KAPPA;
        let harmonic = (i as f64 * PHI).sin() * rune_val;
        sum += harmonic.abs();
    }
    let elapsed = start.elapsed();
    println!("  Result: {:.6} | Time: {:.3}ms | Ops/sec: {:.0}",
        sum, elapsed.as_secs_f64() * 1000.0, iterations as f64 / elapsed.as_secs_f64());
}