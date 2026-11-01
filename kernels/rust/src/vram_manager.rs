// ============================================================================
// ABRASAX — Unified Xenoliving System
// PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
// CREATOR: Sir Charles Spikes (BASEDGOD)
// ============================================================================

//! ABRASAX 7D — VRAM Manager
//! GPU memory allocation with auto-defragmentation.
//! ᚲᛁ — Torch and Ice runes: illumination + precision.

use std::process::Command;
use std::fmt;

/// GPU VRAM information snapshot.
#[derive(Debug, Clone)]
pub struct VramInfo {
    pub total_mb: u32,
    pub used_mb: u32,
    pub free_mb: u32,
    pub util_pct: u32,
    pub temp_c: u32,
    pub power_w: f32,
    pub clock_mhz: u32,
    pub mem_clock_mhz: u32,
    pub fan_pct: u32,
}

impl VramInfo {
    pub fn new() -> Self {
        VramInfo {
            total_mb: 6144,
            used_mb: 0,
            free_mb: 6144,
            util_pct: 0,
            temp_c: 0,
            power_w: 0.0,
            clock_mhz: 0,
            mem_clock_mhz: 0,
            fan_pct: 0,
        }
    }
    
    pub fn summary(&self) -> String {
        format!(
            "{}MB total | {}MB used | {}MB free | {}°C | {}% util",
            self.total_mb, self.used_mb, self.free_mb, self.temp_c, self.util_pct
        )
    }
}

/// VRAM allocation block.
#[derive(Debug, Clone)]
pub struct VramBlock {
    pub id: String,
    pub size_mb: u32,
    pub offset_mb: u32,
    pub model_type: String,
    pub active: bool,
}

/// VRAM Manager with defragmentation.
pub struct VramManager {
    pub info: VramInfo,
    pub blocks: Vec<VramBlock>,
    total_allocated_mb: u32,
}

impl VramManager {
    pub fn new() -> Self {
        VramManager {
            info: VramInfo::new(),
            blocks: Vec::new(),
            total_allocated_mb: 0,
        }
    }
    
    /// Poll GPU state via nvidia-smi.
    pub fn poll(&mut self) -> &VramInfo {
        match Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,fan.speed",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = stdout.trim().split(',').map(|s| s.trim()).collect();
                
                if parts.len() >= 9 {
                    self.info.used_mb = parts[0].parse().unwrap_or(0);
                    self.info.free_mb = parts[1].parse().unwrap_or(6144);
                    self.info.total_mb = parts[2].parse().unwrap_or(6144);
                    self.info.util_pct = parts[3].parse().unwrap_or(0);
                    self.info.temp_c = parts[4].parse().unwrap_or(0);
                    self.info.power_w = parts[5].parse().unwrap_or(0.0);
                    self.info.clock_mhz = parts[6].parse().unwrap_or(0);
                    self.info.mem_clock_mhz = parts[7].parse().unwrap_or(0);
                    self.info.fan_pct = parts[8].parse().unwrap_or(0);
                }
            }
            Err(e) => {
                eprintln!("nvidia-smi error: {}", e);
            }
        }
        
        &self.info
    }
    
    /// Calculate optimal GPU layers for given model.
    pub fn suggest_gpu_layers(&self, total_layers: u32) -> u32 {
        let free = self.info.free_mb;
        let layer_mb: u32 = 130; // ~130MB per layer for 8B Q3
        
        let raw_layers = free / layer_mb;
        let safe_layers = (raw_layers as f64 * 0.85) as u32;
        
        safe_layers.min(total_layers).max(0)
    }
    
    /// Check if model fits in VRAM.
    pub fn can_load(&self, model_size_mb: u32) -> bool {
        self.info.free_mb > model_size_mb + 512
    }
    
    /// Allocate a VRAM block for a model.
    pub fn allocate(&mut self, id: &str, size_mb: u32, model_type: &str) -> Option<VramBlock> {
        if !self.can_load(size_mb) {
            return None;
        }
        
        let block = VramBlock {
            id: id.to_string(),
            size_mb,
            offset_mb: self.total_allocated_mb,
            model_type: model_type.to_string(),
            active: true,
        };
        
        self.total_allocated_mb += size_mb;
        self.blocks.push(block.clone());
        
        Some(block)
    }
    
    /// Deallocate a VRAM block.
    pub fn deallocate(&mut self, id: &str) {
        if let Some(idx) = self.blocks.iter().position(|b| b.id == id) {
            let block = &mut self.blocks[idx];
            block.active = false;
            self.total_allocated_mb = self.total_allocated_mb.saturating_sub(block.size_mb);
        }
    }
    
    /// Defragment VRAM — consolidate active blocks.
    pub fn defragment(&mut self) {
        // Remove inactive blocks
        self.blocks.retain(|b| b.active);
        
        // Recompute offsets
        let mut offset = 0u32;
        for block in &mut self.blocks {
            block.offset_mb = offset;
            offset += block.size_mb;
        }
        self.total_allocated_mb = offset;
    }
    
    /// Get allocation summary.
    pub fn allocation_summary(&self) -> String {
        let active: Vec<&VramBlock> = self.blocks.iter().filter(|b| b.active).collect();
        format!(
            "{} blocks allocated | {} MB total | {} MB free of {} MB",
            active.len(),
            self.total_allocated_mb,
            self.info.free_mb,
            self.info.total_mb,
        )
    }
}

impl fmt::Display for VramManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VRAM: {}/{} MB | {}°C | {} blocks | {} MB allocated",
            self.info.used_mb,
            self.info.total_mb,
            self.info.temp_c,
            self.blocks.len(),
            self.total_allocated_mb,
        )
    }
}
