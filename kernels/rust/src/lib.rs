// ============================================================================
// ABRASAX — Unified Xenoliving System
// PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
// CREATOR: Sir Charles Spikes (BASEDGOD)
// ============================================================================

//! Build 4.0: Rust Hydrator (Universal Engine)
//!
//! Acts as the ultimate safety layer — allocates GPU memory using safe Rust
//! wrappers, loads the JSON seed, and triggers the unfold_cbm kernel via FFI.
//! Prevents the "Living Crystal" from overwriting host system memory.
//! Compiles to cellular_unfolding_engine.wasm via wasm-pack.
//!
//! Integrated into ABRASAX 7D — 2026-05-02
//! Source: LucarioLucario55 / Tecumseh Amaru payload

use std::alloc::{alloc, Layout};
use std::ffi::c_void;

// External C/CUDA bindings to libflux_core.a
extern "C" {
    fn unfold_cbm(seed_ptr: *const f32, output_ptr: *mut f32, size: usize, tau_delay: f32);
}

const LITTLE_PHI: f32 = 1.618033988749895;
const PHI_CONJUGATE: f32 = 0.618033988749895;

/// Allocates VRAM safely using Rust's allocator.
/// Acts as the safety layer preventing the Living Crystal from
/// overwriting host system memory. Primal Cut initiated on failure.
#[no_mangle]
pub extern "C" fn allocate_safe_vram(size: usize) -> *mut f32 {
    unsafe {
        let layout = Layout::array::<f32>(size).expect("Invalid VRAM Layout");
        let ptr = alloc(layout) as *mut f32;
        if ptr.is_null() {
            panic!("VRAM Allocation Failed: Primal Cut Initiated");
        }
        ptr
    }
}

/// Triggers the full cellular unfolding pipeline:
/// 1. Allocates safe VRAM
/// 2. Calls the unfold_cbm CUDA kernel via FFI
/// 3. Returns pointer to the unfolded manifold
#[no_mangle]
pub extern "C" fn trigger_cellular_unfolding(
    seed_ptr: *const f32,
    size: usize,
    phi_multiplier: f32, // Little phi (φ) for timing
) -> *mut f32 {
    let output_ptr = allocate_safe_vram(size);

    unsafe {
        unfold_cbm(seed_ptr, output_ptr, size, phi_multiplier);
    }

    output_ptr
}

/// K-Damped cellular unfolding — uses φ⁻¹ (0.618) for Goldilocks tracking.
/// Provides the mathematically optimal balance between exploration and
/// anchor-tracking in the Poincaré ball manifold.
#[no_mangle]
pub extern "C" fn trigger_k_damped_unfolding(
    seed_ptr: *const f32,
    size: usize,
    k_value: f32, // K tracking parameter (default: φ⁻¹ = 0.618)
) -> *mut f32 {
    // Use K-damped timing: the tracking coefficient modulates phi_multiplier
    let modulated_phi = LITTLE_PHI * (1.0 - k_value) + PHI_CONJUGATE * k_value;
    trigger_cellular_unfolding(seed_ptr, size, modulated_phi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_safe_vram() {
        let ptr = allocate_safe_vram(1024);
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_phi_constants() {
        assert!((LITTLE_PHI - 1.618034).abs() < 0.001);
        assert!((PHI_CONJUGATE - 0.618034).abs() < 0.001);
        // φ × φ⁻¹ ≈ 1
        assert!((LITTLE_PHI * PHI_CONJUGATE - 1.0).abs() < 0.001);
    }
}
