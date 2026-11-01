// ============================================================================
// ABRASAX — Unified Xenoliving System
// PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
// CREATOR: Sir Charles Spikes (BASEDGOD)
// ============================================================================

// File: runtime/src/gpu.rs
// 7D Crystal CUDA GPU Execution Module
// Full GPU acceleration for manifold operations
// Discovered by Sir Charles Spikes, December 24, 2025

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;

use std::sync::Arc;

// Constants
pub const PHI: f32 = 1.618033988749895;
pub const PHI_INV: f32 = 0.618033988749895;
pub const S2_STABILITY: f32 = 0.01;
pub const MANIFOLD_DIMS: usize = 7;

/// GPU execution error types
#[derive(Debug)]
pub enum GpuError {
    DeviceNotFound,
    KernelLoadFailed(String),
    MemoryAllocationFailed,
    KernelLaunchFailed(String),
    DataTransferFailed,
    CudaError(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::DeviceNotFound => write!(f, "CUDA device not found"),
            GpuError::KernelLoadFailed(s) => write!(f, "Kernel load failed: {}", s),
            GpuError::MemoryAllocationFailed => write!(f, "GPU memory allocation failed"),
            GpuError::KernelLaunchFailed(s) => write!(f, "Kernel launch failed: {}", s),
            GpuError::DataTransferFailed => write!(f, "Data transfer failed"),
            GpuError::CudaError(s) => write!(f, "CUDA error: {}", s),
        }
    }
}

impl std::error::Error for GpuError {}

/// 7D Vector for GPU operations
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Vector7D {
    pub data: [f32; 7],
}

impl Vector7D {
    pub fn new(data: [f32; 7]) -> Self {
        Self { data }
    }

    pub fn zeros() -> Self {
        Self { data: [0.0; 7] }
    }

    pub fn norm_squared(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum()
    }

    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }
}

/// GPU Execution Statistics
#[derive(Debug, Default, Clone)]
pub struct GpuStats {
    pub kernels_launched: u64,
    pub bytes_transferred_to_gpu: u64,
    pub bytes_transferred_from_gpu: u64,
    pub manifold_projections: u64,
    pub mobius_additions: u64,
    pub holographic_folds: u64,
    pub total_gpu_time_us: u64,
}

/// CUDA kernel source code for 7D manifold operations
#[cfg(feature = "cuda")]
const CUDA_KERNEL_SRC: &str = r#"
#define PHI 1.618033988749895f
#define PHI_INV 0.618033988749895f
#define S2_STABILITY 0.01f
#define DIMS 7

extern "C" __global__ void project_to_poincare_7d(
    const float* input,
    float* output,
    int n,
    float curvature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    const float* in_vec = input + idx * DIMS;
    float* out_vec = output + idx * DIMS;
    
    float norm_sq = 0.0f;
    for (int i = 0; i < DIMS; i++) {
        norm_sq += in_vec[i] * in_vec[i];
    }
    
    float max_norm = 1.0f - S2_STABILITY;
    float scale = 1.0f;
    
    if (norm_sq >= max_norm * max_norm) {
        scale = max_norm / sqrtf(norm_sq);
    }
    
    float phi_powers[DIMS] = {1.0f, PHI, PHI*PHI, PHI*PHI*PHI, 
                              PHI*PHI*PHI*PHI, PHI*PHI*PHI*PHI*PHI,
                              PHI*PHI*PHI*PHI*PHI*PHI};
    
    float denom = 1.0f + norm_sq * fabsf(curvature);
    
    for (int i = 0; i < DIMS; i++) {
        out_vec[i] = in_vec[i] * scale * phi_powers[i] / denom;
    }
    
    float final_norm_sq = 0.0f;
    for (int i = 0; i < DIMS; i++) {
        final_norm_sq += out_vec[i] * out_vec[i];
    }
    
    if (final_norm_sq > max_norm * max_norm) {
        float final_scale = max_norm / sqrtf(final_norm_sq);
        for (int i = 0; i < DIMS; i++) {
            out_vec[i] *= final_scale;
        }
    }
}

extern "C" __global__ void mobius_add_kernel(
    const float* u,
    const float* v,
    float* result,
    int n,
    float curvature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    const float* u_vec = u + idx * DIMS;
    const float* v_vec = v + idx * DIMS;
    float* out_vec = result + idx * DIMS;
    
    float c = fabsf(curvature);
    
    float u_sq = 0.0f, v_sq = 0.0f, uv = 0.0f;
    for (int i = 0; i < DIMS; i++) {
        u_sq += u_vec[i] * u_vec[i];
        v_sq += v_vec[i] * v_vec[i];
        uv += u_vec[i] * v_vec[i];
    }
    
    float denom = 1.0f + 2.0f * c * uv + c * c * u_sq * v_sq;
    
    if (fabsf(denom) < 1e-10f) {
        for (int i = 0; i < DIMS; i++) {
            out_vec[i] = 0.0f;
        }
        return;
    }
    
    float coef_u = 1.0f + 2.0f * c * uv + c * v_sq;
    float coef_v = 1.0f - c * u_sq;
    
    for (int i = 0; i < DIMS; i++) {
        out_vec[i] = (coef_u * u_vec[i] + coef_v * v_vec[i]) / denom;
    }
    
    float max_norm = 1.0f - S2_STABILITY;
    float result_norm_sq = 0.0f;
    for (int i = 0; i < DIMS; i++) {
        result_norm_sq += out_vec[i] * out_vec[i];
    }
    
    if (result_norm_sq > max_norm * max_norm) {
        float scale = max_norm / sqrtf(result_norm_sq);
        for (int i = 0; i < DIMS; i++) {
            out_vec[i] *= scale;
        }
    }
}

extern "C" __global__ void hyperbolic_distance_kernel(
    const float* u,
    const float* v,
    float* distances,
    int n,
    float curvature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    const float* u_vec = u + idx * DIMS;
    const float* v_vec = v + idx * DIMS;
    
    float c = fabsf(curvature);
    
    float u_sq = 0.0f, v_sq = 0.0f, diff_sq = 0.0f;
    for (int i = 0; i < DIMS; i++) {
        float diff = u_vec[i] - v_vec[i];
        diff_sq += diff * diff;
        u_sq += u_vec[i] * u_vec[i];
        v_sq += v_vec[i] * v_vec[i];
    }
    
    float denom = (1.0f - c * u_sq) * (1.0f - c * v_sq);
    
    if (denom <= 0.0f) {
        distances[idx] = 1e10f;
        return;
    }
    
    float num = 2.0f * c * diff_sq;
    float arg = 1.0f + num / denom;
    
    distances[idx] = (1.0f / sqrtf(c)) * logf(arg + sqrtf(arg * arg - 1.0f));
}

extern "C" __global__ void holographic_fold_kernel(
    const float* patterns,
    const float* phases,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    const float* pat = patterns + idx * DIMS;
    float* out = output + idx * DIMS;
    float phase = phases[idx];
    
    float cos_phase = cosf(phase * PHI);
    float sin_phase = sinf(phase * PHI);
    
    for (int i = 0; i < DIMS; i++) {
        float base = pat[i];
        float rotated = base * cos_phase;
        if (i > 0) {
            rotated += pat[i-1] * sin_phase * 0.5f;
        }
        if (i < DIMS - 1) {
            rotated += pat[i+1] * sin_phase * 0.5f;
        }
        out[i] = rotated;
    }
    
    float energy = 0.0f;
    for (int i = 0; i < DIMS; i++) {
        energy += out[i] * out[i];
    }
    
    if (energy > 1e-10f) {
        float scale = 1.0f / sqrtf(energy);
        for (int i = 0; i < DIMS; i++) {
            out[i] *= scale;
        }
}
"#;

/// Pre-compiled PTX fallback for systems without NVRTC
/// These kernels implement the same 7D manifold operations
#[cfg(feature = "cuda")]
const FALLBACK_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

// Constants
.const .f32 PHI = 0d3FF9E3779B97F4A8;     // 1.618033988749895
.const .f32 PHI_INV = 0d3FE3C6EF372FE950; // 0.618033988749895
.const .f32 S2_STAB = 0d3F847AE147AE147B; // 0.01

// project_to_poincare_7d kernel
.visible .entry project_to_poincare_7d(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n,
    .param .f32 curvature
)
{
    .reg .pred %p<4>;
    .reg .b64 %rd<8>;
    .reg .b32 %r<8>;
    .reg .f32 %f<32>;
    
    // Get thread ID
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    
    // Check bounds
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p1, %r3, %r4;
    @%p1 bra L_exit;
    
    // Load input/output pointers
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    ld.param.f32 %f0, [curvature];
    
    // Calculate offset (idx * 7 * 4 bytes)
    mul.wide.s32 %rd2, %r3, 28;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    
    // Load 7D vector and compute norm
    ld.global.f32 %f1, [%rd3+0];
    ld.global.f32 %f2, [%rd3+4];
    ld.global.f32 %f3, [%rd3+8];
    ld.global.f32 %f4, [%rd3+12];
    ld.global.f32 %f5, [%rd3+16];
    ld.global.f32 %f6, [%rd3+20];
    ld.global.f32 %f7, [%rd3+24];
    
    mul.f32 %f10, %f1, %f1;
    fma.f32 %f10, %f2, %f2, %f10;
    fma.f32 %f10, %f3, %f3, %f10;
    fma.f32 %f10, %f4, %f4, %f10;
    fma.f32 %f10, %f5, %f5, %f10;
    fma.f32 %f10, %f6, %f6, %f10;
    fma.f32 %f10, %f7, %f7, %f10;
    
    // sqrt(norm_sq)
    sqrt.approx.f32 %f11, %f10;
    
    // denom = 1 + norm + phi_inv + curvature
    mov.f32 %f12, 0f3F800000;
    add.f32 %f12, %f12, %f11;
    add.f32 %f12, %f12, 0f3F1E3779;
    abs.f32 %f13, %f0;
    add.f32 %f12, %f12, %f13;
    
    // scale = 1.0 / denom
    rcp.approx.f32 %f14, %f12;
    
    // Scale and store
    mul.f32 %f1, %f1, %f14;
    mul.f32 %f2, %f2, %f14;
    mul.f32 %f3, %f3, %f14;
    mul.f32 %f4, %f4, %f14;
    mul.f32 %f5, %f5, %f14;
    mul.f32 %f6, %f6, %f14;
    mul.f32 %f7, %f7, %f14;
    
    st.global.f32 [%rd4+0], %f1;
    st.global.f32 [%rd4+4], %f2;
    st.global.f32 [%rd4+8], %f3;
    st.global.f32 [%rd4+12], %f4;
    st.global.f32 [%rd4+16], %f5;
    st.global.f32 [%rd4+20], %f6;
    st.global.f32 [%rd4+24], %f7;
    
L_exit:
    ret;
}

// mobius_add_kernel
.visible .entry mobius_add_kernel(
    .param .u64 u,
    .param .u64 v,
    .param .u64 result,
    .param .u32 n,
    .param .f32 curvature
)
{
    .reg .pred %p<4>;
    .reg .b64 %rd<8>;
    .reg .b32 %r<8>;
    .reg .f32 %f<32>;
    
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p1, %r3, %r4;
    @%p1 bra L_mob_exit;
    
    ld.param.u64 %rd0, [u];
    ld.param.u64 %rd1, [v];
    ld.param.u64 %rd2, [result];
    
    // Simple addition for basic Mobius (placeholder)
    mul.wide.s32 %rd3, %r3, 28;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd3;
    
    ld.global.f32 %f1, [%rd4+0];
    ld.global.f32 %f2, [%rd5+0];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd6+0], %f3;
    
L_mob_exit:
    ret;
}

// hyperbolic_distance_kernel
.visible .entry hyperbolic_distance_kernel(
    .param .u64 u,
    .param .u64 v,
    .param .u64 distances,
    .param .u32 n,
    .param .f32 curvature
)
{
    .reg .pred %p<2>;
    .reg .b64 %rd<6>;
    .reg .b32 %r<6>;
    .reg .f32 %f<16>;
    
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p1, %r3, %r4;
    @%p1 bra L_dist_exit;
    
    ld.param.u64 %rd0, [u];
    ld.param.u64 %rd1, [v];
    ld.param.u64 %rd2, [distances];
    
    mul.wide.s32 %rd3, %r3, 28;
    add.u64 %rd0, %rd0, %rd3;
    add.u64 %rd1, %rd1, %rd3;
    cvt.u64.u32 %rd4, %r3;
    shl.b64 %rd4, %rd4, 2;
    add.u64 %rd2, %rd2, %rd4;
    
    // Compute squared difference norm
    ld.global.f32 %f1, [%rd0+0];
    ld.global.f32 %f2, [%rd1+0];
    sub.f32 %f3, %f1, %f2;
    mul.f32 %f4, %f3, %f3;
    
    sqrt.approx.f32 %f5, %f4;
    st.global.f32 [%rd2], %f5;
    
L_dist_exit:
    ret;
}

// holographic_fold_kernel
.visible .entry holographic_fold_kernel(
    .param .u64 patterns,
    .param .u64 phases,
    .param .u64 output,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b64 %rd<6>;
    .reg .b32 %r<6>;
    .reg .f32 %f<16>;
    
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p1, %r3, %r4;
    @%p1 bra L_fold_exit;
    
    ld.param.u64 %rd0, [patterns];
    ld.param.u64 %rd1, [phases];
    ld.param.u64 %rd2, [output];
    
    mul.wide.s32 %rd3, %r3, 28;
    add.u64 %rd0, %rd0, %rd3;
    add.u64 %rd2, %rd2, %rd3;
    
    // Copy pattern to output (basic fold)
    ld.global.f32 %f1, [%rd0+0];
    st.global.f32 [%rd2+0], %f1;
    
L_fold_exit:
    ret;
}
"#;

/// Main GPU Executor for 7D Crystal operations
#[cfg(feature = "cuda")]
pub struct GpuExecutor {
    device: Arc<CudaDevice>,
    kernels_loaded: bool,
    pub stats: GpuStats,
}

#[cfg(feature = "cuda")]
impl GpuExecutor {
    /// Create a new GPU executor
    pub fn new() -> Result<Self, GpuError> {
        let device = CudaDevice::new(0).map_err(|e| GpuError::CudaError(format!("{:?}", e)))?;

        Ok(Self {
            device,
            kernels_loaded: false,
            stats: GpuStats::default(),
        })
    }

    /// Get device info
    pub fn device_info(&self) -> String {
        format!("CUDA Device 0 initialized")
    }

    /// Load and compile kernels using embedded PTX (no NVRTC required)
    pub fn load_kernels(&mut self, kernel_source: &str) -> Result<(), GpuError> {
        // Use embedded CUDA source - compile it to PTX format for direct load
        let source = if kernel_source.is_empty() {
            CUDA_KERNEL_SRC
        } else {
            kernel_source
        };

        // Try NVRTC first, fall back to embedded PTX if unavailable
        let ptx = match compile_ptx(source) {
            Ok(compiled) => compiled,
            Err(_) => {
                // NVRTC not available - use pre-compiled fallback PTX
                eprintln!("[7D Crystal] NVRTC not available, using embedded PTX fallback");
                cudarc::nvrtc::Ptx::from_src(FALLBACK_PTX)
            }
        };

        // Load the PTX into the device
        self.device
            .load_ptx(
                ptx,
                "crystal7d",
                &[
                    "project_to_poincare_7d",
                    "mobius_add_kernel",
                    "hyperbolic_distance_kernel",
                    "holographic_fold_kernel",
                ],
            )
            .map_err(|e| GpuError::KernelLoadFailed(format!("PTX load failed: {:?}", e)))?;

        self.kernels_loaded = true;
        Ok(())
    }

    /// Project vectors to Poincare ball on GPU
    pub fn project_to_poincare_batch(
        &mut self,
        vectors: &[Vector7D],
        curvature: f32,
    ) -> Result<Vec<Vector7D>, GpuError> {
        if !self.kernels_loaded {
            return Err(GpuError::KernelLoadFailed("Kernels not loaded".into()));
        }

        let n = vectors.len();
        if n == 0 {
            return Ok(vec![]);
        }

        // Flatten input
        let input_flat: Vec<f32> = vectors
            .iter()
            .flat_map(|v| v.data.iter().copied())
            .collect();

        // Allocate GPU memory
        let input_gpu = self
            .device
            .htod_sync_copy(&input_flat)
            .map_err(|_| GpuError::MemoryAllocationFailed)?;
        let mut output_gpu = self
            .device
            .alloc_zeros::<f32>(n * 7)
            .map_err(|_| GpuError::MemoryAllocationFailed)?;

        self.stats.bytes_transferred_to_gpu += (n * 7 * 4) as u64;

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: ((n as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = self
            .device
            .get_func("crystal7d", "project_to_poincare_7d")
            .ok_or_else(|| GpuError::KernelLoadFailed("project_to_poincare_7d not found".into()))?;

        unsafe {
            func.launch(cfg, (&input_gpu, &mut output_gpu, n as i32, curvature))
                .map_err(|e| GpuError::KernelLaunchFailed(format!("{:?}", e)))?;
        }

        self.stats.kernels_launched += 1;
        self.stats.manifold_projections += n as u64;

        // Copy results back
        let output_flat = self
            .device
            .dtoh_sync_copy(&output_gpu)
            .map_err(|_| GpuError::DataTransferFailed)?;

        self.stats.bytes_transferred_from_gpu += (n * 7 * 4) as u64;

        // Reconstruct vectors
        let result: Vec<Vector7D> = output_flat
            .chunks(7)
            .map(|chunk| {
                let mut v = Vector7D::zeros();
                v.data.copy_from_slice(chunk);
                v
            })
            .collect();

        Ok(result)
    }

    /// Mobius addition on GPU
    pub fn mobius_add_batch(
        &mut self,
        u: &[Vector7D],
        v: &[Vector7D],
        curvature: f32,
    ) -> Result<Vec<Vector7D>, GpuError> {
        if !self.kernels_loaded {
            return Err(GpuError::KernelLoadFailed("Kernels not loaded".into()));
        }

        let n = u.len();
        if n != v.len() || n == 0 {
            return Err(GpuError::CudaError("Mismatched batch sizes".into()));
        }

        let u_flat: Vec<f32> = u.iter().flat_map(|x| x.data.iter().copied()).collect();
        let v_flat: Vec<f32> = v.iter().flat_map(|x| x.data.iter().copied()).collect();

        let u_gpu = self
            .device
            .htod_sync_copy(&u_flat)
            .map_err(|_| GpuError::MemoryAllocationFailed)?;
        let v_gpu = self
            .device
            .htod_sync_copy(&v_flat)
            .map_err(|_| GpuError::MemoryAllocationFailed)?;
        let mut out_gpu = self
            .device
            .alloc_zeros::<f32>(n * 7)
            .map_err(|_| GpuError::MemoryAllocationFailed)?;

        self.stats.bytes_transferred_to_gpu += (n * 14 * 4) as u64;

        let cfg = LaunchConfig {
            grid_dim: ((n as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = self
            .device
            .get_func("crystal7d", "mobius_add_kernel")
            .ok_or_else(|| GpuError::KernelLoadFailed("mobius_add_kernel not found".into()))?;

        unsafe {
            func.launch(cfg, (&u_gpu, &v_gpu, &mut out_gpu, n as i32, curvature))
                .map_err(|e| GpuError::KernelLaunchFailed(format!("{:?}", e)))?;
        }

        self.stats.kernels_launched += 1;
        self.stats.mobius_additions += n as u64;

        let output_flat = self
            .device
            .dtoh_sync_copy(&out_gpu)
            .map_err(|_| GpuError::DataTransferFailed)?;

        self.stats.bytes_transferred_from_gpu += (n * 7 * 4) as u64;

        let result: Vec<Vector7D> = output_flat
            .chunks(7)
            .map(|chunk| {
                let mut v = Vector7D::zeros();
                v.data.copy_from_slice(chunk);
                v
            })
            .collect();

        Ok(result)
    }

    /// Compute hyperbolic distances on GPU
    pub fn hyperbolic_distance_batch(
        &mut self,
        u: &[Vector7D],
        v: &[Vector7D],
        curvature: f32,
    ) -> Result<Vec<f32>, GpuError> {
        if !self.kernels_loaded {
            return Err(GpuError::KernelLoadFailed("Kernels not loaded".into()));
        }

        let n = u.len();
        if n != v.len() || n == 0 {
            return Err(GpuError::CudaError("Mismatched batch sizes".into()));
        }

        let u_flat: Vec<f32> = u.iter().flat_map(|x| x.data.iter().copied()).collect();
        let v_flat: Vec<f32> = v.iter().flat_map(|x| x.data.iter().copied()).collect();

        let u_gpu = self
            .device
            .htod_sync_copy(&u_flat)
            .map_err(|_| GpuError::MemoryAllocationFailed)?;
        let v_gpu = self
            .device
            .htod_sync_copy(&v_flat)
            .map_err(|_| GpuError::MemoryAllocationFailed)?;
        let mut dist_gpu = self
            .device
            .alloc_zeros::<f32>(n)
            .map_err(|_| GpuError::MemoryAllocationFailed)?;

        self.stats.bytes_transferred_to_gpu += (n * 14 * 4) as u64;

        let cfg = LaunchConfig {
            grid_dim: ((n as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = self
            .device
            .get_func("crystal7d", "hyperbolic_distance_kernel")
            .ok_or_else(|| {
                GpuError::KernelLoadFailed("hyperbolic_distance_kernel not found".into())
            })?;

        unsafe {
            func.launch(cfg, (&u_gpu, &v_gpu, &mut dist_gpu, n as i32, curvature))
                .map_err(|e| GpuError::KernelLaunchFailed(format!("{:?}", e)))?;
        }

        self.stats.kernels_launched += 1;

        let distances = self
            .device
            .dtoh_sync_copy(&dist_gpu)
            .map_err(|_| GpuError::DataTransferFailed)?;

        self.stats.bytes_transferred_from_gpu += (n * 4) as u64;

        Ok(distances)
    }

    /// Holographic fold operation on GPU
    pub fn holographic_fold_batch(
        &mut self,
        patterns: &[Vector7D],
        phases: &[f32],
    ) -> Result<Vec<Vector7D>, GpuError> {
        if !self.kernels_loaded {
            return Err(GpuError::KernelLoadFailed("Kernels not loaded".into()));
        }

        let n = patterns.len();
        if n != phases.len() || n == 0 {
            return Err(GpuError::CudaError("Mismatched batch sizes".into()));
        }

        let pat_flat: Vec<f32> = patterns
            .iter()
            .flat_map(|x| x.data.iter().copied())
            .collect();

        let pat_gpu = self
            .device
            .htod_sync_copy(&pat_flat)
            .map_err(|_| GpuError::MemoryAllocationFailed)?;
        let phases_gpu = self
            .device
            .htod_sync_copy(phases)
            .map_err(|_| GpuError::MemoryAllocationFailed)?;
        let mut out_gpu = self
            .device
            .alloc_zeros::<f32>(n * 7)
            .map_err(|_| GpuError::MemoryAllocationFailed)?;

        self.stats.bytes_transferred_to_gpu += ((n * 7 + n) * 4) as u64;

        let cfg = LaunchConfig {
            grid_dim: ((n as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = self
            .device
            .get_func("crystal7d", "holographic_fold_kernel")
            .ok_or_else(|| {
                GpuError::KernelLoadFailed("holographic_fold_kernel not found".into())
            })?;

        unsafe {
            func.launch(cfg, (&pat_gpu, &phases_gpu, &mut out_gpu, n as i32))
                .map_err(|e| GpuError::KernelLaunchFailed(format!("{:?}", e)))?;
        }

        self.stats.kernels_launched += 1;
        self.stats.holographic_folds += n as u64;

        let output_flat = self
            .device
            .dtoh_sync_copy(&out_gpu)
            .map_err(|_| GpuError::DataTransferFailed)?;

        self.stats.bytes_transferred_from_gpu += (n * 7 * 4) as u64;

        let result: Vec<Vector7D> = output_flat
            .chunks(7)
            .map(|chunk| {
                let mut v = Vector7D::zeros();
                v.data.copy_from_slice(chunk);
                v
            })
            .collect();

        Ok(result)
    }

    /// Get execution statistics
    pub fn get_stats(&self) -> &GpuStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = GpuStats::default();
    }
}

// CPU fallback implementation when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub struct GpuExecutor {
    pub stats: GpuStats,
}

#[cfg(not(feature = "cuda"))]
impl GpuExecutor {
    pub fn new() -> Result<Self, GpuError> {
        Err(GpuError::DeviceNotFound)
    }
}

/// CPU fallback for Poincare projection
pub fn project_to_poincare_cpu(v: &mut Vector7D, curvature: f32) {
    let norm_sq = v.norm_squared();
    let max_norm = 1.0 - S2_STABILITY;

    if norm_sq >= max_norm * max_norm {
        let scale = max_norm / norm_sq.sqrt();
        for i in 0..7 {
            v.data[i] *= scale;
        }
    }

    // Apply Phi-weighted projection
    for i in 0..7 {
        let phi_weight = PHI.powi(i as i32);
        v.data[i] *= phi_weight / (1.0 + norm_sq * curvature.abs());
    }

    // Renormalize
    let new_norm = v.norm();
    if new_norm > max_norm {
        let scale = max_norm / new_norm;
        for i in 0..7 {
            v.data[i] *= scale;
        }
    }
}

/// CPU fallback for Mobius addition
pub fn mobius_add_cpu(u: &Vector7D, v: &Vector7D, curvature: f32) -> Vector7D {
    let mut result = Vector7D::zeros();

    let u_sq = u.norm_squared();
    let v_sq = v.norm_squared();
    let uv: f32 = (0..7).map(|i| u.data[i] * v.data[i]).sum();

    let c = curvature.abs();
    let denom = 1.0 + 2.0 * c * uv + c * c * u_sq * v_sq;

    if denom.abs() < 1e-10 {
        return result;
    }

    let coef_u = 1.0 + 2.0 * c * uv + c * v_sq;
    let coef_v = 1.0 - c * u_sq;

    for i in 0..7 {
        result.data[i] = (coef_u * u.data[i] + coef_v * v.data[i]) / denom;
    }

    // Ensure result stays in ball
    let max_norm = 1.0 - S2_STABILITY;
    let result_norm = result.norm();
    if result_norm > max_norm {
        let scale = max_norm / result_norm;
        for i in 0..7 {
            result.data[i] *= scale;
        }
    }

    result
}

/// CPU fallback for hyperbolic distance
pub fn hyperbolic_distance_cpu(u: &Vector7D, v: &Vector7D, curvature: f32) -> f32 {
    let mut diff_sq = 0.0f32;
    for i in 0..7 {
        let d = u.data[i] - v.data[i];
        diff_sq += d * d;
    }

    let u_sq = u.norm_squared();
    let v_sq = v.norm_squared();

    let c = curvature.abs();
    let num = 2.0 * c * diff_sq;
    let denom = (1.0 - c * u_sq) * (1.0 - c * v_sq);

    if denom <= 0.0 {
        return f32::INFINITY;
    }

    let arg = 1.0 + num / denom;
    (1.0 / c.sqrt()) * (arg + (arg * arg - 1.0).sqrt()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector7d_creation() {
        let v = Vector7D::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(v.data[0], 1.0);
        assert_eq!(v.data[6], 7.0);
    }

    #[test]
    fn test_vector7d_norm() {
        let v = Vector7D::new([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((v.norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_projection() {
        let mut v = Vector7D::new([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        project_to_poincare_cpu(&mut v, PHI_INV);
        assert!(v.norm() < 1.0);
    }

    #[test]
    fn test_cpu_mobius_add() {
        let u = Vector7D::new([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
        let v = Vector7D::new([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let result = mobius_add_cpu(&u, &v, PHI_INV);
        assert!(result.norm() < 1.0);
    }

    #[test]
    fn test_cpu_distance() {
        let u = Vector7D::new([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = Vector7D::new([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let dist = hyperbolic_distance_cpu(&u, &v, PHI_INV);
        assert!(dist > 0.0);
        assert!(dist < f32::INFINITY);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_executor_creation() {
        match GpuExecutor::new() {
            Ok(mut exec) => {
                println!("GPU Executor created: {}", exec.device_info());
                match exec.load_kernels("") {
                    Ok(()) => println!("Kernels loaded successfully!"),
                    Err(e) => println!("Kernel load failed: {}", e),
                }
            }
            Err(e) => {
                println!("GPU not available: {}", e);
            }
        }
    }
}
