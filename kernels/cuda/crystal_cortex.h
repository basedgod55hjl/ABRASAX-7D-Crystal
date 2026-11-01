#ifndef CRYSTAL_CORTEX_H
#define CRYSTAL_CORTEX_H

/*
 * crystal_cortex.h -- Sacred Constants & Kernel Declarations
 * 7D mH-Q Crystal Architecture | GTX 1660 Ti (sm_75)
 *
 * This header is included by the translated .cu files.
 * It provides all PHI-based constants and declares the
 * exported __global__ kernel entry points.
 */

#include <cuda_runtime.h>
#include <math.h>

/* ===================================================================
 *  SACRED CONSTANTS
 * =================================================================== */

#define PHI                1.618033988749895f   /* Golden Ratio               */
#define PHI_INV            0.618033988749895f   /* Phi inverse                */
#define PHI_SQUARED        2.618033988749895f   /* Phi squared                */
#define SQRT_PHI           1.272019649514069f   /* sqrt(Phi)                 */
#define PI                 3.141592653589793f   /* Pi                        */
#define TWO_PI             6.283185307179586f   /* 2 * Pi                    */
#define E                  2.718281828459045f   /* Euler's number            */
#define STABILITY_EPSILON  0.01f                /* S2 stability offset       */

#define MANIFOLD_DIMS      7                    /* 7D manifold dimensions    */
#define BLOCK_SIZE_1D      256                  /* Default 1D block size     */
#define BLOCK_SIZE_2D_X    16                   /* 2D block X dim            */
#define BLOCK_SIZE_2D_Y    16                   /* 2D block Y dim            */

/* ===================================================================
 *  DEVICE UTILITY FUNCTIONS (defined in crystal_kernels_v2.cu)
 * =================================================================== */

#ifdef __CUDACC__

__forceinline__ __device__ float sacred_sigmoid(float x);
__forceinline__ __device__ float phi_tanh(float x);
__forceinline__ __device__ float fast_exp(float x);

#endif /* __CUDACC__ */

/* ===================================================================
 *  KERNEL DECLARATIONS (v2 -- crystal.7d)
 * =================================================================== */

extern "C" {

__global__ void manifold_projection_7d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n, int dim
);

__global__ void manifold_projection_batch_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int dim
);

__global__ void phi_harmonic_modulation_kernel(
    float* __restrict__ data,
    int n, float phase, int harmonic_order
);

__global__ void fibonacci_spiral_embedding_kernel(
    const float* __restrict__ input,
    float* __restrict__ output_x,
    float* __restrict__ output_y,
    int n
);

__global__ void cbm_flux_unfold_v2_kernel(
    const float* __restrict__ seed,
    float* __restrict__ output,
    int seed_size, int output_size,
    float phi_flux, float time_factor
);

__global__ void cbm_flux_fold_kernel(
    const float* __restrict__ weights,
    float* __restrict__ seed,
    int seed_size, int weight_size
);

__global__ void holographic_interference_kernel(
    const float* __restrict__ reference,
    const float* __restrict__ object,
    float* __restrict__ hologram,
    int n
);

__global__ void holographic_reconstruction_kernel(
    const float* __restrict__ hologram,
    const float* __restrict__ reference,
    float* __restrict__ reconstructed,
    int n
);

__global__ void quantum_evolution_2d_kernel(
    const float* __restrict__ field_in,
    float* __restrict__ field_out,
    int width, int height,
    float phase, float dt
);

__global__ void crystal_resonance_kernel(
    const float* __restrict__ pattern,
    float* __restrict__ phi_alignment,
    float* __restrict__ coherence,
    int n
);

__global__ void crystal_entropy_mining_kernel(
    float* __restrict__ entropy,
    int n,
    unsigned long long base_seed,
    float time_factor
);

__global__ void flower_of_life_kernel(
    float* __restrict__ pattern,
    int width, int height, float scale
);

__global__ void metatron_cube_kernel(
    float* __restrict__ pattern,
    int width, int height, float rotation
);

} /* extern "C" */

/* ===================================================================
 *  KERNEL DECLARATIONS (v1 -- crystal_cortex.7d)
 * =================================================================== */

extern "C" {

__global__ void mhq7d_attention(
    const float* Q, const float* K, const float* V,
    float* O, int heads, int dim
);

__global__ void mhq7d_persistent(float* manifold, int size);

__global__ void manifold_projection_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n, int dim
);

__global__ void phi_modulation_kernel(
    float* __restrict__ data,
    int n, float phase
);

__global__ void cbm_flux_unfold_kernel(
    const float* __restrict__ seed,
    float* __restrict__ weights,
    int dna_size, int matrix_size,
    float phi_flux
);

} /* extern "C" */

#endif /* CRYSTAL_CORTEX_H */
