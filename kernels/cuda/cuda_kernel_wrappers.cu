/**
 * cuda_kernel_wrappers.cu -- Host-callable wrappers for 7D Crystal CUDA Kernels
 * ============================================================================
 *
 * Provides extern "C" functions that wrap each __global__ kernel with
 * device memory management (cudaMalloc/cudaMemcpy/launch/cudaMemcpy/cudaFree).
 *
 * These are compiled by compile_o_to_dll.bat into crystal_kernels.dll,
 * which can then be loaded by ctypes.CDLL in cuda_kernel_loader.py.
 *
 * Compilation (standalone, via compile_o_to_dll.bat):
 *   nvcc -O3 --shared -arch=sm_75 -o build/crystal_kernels.dll ^
 *       cuda_kernel_wrappers.cu crystal_kernels_v1.cu crystal_kernels_v2.cu physx_crystal_dynamics.cu ^
 *       -Icuda
 *
 * © 2026 Sir Charles Spikes | Crystal Architecture
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

/* =========================================================================
 *  SACRED CONSTANTS (mirrored from crystal_cortex.h)
 * ========================================================================= */
#define PHI           1.618033988749895f
#define PHI_INV       0.618033988749895f
#define BLOCK_SIZE_1D 256
#define BLOCK_2D_X    16
#define BLOCK_2D_Y    16

/* =========================================================================
 *  KERNEL DECLARATIONS (extern -- device code from linked .o files)
 * ========================================================================= */

/* -- v1: crystal_kernels_v1.cu -- */
extern __global__ void mhq7d_attention(
    const float* Q, const float* K, const float* V,
    float* O, int heads, int dim
);
extern __global__ void mhq7d_persistent(float* manifold, int size);
extern __global__ void manifold_projection_kernel(
    const float* input, float* output, int n, int dim
);
extern __global__ void phi_modulation_kernel(
    float* data, int n, float phase
);
extern __global__ void cbm_flux_unfold_kernel(
    const float* seed, float* weights, int dna_size, int matrix_size, float phi_flux
);

/* -- v2: crystal_kernels_v2.cu -- */
extern __global__ void manifold_projection_7d_kernel(
    const float* input, float* output, int n, int dim
);
extern __global__ void manifold_projection_batch_kernel(
    const float* input, float* output, int batch_size, int dim
);
extern __global__ void phi_harmonic_modulation_kernel(
    float* data, int n, float phase, int harmonic_order
);
extern __global__ void fibonacci_spiral_embedding_kernel(
    const float* input, float* output_x, float* output_y, int n
);
extern __global__ void cbm_flux_unfold_v2_kernel(
    const float* seed, float* output, int seed_size, int output_size,
    float phi_flux, float time_factor
);
extern __global__ void cbm_flux_fold_kernel(
    const float* weights, float* seed, int seed_size, int weight_size
);
extern __global__ void holographic_interference_kernel(
    const float* reference, const float* object, float* hologram, int n
);
extern __global__ void holographic_reconstruction_kernel(
    const float* hologram, const float* reference, float* reconstructed, int n
);
extern __global__ void quantum_evolution_2d_kernel(
    const float* field_in, float* field_out, int width, int height,
    float phase, float dt
);
extern __global__ void crystal_resonance_kernel(
    const float* pattern, float* phi_alignment, float* coherence, int n
);
extern __global__ void crystal_entropy_mining_kernel(
    float* entropy, int n, unsigned long long base_seed, float time_factor
);
extern __global__ void flower_of_life_kernel(
    float* pattern, int width, int height, float scale
);
extern __global__ void metatron_cube_kernel(
    float* pattern, int width, int height, float rotation
);

/* -- physx crystal dynamics -- */
struct CrystalRigidBody {
    float position[7];
    float velocity[7];
    float mass;
    float resonance;
};
extern __global__ void physx_crystal_dynamics_kernel(
    struct CrystalRigidBody* bodies,
    const float* cosmos_predictions,
    int n_bodies,
    float dt
);

/*
 * HELPER: launch 1D kernel with grid/block computation
 * NOTE: Deprecated / placeholder — each wrapper calls its kernel via <<<>>> directly.
 */
#if 0
static int launch_1d(
    const void* func,
    dim3 grid, dim3 block,
    const void** params, size_t num_params,
    cudaStream_t stream
) {
    (void)func; (void)grid; (void)block; (void)params; (void)stream;
    return cudaErrorNotSupported;
}
#endif

/* =========================================================================
 *  V2 KERNEL WRAPPERS
 * ========================================================================= */

/**
 * cuda_manifold_projection_7d - Host wrapper for manifold_projection_7d_kernel.
 * Allocates device memory, copies input, launches kernel, copies output,
 * frees device memory.
 * Returns 0 on success, cudaError_t on failure.
 */
extern "C" int cuda_manifold_projection_7d(
    const float* h_input, float* h_output, int n, int dim
) {
    float *d_input = NULL, *d_output = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_input, (size_t)n * sizeof(float));
    if (err) return err;
    err = cudaMalloc(&d_output, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_input); return err; }

    err = cudaMemcpy(d_input, h_input, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_input); cudaFree(d_output); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (n + blockSize - 1) / blockSize;
        manifold_projection_7d_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n, dim);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_input); cudaFree(d_output); return err; }

    err = cudaMemcpy(h_output, d_output, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    return err;
}

/**
 * cuda_manifold_projection_batch - Host wrapper.
 */
extern "C" int cuda_manifold_projection_batch(
    const float* h_input, float* h_output, int batch_size, int dim
) {
    float *d_input = NULL, *d_output = NULL;
    cudaError_t err;
    int n = batch_size * dim;

    err = cudaMalloc(&d_input, (size_t)n * sizeof(float));
    if (err) return err;
    err = cudaMalloc(&d_output, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_input); return err; }

    err = cudaMemcpy(d_input, h_input, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_input); cudaFree(d_output); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (batch_size + blockSize - 1) / blockSize;
        manifold_projection_batch_kernel<<<numBlocks, blockSize>>>(d_input, d_output, batch_size, dim);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_input); cudaFree(d_output); return err; }

    err = cudaMemcpy(h_output, d_output, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    return err;
}

/**
 * cuda_phi_harmonic_modulation - Host wrapper (in-place kernel).
 */
extern "C" int cuda_phi_harmonic_modulation(
    float* h_data, int n, float phase, int harmonic_order
) {
    float *d_data = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_data, (size_t)n * sizeof(float));
    if (err) return err;

    err = cudaMemcpy(d_data, h_data, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_data); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (n + blockSize - 1) / blockSize;
        phi_harmonic_modulation_kernel<<<numBlocks, blockSize>>>(d_data, n, phase, harmonic_order);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_data); return err; }

    err = cudaMemcpy(h_data, d_data, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    return err;
}

/**
 * cuda_fibonacci_spiral_embedding - Host wrapper.
 */
extern "C" int cuda_fibonacci_spiral_embedding(
    const float* h_input, float* h_output_x, float* h_output_y, int n
) {
    float *d_input = NULL, *d_out_x = NULL, *d_out_y = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_input, (size_t)n * sizeof(float));
    if (err) return err;
    err = cudaMalloc(&d_out_x, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_input); return err; }
    err = cudaMalloc(&d_out_y, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_input); cudaFree(d_out_x); return err; }

    err = cudaMemcpy(d_input, h_input, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_input); cudaFree(d_out_x); cudaFree(d_out_y); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (n + blockSize - 1) / blockSize;
        fibonacci_spiral_embedding_kernel<<<numBlocks, blockSize>>>(d_input, d_out_x, d_out_y, n);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_input); cudaFree(d_out_x); cudaFree(d_out_y); return err; }

    err = cudaMemcpy(h_output_x, d_out_x, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err) { cudaFree(d_input); cudaFree(d_out_x); cudaFree(d_out_y); return err; }
    err = cudaMemcpy(h_output_y, d_out_y, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_out_x);
    cudaFree(d_out_y);
    return err;
}

/**
 * cuda_cbm_flux_unfold_v2 - Host wrapper.
 */
extern "C" int cuda_cbm_flux_unfold_v2(
    const float* h_seed, float* h_output, int seed_size, int output_size,
    float phi_flux, float time_factor
) {
    float *d_seed = NULL, *d_output = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_seed, (size_t)seed_size * sizeof(float));
    if (err) return err;
    err = cudaMalloc(&d_output, (size_t)output_size * sizeof(float));
    if (err) { cudaFree(d_seed); return err; }

    err = cudaMemcpy(d_seed, h_seed, (size_t)seed_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_seed); cudaFree(d_output); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (output_size + blockSize - 1) / blockSize;
        cbm_flux_unfold_v2_kernel<<<numBlocks, blockSize>>>(
            d_seed, d_output, seed_size, output_size, phi_flux, time_factor
        );
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_seed); cudaFree(d_output); return err; }

    err = cudaMemcpy(h_output, d_output, (size_t)output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_seed);
    cudaFree(d_output);
    return err;
}

/**
 * cuda_cbm_flux_fold - Host wrapper.
 */
extern "C" int cuda_cbm_flux_fold(
    const float* h_weights, float* h_seed, int seed_size, int weight_size
) {
    float *d_weights = NULL, *d_seed = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_weights, (size_t)weight_size * sizeof(float));
    if (err) return err;
    err = cudaMalloc(&d_seed, (size_t)seed_size * sizeof(float));
    if (err) { cudaFree(d_weights); return err; }

    err = cudaMemcpy(d_weights, h_weights, (size_t)weight_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_weights); cudaFree(d_seed); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (seed_size + blockSize - 1) / blockSize;
        cbm_flux_fold_kernel<<<numBlocks, blockSize>>>(d_weights, d_seed, seed_size, weight_size);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_weights); cudaFree(d_seed); return err; }

    err = cudaMemcpy(h_seed, d_seed, (size_t)seed_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_weights);
    cudaFree(d_seed);
    return err;
}

/**
 * cuda_holographic_interference - Host wrapper.
 */
extern "C" int cuda_holographic_interference(
    const float* h_reference, const float* h_object, float* h_hologram, int n
) {
    float *d_ref = NULL, *d_obj = NULL, *d_holo = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_ref, (size_t)n * sizeof(float));
    if (err) return err;
    err = cudaMalloc(&d_obj, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_ref); return err; }
    err = cudaMalloc(&d_holo, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_ref); cudaFree(d_obj); return err; }

    err = cudaMemcpy(d_ref, h_reference, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_ref); cudaFree(d_obj); cudaFree(d_holo); return err; }
    err = cudaMemcpy(d_obj, h_object, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_ref); cudaFree(d_obj); cudaFree(d_holo); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (n + blockSize - 1) / blockSize;
        holographic_interference_kernel<<<numBlocks, blockSize>>>(d_ref, d_obj, d_holo, n);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_ref); cudaFree(d_obj); cudaFree(d_holo); return err; }

    err = cudaMemcpy(h_hologram, d_holo, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ref);
    cudaFree(d_obj);
    cudaFree(d_holo);
    return err;
}

/**
 * cuda_holographic_reconstruction - Host wrapper.
 */
extern "C" int cuda_holographic_reconstruction(
    const float* h_hologram, const float* h_reference, float* h_reconstructed, int n
) {
    float *d_holo = NULL, *d_ref = NULL, *d_rec = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_holo, (size_t)n * sizeof(float));
    if (err) return err;
    err = cudaMalloc(&d_ref, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_holo); return err; }
    err = cudaMalloc(&d_rec, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_holo); cudaFree(d_ref); return err; }

    err = cudaMemcpy(d_holo, h_hologram, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_holo); cudaFree(d_ref); cudaFree(d_rec); return err; }
    err = cudaMemcpy(d_ref, h_reference, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_holo); cudaFree(d_ref); cudaFree(d_rec); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (n + blockSize - 1) / blockSize;
        holographic_reconstruction_kernel<<<numBlocks, blockSize>>>(d_holo, d_ref, d_rec, n);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_holo); cudaFree(d_ref); cudaFree(d_rec); return err; }

    err = cudaMemcpy(h_reconstructed, d_rec, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_holo);
    cudaFree(d_ref);
    cudaFree(d_rec);
    return err;
}

/**
 * cuda_quantum_evolution_2d - Host wrapper.
 */
extern "C" int cuda_quantum_evolution_2d(
    const float* h_field_in, float* h_field_out, int width, int height,
    float phase, float dt
) {
    float *d_in = NULL, *d_out = NULL;
    cudaError_t err;
    int n = width * height;

    err = cudaMalloc(&d_in, (size_t)n * sizeof(float));
    if (err) return err;
    err = cudaMalloc(&d_out, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_in); return err; }

    err = cudaMemcpy(d_in, h_field_in, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_in); cudaFree(d_out); return err; }

    {
        dim3 block(BLOCK_2D_X, BLOCK_2D_Y);
        dim3 grid((width + BLOCK_2D_X - 1) / BLOCK_2D_X,
                  (height + BLOCK_2D_Y - 1) / BLOCK_2D_Y);
        quantum_evolution_2d_kernel<<<grid, block>>>(d_in, d_out, width, height, phase, dt);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_in); cudaFree(d_out); return err; }

    err = cudaMemcpy(h_field_out, d_out, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return err;
}

/**
 * cuda_crystal_resonance - Host wrapper.
 * phi_alignment and coherence are output scalars (single float each).
 */
extern "C" int cuda_crystal_resonance(
    const float* h_pattern, float* h_phi_alignment, float* h_coherence, int n
) {
    float *d_pattern = NULL, *d_phi = NULL, *d_coh = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_pattern, (size_t)n * sizeof(float));
    if (err) return err;
    err = cudaMalloc(&d_phi, sizeof(float));
    if (err) { cudaFree(d_pattern); return err; }
    err = cudaMalloc(&d_coh, sizeof(float));
    if (err) { cudaFree(d_pattern); cudaFree(d_phi); return err; }

    err = cudaMemcpy(d_pattern, h_pattern, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_pattern); cudaFree(d_phi); cudaFree(d_coh); return err; }

    /* Initialize output scalars to zero */
    err = cudaMemset(d_phi, 0, sizeof(float));
    if (err) { cudaFree(d_pattern); cudaFree(d_phi); cudaFree(d_coh); return err; }
    err = cudaMemset(d_coh, 0, sizeof(float));
    if (err) { cudaFree(d_pattern); cudaFree(d_phi); cudaFree(d_coh); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (n + blockSize - 1) / blockSize;
        crystal_resonance_kernel<<<numBlocks, blockSize>>>(d_pattern, d_phi, d_coh, n);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_pattern); cudaFree(d_phi); cudaFree(d_coh); return err; }

    err = cudaMemcpy(h_phi_alignment, d_phi, sizeof(float), cudaMemcpyDeviceToHost);
    if (err) { cudaFree(d_pattern); cudaFree(d_phi); cudaFree(d_coh); return err; }
    err = cudaMemcpy(h_coherence, d_coh, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_pattern);
    cudaFree(d_phi);
    cudaFree(d_coh);
    return err;
}

/**
 * cuda_crystal_entropy_mining - Host wrapper.
 */
extern "C" int cuda_crystal_entropy_mining(
    float* h_entropy, int n, unsigned long long base_seed, float time_factor
) {
    float* d_entropy = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_entropy, (size_t)n * sizeof(float));
    if (err) return err;

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (n + blockSize - 1) / blockSize;
        crystal_entropy_mining_kernel<<<numBlocks, blockSize>>>(d_entropy, n, base_seed, time_factor);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_entropy); return err; }

    err = cudaMemcpy(h_entropy, d_entropy, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_entropy);
    return err;
}

/**
 * cuda_flower_of_life - Host wrapper (2D kernel).
 */
extern "C" int cuda_flower_of_life(
    float* h_pattern, int width, int height, float scale
) {
    float* d_pattern = NULL;
    cudaError_t err;
    int n = width * height;

    err = cudaMalloc(&d_pattern, (size_t)n * sizeof(float));
    if (err) return err;

    {
        dim3 block(BLOCK_2D_X, BLOCK_2D_Y);
        dim3 grid((width + BLOCK_2D_X - 1) / BLOCK_2D_X,
                  (height + BLOCK_2D_Y - 1) / BLOCK_2D_Y);
        flower_of_life_kernel<<<grid, block>>>(d_pattern, width, height, scale);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_pattern); return err; }

    err = cudaMemcpy(h_pattern, d_pattern, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_pattern);
    return err;
}

/**
 * cuda_metatron_cube - Host wrapper (2D kernel).
 */
extern "C" int cuda_metatron_cube(
    float* h_pattern, int width, int height, float rotation
) {
    float* d_pattern = NULL;
    cudaError_t err;
    int n = width * height;

    err = cudaMalloc(&d_pattern, (size_t)n * sizeof(float));
    if (err) return err;

    {
        dim3 block(BLOCK_2D_X, BLOCK_2D_Y);
        dim3 grid((width + BLOCK_2D_X - 1) / BLOCK_2D_X,
                  (height + BLOCK_2D_Y - 1) / BLOCK_2D_Y);
        metatron_cube_kernel<<<grid, block>>>(d_pattern, width, height, rotation);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_pattern); return err; }

    err = cudaMemcpy(h_pattern, d_pattern, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_pattern);
    return err;
}

/* =========================================================================
 *  V1 KERNEL WRAPPERS
 * ========================================================================= */

/**
 * cuda_mhq7d_attention - Host wrapper.
 */
extern "C" int cuda_mhq7d_attention(
    const float* h_Q, const float* h_K, const float* h_V,
    float* h_O, int heads, int dim
) {
    float *d_Q = NULL, *d_K = NULL, *d_V = NULL, *d_O = NULL;
    cudaError_t err;
    int n = heads * dim;

    err = cudaMalloc(&d_Q, (size_t)n * sizeof(float));
    if (err) return err;
    err = cudaMalloc(&d_K, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_Q); return err; }
    err = cudaMalloc(&d_V, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_Q); cudaFree(d_K); return err; }
    err = cudaMalloc(&d_O, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); return err; }

    err = cudaMemcpy(d_Q, h_Q, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); return err; }
    err = cudaMemcpy(d_K, h_K, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); return err; }
    err = cudaMemcpy(d_V, h_V, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (dim + blockSize - 1) / blockSize;
        mhq7d_attention<<<numBlocks, blockSize>>>(d_Q, d_K, d_V, d_O, heads, dim);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); return err; }

    err = cudaMemcpy(h_O, d_O, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    return err;
}

/**
 * cuda_mhq7d_persistent_start - Launch persistent kernel.
 * Returns stream handle so caller can signal termination.
 */
extern "C" int cuda_mhq7d_persistent_start(
    float** d_manifold_out, cudaStream_t* stream_out, int size
) {
    cudaError_t err;

    err = cudaMalloc(d_manifold_out, (size_t)size * sizeof(float));
    if (err) return err;

    err = cudaMemset(*d_manifold_out, 0, (size_t)size * sizeof(float));
    if (err) { cudaFree(*d_manifold_out); return err; }

    err = cudaStreamCreate(stream_out);
    if (err) { cudaFree(*d_manifold_out); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (size + blockSize - 1) / blockSize;
        mhq7d_persistent<<<numBlocks, blockSize, 0, *stream_out>>>(*d_manifold_out, size);
    }

    return cudaGetLastError();
}

/**
 * cuda_mhq7d_persistent_stop - Signal termination and clean up.
 */
extern "C" int cuda_mhq7d_persistent_stop(
    float* d_manifold, cudaStream_t stream, int size
) {
    cudaError_t err;
    float signal = 999.0f;

    /* Signal termination */
    err = cudaMemcpy(d_manifold, &signal, sizeof(float), cudaMemcpyHostToDevice);
    if (err) return err;

    /* Wait for kernel to finish */
    err = cudaStreamSynchronize(stream);
    if (err) return err;

    cudaFree(d_manifold);
    cudaStreamDestroy(stream);
    return cudaSuccess;
}

/**
 * cuda_manifold_projection_v1 - Host wrapper.
 */
extern "C" int cuda_manifold_projection_v1(
    const float* h_input, float* h_output, int n, int dim
) {
    float *d_input = NULL, *d_output = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_input, (size_t)n * sizeof(float));
    if (err) return err;
    err = cudaMalloc(&d_output, (size_t)n * sizeof(float));
    if (err) { cudaFree(d_input); return err; }

    err = cudaMemcpy(d_input, h_input, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_input); cudaFree(d_output); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (n + blockSize - 1) / blockSize;
        manifold_projection_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n, dim);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_input); cudaFree(d_output); return err; }

    err = cudaMemcpy(h_output, d_output, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    return err;
}

/**
 * cuda_phi_modulation - Host wrapper (in-place).
 */
extern "C" int cuda_phi_modulation(
    float* h_data, int n, float phase
) {
    float* d_data = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_data, (size_t)n * sizeof(float));
    if (err) return err;

    err = cudaMemcpy(d_data, h_data, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_data); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (n + blockSize - 1) / blockSize;
        phi_modulation_kernel<<<numBlocks, blockSize>>>(d_data, n, phase);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_data); return err; }

    err = cudaMemcpy(h_data, d_data, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    return err;
}

/**
 * cuda_cbm_flux_unfold_v1 - Host wrapper.
 */
extern "C" int cuda_cbm_flux_unfold_v1(
    const float* h_seed, float* h_weights, int dna_size, int matrix_size, float phi_flux
) {
    float *d_seed = NULL, *d_weights = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_seed, (size_t)dna_size * sizeof(float));
    if (err) return err;
    err = cudaMalloc(&d_weights, (size_t)matrix_size * sizeof(float));
    if (err) { cudaFree(d_seed); return err; }

    err = cudaMemcpy(d_seed, h_seed, (size_t)dna_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_seed); cudaFree(d_weights); return err; }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (matrix_size + blockSize - 1) / blockSize;
        cbm_flux_unfold_kernel<<<numBlocks, blockSize>>>(d_seed, d_weights, dna_size, matrix_size, phi_flux);
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_seed); cudaFree(d_weights); return err; }

    err = cudaMemcpy(h_weights, d_weights, (size_t)matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_seed);
    cudaFree(d_weights);
    return err;
}

/* =========================================================================
 *  PHYSX KERNEL WRAPPER
 * ========================================================================= */

/**
 * cuda_physx_crystal_dynamics - Host wrapper for physics simulation step.
 *
 * Host data layout: [pos(7), vel(7), mass, resonance] * n_bodies
 * Each body is 16 floats = 64 bytes.
 */
extern "C" int cuda_physx_crystal_dynamics(
    float* h_bodies,         /* host: packed body array (n_bodies * 16 floats) */
    const float* h_cosmos,   /* host: external forces (n_bodies * 7 floats), or NULL */
    int n_bodies,
    float dt
) {
    struct CrystalRigidBody *d_bodies = NULL;
    float *d_cosmos = NULL;
    cudaError_t err;
    int body_floats = 16;  /* pos(7) + vel(7) + mass + resonance */

    size_t bodies_bytes = (size_t)n_bodies * body_floats * sizeof(float);

    err = cudaMalloc(&d_bodies, bodies_bytes);
    if (err) return err;

    err = cudaMemcpy(d_bodies, h_bodies, bodies_bytes, cudaMemcpyHostToDevice);
    if (err) { cudaFree(d_bodies); return err; }

    if (h_cosmos != NULL) {
        size_t cosmos_bytes = (size_t)n_bodies * 7 * sizeof(float);
        err = cudaMalloc(&d_cosmos, cosmos_bytes);
        if (err) { cudaFree(d_bodies); return err; }
        err = cudaMemcpy(d_cosmos, h_cosmos, cosmos_bytes, cudaMemcpyHostToDevice);
        if (err) { cudaFree(d_bodies); cudaFree(d_cosmos); return err; }
    }

    {
        int blockSize = BLOCK_SIZE_1D;
        int numBlocks = (n_bodies + blockSize - 1) / blockSize;
        physx_crystal_dynamics_kernel<<<numBlocks, blockSize>>>(
            d_bodies, d_cosmos, n_bodies, dt
        );
    }

    err = cudaGetLastError();
    if (err) { cudaFree(d_bodies); if (d_cosmos) cudaFree(d_cosmos); return err; }

    err = cudaMemcpy(h_bodies, d_bodies, bodies_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_bodies);
    if (d_cosmos) cudaFree(d_cosmos);
    return err;
}

/**
 * cuda_get_device_info - Return device properties as a struct.
 */
extern "C" int cuda_get_device_info(
    char* name_buf, int name_buf_size,
    int* compute_major, int* compute_minor,
    size_t* total_memory
) {
    cudaError_t err;
    int device_count = 0;
    int dev = 0;
    cudaDeviceProp prop;

    err = cudaGetDeviceCount(&device_count);
    if (err) return err;
    if (device_count == 0) return cudaErrorNoDevice;

    err = cudaGetDevice(&dev);
    if (err) return err;

    err = cudaGetDeviceProperties(&prop, dev);
    if (err) return err;

    if (name_buf && name_buf_size > 0) {
        int len = (int)strlen(prop.name);
        int copy_len = (len < name_buf_size - 1) ? len : name_buf_size - 1;
        memcpy(name_buf, prop.name, (size_t)copy_len);
        name_buf[copy_len] = '\0';
    }
    if (compute_major) *compute_major = prop.major;
    if (compute_minor) *compute_minor = prop.minor;
    if (total_memory) *total_memory = prop.totalGlobalMem;

    return cudaSuccess;
}
