#include "flux_core.h"

/**
 * Rule 110 Bit-Packed Kernel
 * Processes 32 cells per thread using bitwise parallelism.
 * Logic: (C & R) ^ (C | R) ^ L
 */
extern "C" __global__ void rule110_evolution_kernel(
    unsigned int* d_out, 
    const unsigned int* d_in, 
    int width_words,
    float entropy_bias
) {
    // Dynamic Shared Memory for Halo Exchange (Passed in via launcher)
    extern __shared__ unsigned int s_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // 1. Load Data into Shared Memory with Halo (Periodic Boundaries)
    if (gid < width_words) {
        s_data[tid + 1] = d_in[gid];
    }
    
    if (tid == 0) {
        int left_idx = (gid > 0) ? gid - 1 : width_words - 1;
        s_data[0] = d_in[left_idx];
    }
    if (tid == blockDim.x - 1 || gid == width_words - 1) {
        int right_idx = (gid < width_words - 1) ? gid + 1 : 0;
        s_data[tid + 2] = d_in[right_idx];
    }
    
    __syncthreads();

    if (gid >= width_words) return;

    // 2. Extract Neighbors using sliding window logic
    unsigned int center = s_data[tid + 1];
    unsigned int left_word = s_data[tid];
    unsigned int right_word = s_data[tid + 2];

    // Create L and R neighbors for each bit in C
    unsigned int L = (left_word << 31) | (center >> 1);
    unsigned int R = (center << 1) | (right_word >> 31);
    unsigned int C = center;

    // 3. Apply Rule 110: (C & R) ^ (C | R) ^ L
    unsigned int next_state = (C & R) ^ (C | R) ^ L;

    // 4. Entropy Modulation & Writeback
    d_out[gid] = next_state;
}
