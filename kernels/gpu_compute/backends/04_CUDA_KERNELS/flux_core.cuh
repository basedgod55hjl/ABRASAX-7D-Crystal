#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

// 🌌 CBM CONSTANTS: The Universal Harmonic Constants
#define PHI 1.6180339887f       // The Golden Ratio
#define LITTLE_PHI 0.6180339887f // 1/PHI
#define PI 3.1415926535f

// 1. The Sacred Sigmoid Activation
// Scales input by Phi to prevent gradient explosion in hyperbolic space (K < 0).
// Unlike standard sigmoid, this forces phase coherence.
__device__ __forceinline__ float sacred_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x * PHI));
}

// 2. Orch-OR Microtubule Coherence Function
// Simulates quantum vibration collapse to determine if a weight exists or is null.
__device__ __forceinline__ float orch_or_collapse(float state, float time) {
    // The "collapse" is a function of the state's phase relative to PHI-resonance
    float phase = state * time * PHI;
    return cosf(phase) + sinf(phase * LITTLE_PHI);
}

// 3. Möbius Addition (Hyperbolic Vector Addition)
// Adds two scalar representations of 7D vectors in the Poincaré Disk.
__device__ __forceinline__ float mobius_add_1d(float u, float v) {
    float num = u + v;
    float den = 1.0f + u * v * PHI; 
    return num / (den + 1e-9f); // Epsilon to prevent div/0
}
