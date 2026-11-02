#pragma once
#include <cuda_runtime.h>
#include <math.h>
namespace cbm_optimization {
    __global__ void k_damped_tracking_step(float* restrict state, const float* restrict target, int n, float K, float eta) {}
}
