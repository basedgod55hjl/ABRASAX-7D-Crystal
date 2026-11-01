#include "backend_bridge.h"

#include <cstdlib>
#include <cstring>

extern "C" NONCUDA_API const char* detect_backend() {
    // Non-CUDA backend order: Vulkan -> OpenCL -> DirectML -> CPU.
    // Real probing can be added here (vkEnumeratePhysicalDevices, clGetPlatformIDs, DML adapter checks).
    const char* force = std::getenv("NONCUDA_BACKEND");
    if (force && std::strlen(force) > 0) {
        return force;
    }
    return "vulkan";
}

extern "C" NONCUDA_API int transfer_hidden_state(const float* src, float* dst, int n) {
    if (!src || !dst || n <= 0) {
        return 0;
    }
    std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(n));
    return n;
}
