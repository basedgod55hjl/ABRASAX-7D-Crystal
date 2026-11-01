#pragma once

#ifdef _WIN32
#define NONCUDA_API __declspec(dllexport)
#else
#define NONCUDA_API
#endif

extern "C" {
NONCUDA_API const char* detect_backend();
NONCUDA_API int transfer_hidden_state(const float* src, float* dst, int n);
}
