// SPDX-License-Identifier: GPL-3.0-or-later
// Source-Fusion Provenance: KeyHunt-CUDA project
// Original: KeyHunt-CUDA (2025-09-30)

#ifndef GPU_CUDA_CHECKS_H
#define GPU_CUDA_CHECKS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

inline void cudaCheckImpl(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d -> %s\n", file, line, cudaGetErrorString(err));
        std::abort();
    }
}

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) cudaCheckImpl((call), __FILE__, __LINE__)
#endif

#endif // GPU_CUDA_CHECKS_H
