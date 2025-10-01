// SPDX-License-Identifier: GPL-3.0
/*
 * CUDA 12.0 + GCC 12 Compatibility Header
 * Purpose: Disable Intel AMX intrinsics that cause compilation errors
 * 
 * Known Issue: CUDA 12.0 includes GCC's amxtileintrin.h which contains
 * intrinsics not recognized by nvcc parser
 * 
 * Solution: Define stub implementations for problematic builtins
 * Reference: https://forums.developer.nvidia.com/t/cuda-12-0-gcc-12-compilation-error/242474
 */

#ifndef CUDA_COMPAT_H
#define CUDA_COMPAT_H

// Define stubs for Intel AMX builtins that nvcc doesn't recognize
#ifndef __builtin_ia32_ldtilecfg
#define __builtin_ia32_ldtilecfg(X) ((void)(X))
#endif

#ifndef __builtin_ia32_sttilecfg
#define __builtin_ia32_sttilecfg(X) ((void)(X))
#endif

#ifndef __builtin_ia32_tileloadd64
#define __builtin_ia32_tileloadd64(X, Y, Z) ((void)(X),(void)(Y),(void)(Z))
#endif

#ifndef __builtin_ia32_tilestored64
#define __builtin_ia32_tilestored64(X, Y, Z) ((void)(X),(void)(Y),(void)(Z))
#endif

#ifndef __builtin_ia32_tilerelease
#define __builtin_ia32_tilerelease() ((void)0)
#endif

#ifndef __builtin_ia32_tilezero
#define __builtin_ia32_tilezero(X) ((void)(X))
#endif

#endif // CUDA_COMPAT_H
