// SPDX-License-Identifier: GPL-3.0-or-later
/**
 * @file tests/stage3/test_scan_kernel_correctness.cu
 * @origin docs/stage3_phase6_progress.md
 * @origin_path tests/stage3/test_scan_kernel_correctness.cu
 * @origin_commit 905a1627ce6c5ea8383f633d34f6f73893da5859
 * @origin_license GPL-3.0-or-later
 * @modified_by AI-Agent (Droid)
 * @modifications "Stage 3 scan_kernel helper validation using targeted CUDA kernels"
 * @fusion_date 2025-10-01
 * @spdx_license_identifier GPL-3.0-or-later
 */

// CUDA 12.0 + GCC 12 compatibility fix
#include "cuda_compat.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include "../../GPU/GPUCompute.h"
#include "../../GPU/GPUHash.h"

#ifdef CUDA_CHECK
#undef CUDA_CHECK
#endif
#include "../../GPU/CudaChecks.h"

// ===== GPU Test Kernels =====
// These kernels call the device helper functions for testing

/**
 * Test kernel for scan_is_infinity
 * @reuse_check_L5 New: Minimal test kernel for Stage 3 validation
 */
__global__ void test_scan_is_infinity_kernel(
    const uint64_t* px,
    const uint64_t* py,
    bool* result)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = scan_is_infinity(px, py);
    }
}

/**
 * Test kernel for scan_match_hash160
 * @reuse_check_L5 New: Minimal test kernel for Stage 3 validation
 */
__global__ void test_scan_match_hash160_kernel(
    const uint32_t* hash1,
    const uint32_t* hash2,
    bool* result)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = scan_match_hash160(hash1, hash2);
    }
}

/**
 * Test kernel for scan_hash160_compressed
 * @reuse_check_L5 New: Minimal test kernel for Stage 3 validation
 */
__global__ void test_scan_hash160_compressed_kernel(
    const uint64_t* px,
    const uint64_t* py,
    uint8_t* hash160)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        scan_hash160_compressed(px, py, hash160);
    }
}

/**
 * Test kernel for scan_point_add
 * @reuse_check_L1 Current: scan_point_add device helper
 * @reuse_check_L2 VanitySearch: GPU/GPUCompute.h::compute_ec_point_add
 * @sot_ref SOT-CRYPTO: secp256k1 generator multiples (2G)
 */
__global__ void test_scan_point_add_kernel(
    const uint64_t* p1x,
    const uint64_t* p1y,
    const uint64_t* p2x,
    const uint64_t* p2y,
    uint64_t* outx,
    uint64_t* outy)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint64_t local_x[4];
        uint64_t local_y[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            local_x[i] = p1x[i];
            local_y[i] = p1y[i];
        }

        scan_point_add(local_x, local_y, p2x, p2y);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            outx[i] = local_x[i];
            outy[i] = local_y[i];
        }
    }
}

class ScanKernelCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err == cudaErrorNoDevice) {
            GTEST_SKIP() << "No CUDA device available";
        }
        CUDA_CHECK(err);
    }
    
    void TearDown() override {
        CUDA_CHECK(cudaDeviceReset());
    }
};

TEST_F(ScanKernelCorrectnessTest, ScanIsInfinity_ZeroPoint_ReturnsTrue) {
    // Allocate device memory
    uint64_t zero_point[4] = {0, 0, 0, 0};
    uint64_t *d_px, *d_py;
    bool *d_result;
    bool h_result;
    
    CUDA_CHECK(cudaMalloc(&d_px, 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_py, 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(bool)));
    
    // Copy zero point to device
    CUDA_CHECK(cudaMemcpy(d_px, zero_point, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, zero_point, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    // Launch test kernel
    test_scan_is_infinity_kernel<<<1, 1>>>(d_px, d_py, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
    
    // Verify
    EXPECT_TRUE(h_result) << "Zero point should be identified as infinity";
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_result));
}

TEST_F(ScanKernelCorrectnessTest, ScanIsInfinity_NonZeroPoint_ReturnsFalse) {
    // Generator point G (non-zero) - little-endian limbs
    uint64_t gx[4] = {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
                      0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};
    uint64_t gy[4] = {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
                      0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL};
    
    uint64_t *d_px, *d_py;
    bool *d_result;
    bool h_result;
    
    CUDA_CHECK(cudaMalloc(&d_px, 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_py, 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(bool)));
    
    // Copy generator point to device
    CUDA_CHECK(cudaMemcpy(d_px, gx, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, gy, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    // Launch test kernel
    test_scan_is_infinity_kernel<<<1, 1>>>(d_px, d_py, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
    
    // Verify
    EXPECT_FALSE(h_result) << "Generator point should NOT be infinity";
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_result));
}

TEST_F(ScanKernelCorrectnessTest, HelperFunctions_Hash160Compressed_PlaceholderForPhase6) {
    // TDD Red Phase: Test framework exists, implementation pending  
    // Will test scan_hash160_compressed() matches known test vectors
    GTEST_SKIP() << "Phase 6: Implementation and GPU test kernels pending";
}

TEST_F(ScanKernelCorrectnessTest, HelperFunctions_Hash160Uncompressed_PlaceholderForPhase6) {
    // TDD Red Phase: Test framework exists, implementation pending
    // Will test scan_hash160_uncompressed() matches known test vectors
    GTEST_SKIP() << "Phase 6: Implementation and GPU test kernels pending";
}

TEST_F(ScanKernelCorrectnessTest, ScanMatchHash160_IdenticalHashes_ReturnsTrue) {
    // Test with identical hash160 values
    uint32_t hash1[5] = {0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222, 0x33333333};
    uint32_t hash2[5] = {0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222, 0x33333333};
    
    uint32_t *d_hash1, *d_hash2;
    bool *d_result;
    bool h_result;
    
    CUDA_CHECK(cudaMalloc(&d_hash1, 5 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hash2, 5 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(bool)));
    
    CUDA_CHECK(cudaMemcpy(d_hash1, hash1, 5 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hash2, hash2, 5 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    test_scan_match_hash160_kernel<<<1, 1>>>(d_hash1, d_hash2, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
    
    EXPECT_TRUE(h_result) << "Identical hashes should match";
    
    CUDA_CHECK(cudaFree(d_hash1));
    CUDA_CHECK(cudaFree(d_hash2));
    CUDA_CHECK(cudaFree(d_result));
}

TEST_F(ScanKernelCorrectnessTest, ScanMatchHash160_DifferentHashes_ReturnsFalse) {
    // Test with different hash160 values (differ in last byte)
    uint32_t hash1[5] = {0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222, 0x33333333};
    uint32_t hash2[5] = {0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222, 0x33333334};
    
    uint32_t *d_hash1, *d_hash2;
    bool *d_result;
    bool h_result;
    
    CUDA_CHECK(cudaMalloc(&d_hash1, 5 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hash2, 5 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(bool)));
    
    CUDA_CHECK(cudaMemcpy(d_hash1, hash1, 5 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hash2, hash2, 5 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    test_scan_match_hash160_kernel<<<1, 1>>>(d_hash1, d_hash2, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
    
    EXPECT_FALSE(h_result) << "Different hashes should NOT match";
    
    CUDA_CHECK(cudaFree(d_hash1));
    CUDA_CHECK(cudaFree(d_hash2));
    CUDA_CHECK(cudaFree(d_result));
}

TEST_F(ScanKernelCorrectnessTest, ScanHash160Compressed_GeneratorPoint_KnownVector) {
    // Known test vector: Generator point G (k=1)
    // Expected Bitcoin address: 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
    // Expected hash160: 751e76e8199196d454941c45d1b3a323f1433bd6
    
    // Generator point G coordinates (little-endian limbs: limbs[0]=lowest 64 bits)
    uint64_t gx[4] = {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
                      0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};
    uint64_t gy[4] = {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
                      0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL};
    
    // Expected hash160 (big-endian bytes) - verified against Bitcoin Core
    uint8_t expected_hash160[20] = {
        0x75, 0x1e, 0x76, 0xe8, 0x19, 0x91, 0x96, 0xd4,
        0x54, 0x94, 0x1c, 0x45, 0xd1, 0xb3, 0xa3, 0x23,
        0xf1, 0x43, 0x3b, 0xd6
    };
    
    uint64_t *d_px, *d_py;
    uint8_t *d_hash160;
    uint8_t h_hash160[20];
    
    CUDA_CHECK(cudaMalloc(&d_px, 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_py, 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_hash160, 20));
    
    CUDA_CHECK(cudaMemcpy(d_px, gx, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, gy, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    test_scan_hash160_compressed_kernel<<<1, 1>>>(d_px, d_py, d_hash160);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_hash160, d_hash160, 20, cudaMemcpyDeviceToHost));
    
    // Verify each byte of hash160
    bool all_match = true;
    for (int i = 0; i < 20; i++) {
        if (h_hash160[i] != expected_hash160[i]) {
            all_match = false;
            printf("Byte %d mismatch: got 0x%02x, expected 0x%02x\n", 
                   i, h_hash160[i], expected_hash160[i]);
        }
    }
    
    EXPECT_TRUE(all_match) << "Hash160 does not match known test vector for G";
    
    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_hash160));
}

TEST_F(ScanKernelCorrectnessTest, ScanPointAdd_GeneratesDoubleGenerator) {
    // P1 = G, P2 = G, expect 2G
    const uint64_t gx[4] = {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
                            0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};
    const uint64_t gy[4] = {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
                            0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL};
    const uint64_t expected_x[4] = {0xABAC09B95C709EE5ULL, 0x5C778E4B8CEF3CA7ULL,
                                    0x3045406E95C07CD8ULL, 0xC6047F9441ED7D6DULL};
    const uint64_t expected_y[4] = {0x236431A950CFE52AULL, 0xF7F632653266D0E1ULL,
                                    0xA3C58419466CEAEEULL, 0x1AE168FEA63DC339ULL};

    uint64_t *d_p1x, *d_p1y, *d_p2x, *d_p2y, *d_outx, *d_outy;
    uint64_t h_outx[4];
    uint64_t h_outy[4];

    CUDA_CHECK(cudaMalloc(&d_p1x, 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_p1y, 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_p2x, 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_p2y, 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_outx, 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_outy, 4 * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_p1x, gx, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p1y, gy, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p2x, gx, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p2y, gy, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    test_scan_point_add_kernel<<<1, 1>>>(d_p1x, d_p1y, d_p2x, d_p2y, d_outx, d_outy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_outx, d_outx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_outy, d_outy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(h_outx[i], expected_x[i]) << "Mismatch in X limb " << i;
        EXPECT_EQ(h_outy[i], expected_y[i]) << "Mismatch in Y limb " << i;
    }

    CUDA_CHECK(cudaFree(d_p1x));
    CUDA_CHECK(cudaFree(d_p1y));
    CUDA_CHECK(cudaFree(d_p2x));
    CUDA_CHECK(cudaFree(d_p2y));
    CUDA_CHECK(cudaFree(d_outx));
    CUDA_CHECK(cudaFree(d_outy));
}

TEST_F(ScanKernelCorrectnessTest, ScanBloomCheck_WrapperCorrectness_Pending) {
    // Pending: Requires Bloom filter initialization
    // Will verify scan_bloom_check() wrapper delegates correctly to BloomCheck()
    GTEST_SKIP() << "Pending: Need Bloom filter setup (Phase 7)";
}

// Additional comprehensive tests for Phase 7:
// - scan_serialize_compressed byte order verification
// - scan_record_match atomic operation testing
// - Fuzz testing with 2^16 random keys (CI) / 2^20 (nightly)
