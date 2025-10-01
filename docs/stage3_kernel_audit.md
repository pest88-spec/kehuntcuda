
#include "ScanKernel_Core.cuh"
#include "ScanKernel_Hash.cuh"
#include "ScanKernel_Match.cuh"
#include "ScalarTypes.cuh"
#include "CudaChecks.h"

__global__ void scan_kernel(
    const uint64_t *priv_limbs,      // Input: n×4 limbs (little-endian Scalar256)
    size_t n,                         // Number of key candidates (public test inputs)
    const uint8_t *target_hashes,     // Target hash160 array
    size_t num_targets,               // Number of targets
    uint32_t *found_flags,            // Output: match flags
    uint64_t *found_keys,             // Output: matching key limbs
    uint32_t *found_count             // Output: number of matches
)
{
    // Thread identification
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    
    // Process keys in grid-stride loop
    for (size_t i = tid; i < n; i += stride) {
        // Load Scalar256 key (4 limbs)
        Scalar256 k;
        k.limbs[0] = priv_limbs[i * 4 + 0];
        k.limbs[1] = priv_limbs[i * 4 + 1];
        k.limbs[2] = priv_limbs[i * 4 + 2];
        k.limbs[3] = priv_limbs[i * 4 + 3];
        
        // Stage B: EC Point Computation (P = k × G)
        uint64_t px[4], py[4];
        scalar_mult_generator(k.limbs, px, py); // Uses precomputed tables
        
        // Stage C: Public Key Serialization (compressed)
        uint8_t pubkey[33];
        serialize_compressed_pubkey(px, py, pubkey);
        
        // Stage D: Address Generation (SHA256 + RIPEMD160)
        uint8_t hash160[20];
        compute_hash160_compressed(px, py, hash160);
        
        // Stage E: Target Matching (Bloom filter or direct)
        bool match = check_target_match(hash160, target_hashes, num_targets);
        
        if (match) {
            // Record match atomically
            uint32_t pos = atomicAdd(found_count, 1);
            if (pos < MAX_FOUND) {
                found_flags[pos] = 1;
                found_keys[pos * 4 + 0] = k.limbs[0];
                found_keys[pos * 4 + 1] = k.limbs[1];
                found_keys[pos * 4 + 2] = k.limbs[2];
                found_keys[pos * 4 + 3] = k.limbs[3];
            }
        }
    }
}
```

**Key Design Decisions:**
- Grid-stride loop for scalability (handles any `n`)
- Uses `Scalar256` type from GPU/ScalarTypes.cuh (Stage 2 compliance)
- Delegates to extracted helper functions (provenance preserved)
- Atomic operations for thread-safe result recording
- Early exit on `MAX_FOUND` to prevent buffer overflow

### Phase 3: CPU Baseline Validator (Week 1, Day 6-7)

**File:** `tests/validators/cpu_baseline_validator.cpp` (new)

```cpp
// Uses libsecp256k1 for EC operations
// Compares GPU scan_kernel output byte-by-byte
// Generates deterministic test vectors
```

### Phase 4: Test Suite (Week 2, Day 1-3)

**Files:**
- `tests/stage3/test_scan_kernel_correctness.cu`
- `tests/stage3/test_scan_kernel_performance.cu`
- `tests/stage3/test_scan_kernel_fuzz.cu`

### Phase 5: CI Integration (Week 2, Day 4-5)

**Files:**
- `.github/workflows/ci.yml` (update)
- `scripts/check_naming.py` (new - detect `_fix/_temp/_broken` files)
- `tests/perf/stage3_baseline.json` (new)

---

## Risk Assessment

### High Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| GLV code provenance unknown | License violation | Remove GLV from Stage 3, defer to Stage 4 with proper source |
| Multiple kernel variants | Merge conflicts | Consolidate before Stage 3 start |
| Hardcoded PUZZLE71 logic | Not generalizable | Parameterize or separate puzzle-specific code |

### Medium Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance regression | Slower than existing | Establish baseline before refactor |
| ABI change breaks host code | Build failures | Update GPUEngine.cu incrementally |
| Insufficient test coverage | Bugs in production | Mandate ≥80% coverage per PROJECT_CHARTER |

---

## Compliance Checklist

Per PROJECT_CHARTER.md:

- [ ] All reused code has provenance headers
- [ ] VanitySearch snapshot captured with SHA256 checksums
- [ ] No `_fix`, `_temp`, `_broken`, `_copy`, `_new` files remain
- [ ] Kernel ABI matches docs/abi_kernel.md specification
- [ ] CPU baseline validator implemented using libsecp256k1
- [ ] Test suite covers compressed + uncompressed modes
- [ ] CI integration with performance regression detection
- [ ] Documentation updated (source_fusion_report.md, license_matrix.md)

---

## Conclusion

**Audit Status:** ✅ Complete  
**Reusable Components:** 15+ functions identified  
**Cleanup Required:** 6 files flagged for deletion  
**Provenance:** Mostly VanitySearch (GPLv3), some custom code needs review

**Next Steps:**
1. Execute Phase 1: Component extraction into reusable headers
2. Delete non-compliant `_fix/_temp/_broken` files
3. Implement standardized `scan_kernel()` per abi_kernel.md
4. Build CPU baseline validator and test suite
5. Integrate into CI with performance tracking

**Estimated Effort:** 2 weeks (matches PROJECT_CHARTER Stage 3 timeline)
