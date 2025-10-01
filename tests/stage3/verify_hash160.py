#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0
"""
Verify Hash160 Computation - Reference Implementation
Purpose: Generate reference values for GPU debugging
"""

import hashlib
import struct

def bytes_to_hex(data):
    """Convert bytes to hex string"""
    return data.hex()

def limbs_to_bigendian(limbs):
    """Convert little-endian limbs to big-endian bytes"""
    # limbs[0] = lowest 64 bits, limbs[3] = highest 64 bits
    result = b''
    for limb in reversed(limbs):
        result += struct.pack('>Q', limb)  # Big-endian uint64
    return result

def compute_hash160(gx_limbs, gy_limbs):
    """
    Compute hash160 for given point coordinates
    
    Args:
        gx_limbs: list of 4 uint64_t (little-endian limbs)
        gy_limbs: list of 4 uint64_t (little-endian limbs)
    
    Returns:
        tuple: (compressed_pubkey, sha256_hash, hash160)
    """
    # Step 1: Convert coordinates to big-endian bytes
    gx_bytes = limbs_to_bigendian(gx_limbs)
    gy_bytes = limbs_to_bigendian(gy_limbs)
    
    print("=== Coordinate Conversion ===")
    print(f"Gx limbs: {[hex(x) for x in gx_limbs]}")
    print(f"Gx bytes (BE): {bytes_to_hex(gx_bytes)}")
    print(f"Gy limbs: {[hex(y) for y in gy_limbs]}")
    print(f"Gy bytes (BE): {bytes_to_hex(gy_bytes)}")
    
    # Step 2: Determine y parity
    y_is_odd = gy_limbs[0] & 1  # Check lowest bit of lowest limb
    prefix = 0x03 if y_is_odd else 0x02
    
    print(f"\n=== Y Parity ===")
    print(f"Y lowest limb: 0x{gy_limbs[0]:016x}")
    print(f"Y is odd: {y_is_odd}")
    print(f"Prefix: 0x{prefix:02x}")
    
    # Step 3: Create compressed public key
    compressed_pubkey = bytes([prefix]) + gx_bytes
    
    print(f"\n=== Compressed Public Key (33 bytes) ===")
    print(f"{bytes_to_hex(compressed_pubkey)}")
    
    # Step 4: Compute SHA256
    sha256_hash = hashlib.sha256(compressed_pubkey).digest()
    
    print(f"\n=== SHA256 Hash (32 bytes) ===")
    print(f"{bytes_to_hex(sha256_hash)}")
    
    # Step 5: Compute RIPEMD160
    ripemd160 = hashlib.new('ripemd160')
    ripemd160.update(sha256_hash)
    hash160 = ripemd160.digest()
    
    print(f"\n=== RIPEMD160 / Hash160 (20 bytes) ===")
    print(f"{bytes_to_hex(hash160)}")
    
    return compressed_pubkey, sha256_hash, hash160

def test_generator_point():
    """Test with generator point G (k=1)"""
    print("=" * 60)
    print("Testing Generator Point G (k=1)")
    print("=" * 60)
    
    # Generator point coordinates (little-endian limbs)
    gx_limbs = [
        0x59F2815B16F81798,
        0x029BFCDB2DCE28D9,
        0x55A06295CE870B07,
        0x79BE667EF9DCBBAC
    ]
    
    gy_limbs = [
        0x9C47D08FFB10D4B8,
        0xFD17B448A6855419,
        0x5DA4FBFC0E1108A8,
        0x483ADA7726A3C465
    ]
    
    compressed, sha256, hash160 = compute_hash160(gx_limbs, gy_limbs)
    
    # Expected values
    expected_compressed = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
    expected_hash160 = "751e76e8199196d454941c45d1b3a323f1433bd6"
    expected_address = "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH"
    
    print(f"\n=== Verification ===")
    print(f"Expected compressed: {expected_compressed}")
    print(f"Actual compressed:   {bytes_to_hex(compressed)}")
    print(f"Match: {bytes_to_hex(compressed) == expected_compressed}")
    
    print(f"\nExpected hash160: {expected_hash160}")
    print(f"Actual hash160:   {bytes_to_hex(hash160)}")
    print(f"Match: {bytes_to_hex(hash160) == expected_hash160}")
    
    print(f"\nBitcoin Address: {expected_address}")
    
    return bytes_to_hex(compressed) == expected_compressed

def test_limbs_interpretation():
    """Test different interpretations of limbs"""
    print("\n" + "=" * 60)
    print("Testing Limbs Interpretation")
    print("=" * 60)
    
    # Test case: first 8 bytes of gx
    # Expected: 79 BE 66 7E F9 DC BB AC (big-endian)
    
    print("\n=== Interpretation 1: limbs[0] = lowest 64 bits ===")
    limbs_le = [0x59F2815B16F81798]  # Little-endian value
    result_be = struct.pack('>Q', limbs_le[0])
    print(f"Limb[0] = 0x{limbs_le[0]:016x}")
    print(f"As big-endian bytes: {bytes_to_hex(result_be)}")
    
    print("\n=== Interpretation 2: limbs[3] = lowest 64 bits ===")
    limbs_reversed = [0x79BE667EF9DCBBAC]
    result_be2 = struct.pack('>Q', limbs_reversed[0])
    print(f"Limb[3] = 0x{limbs_reversed[0]:016x}")
    print(f"As big-endian bytes: {bytes_to_hex(result_be2)}")
    
    print(f"\n✓ Correct interpretation: limbs[3] stores highest 64 bits")
    print(f"  79be667ef9dcbbac is at the START of gx (big-endian)")
    print(f"  59f2815b16f81798 is at the END of gx (big-endian)")

if __name__ == "__main__":
    # Test limbs interpretation
    test_limbs_interpretation()
    
    # Test generator point
    success = test_generator_point()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ Test failed!")
        print("=" * 60)
        exit(1)
