#!/bin/bash
# SPDX-License-Identifier: GPL-3.0
# Build script for CPU-GPU consistency test

set -e
set -o pipefail

echo "=== Building CPU-GPU Consistency Test ==="

# Check dependencies
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found"
    exit 1
fi

if ! pkg-config --exists libsecp256k1; then
    echo "ERROR: libsecp256k1 not found"
    exit 1
fi

# Get libsecp256k1 flags
SECP_CFLAGS=$(pkg-config --cflags libsecp256k1)
SECP_LIBS=$(pkg-config --libs libsecp256k1)

echo "Building with:"
echo "  CUDA: $(nvcc --version | grep release)"
echo "  libsecp256k1: $(pkg-config --modversion libsecp256k1)"
echo "  gtest: system"

# Compile
nvcc -x cu -o cpu_gpu_consistency_test \
    cpu_gpu_consistency_test.cpp \
    -I../../ \
    -I. \
    $SECP_CFLAGS \
    -L/usr/lib/x86_64-linux-gnu \
    -lgtest -lgtest_main -lpthread \
    $SECP_LIBS \
    -lssl -lcrypto \
    -std=c++14 \
    -gencode=arch=compute_75,code=sm_75 \
    --allow-unsupported-compiler \
    2>&1 | tee build_consistency.log

if [ $? -eq 0 ]; then
    echo "✅ Build successful"
    echo "Run: ./cpu_gpu_consistency_test"
else
    echo "❌ Build failed. Check build_consistency.log"
    exit 1
fi
