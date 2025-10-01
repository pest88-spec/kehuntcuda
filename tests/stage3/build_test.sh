#!/bin/bash
# SPDX-License-Identifier: GPL-3.0
# Stage 3 Test Build Script
# Compiles test_scan_kernel_correctness.cu with gtest

set -e

echo "===  Stage 3 Test Build Script ==="

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

# Check if gtest is available
if [ ! -f "/usr/lib/libgtest.a" ] && [ ! -f "/usr/local/lib/libgtest.a" ]; then
    echo "WARNING: gtest library not found. Attempting to use system gtest..."
fi

# Build test
echo "Compiling test_scan_kernel_correctness.cu..."

nvcc -o test_scan_kernel_correctness \
    test_scan_kernel_correctness.cu \
    -I../../ \
    -I/usr/include \
    -L/usr/lib/x86_64-linux-gnu \
    -lgtest -lgtest_main -lpthread \
    -std=c++14 \
    -gencode=arch=compute_75,code=sm_75 \
    --allow-unsupported-compiler \
    --compiler-options "-Wall -Wextra -D_FORCE_INLINES" \
    -Xcompiler "-Wno-psabi" \
    2>&1 | tee build.log

if [ $? -eq 0 ]; then
    echo "✅ Build successful"
    echo "Run: ./test_scan_kernel_correctness"
else
    echo "❌ Build failed. Check build.log for details."
    exit 1
fi
