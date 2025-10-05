#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# @reuse_check Stage3 pipeline requires dedicated benchmark script; no reusable alternative located in repo or reference sources.
# Source-Fusion: AI-Agent开发与运行防错方案.md §4.2 性能反退化监控 → benchmark/bench_gpu.sh 模板
# @file benchmark/bench_gpu.sh
# @origin AI-Agent开发与运行防错方案.md
# @origin_path benchmark/bench_gpu.sh
# @origin_license GPL-3.0-or-later
# @modified_by AI-Agent (Droid)
# @modifications "Stage3 GPU benchmark runner with baseline logging"
# @fusion_date 2025-10-01

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULT_ROOT="${PROJECT_ROOT}/benchmark/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${RESULT_ROOT}/${TIMESTAMP}"

SM_LIST="auto"
ITERATIONS="1000000"
BINARY="${PROJECT_ROOT}/build/keyhunt_cuda"

usage() {
    cat <<EOF
Usage: ${BASH_SOURCE[0]} [--sm-list "sm1,sm2"] [--iterations N] [--binary PATH]

Options:
  --sm-list       Comma-separated list of SM architectures to benchmark (default: auto)
  --iterations    Number of iterations per benchmark run (default: 1000000)
  --binary        Path to benchmark executable (default: build/keyhunt_cuda)
  -h, --help      Show this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sm-list)
            SM_LIST="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --binary)
            BINARY="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: Benchmark binary not found or not executable: ${BINARY}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

RESULT_JSON="${OUTPUT_DIR}/results.json"
SUMMARY_LOG="${OUTPUT_DIR}/summary.log"

echo "[bench_gpu] Writing results to ${OUTPUT_DIR}" | tee "${SUMMARY_LOG}"
echo "[bench_gpu] Using SM list: ${SM_LIST}" | tee -a "${SUMMARY_LOG}"
echo "[bench_gpu] Iterations: ${ITERATIONS}" | tee -a "${SUMMARY_LOG}"

"${BINARY}" \
    --benchmark \
    --sm-list "${SM_LIST}" \
    --iterations "${ITERATIONS}" \
    --output "${RESULT_JSON}" \
    2>&1 | tee -a "${SUMMARY_LOG}"

ln -sfn "${RESULT_JSON}" "${RESULT_ROOT}/latest.json"
ln -sfn "${SUMMARY_LOG}" "${RESULT_ROOT}/latest.log"

echo "[bench_gpu] Completed" | tee -a "${SUMMARY_LOG}"
