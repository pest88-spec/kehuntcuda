# SPDX-License-Identifier: GPL-3.0-or-later
# @reuse_check 原有 shell incident 脚本缺失，仅能在此 Python 文件上增量扩展。
# Source-Fusion: CI incident logging helper created for KeyHunt-CUDA (2024-09-29).

"""Emit a lightweight incident report after failed CI runs，并可检测 CUDA 错误。"""

from __future__ import annotations

import json
import os
import re
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List


OUTPUT_PATH = Path(os.getenv("CI_INCIDENT_PATH", "artifacts/ci_incident.json"))
CUDA_ERROR_PATTERNS = (
    re.compile(r"cudaError", re.IGNORECASE),
    re.compile(r"CUDA_ERROR", re.IGNORECASE),
    re.compile(r"cuCtx", re.IGNORECASE),
    re.compile(r"invalid device", re.IGNORECASE),
)


def gather_context(reason: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "generated_at": now,
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "run_number": os.getenv("GITHUB_RUN_NUMBER"),
        "workflow": os.getenv("GITHUB_WORKFLOW"),
        "job": os.getenv("GITHUB_JOB"),
        "repository": os.getenv("GITHUB_REPOSITORY"),
        "ref": os.getenv("GITHUB_REF"),
        "sha": os.getenv("GITHUB_SHA"),
        "actor": os.getenv("GITHUB_ACTOR"),
        "reason": reason,
    }


def _scan_cuda_logs(paths: Iterable[Path]) -> List[str]:
    findings: List[str] = []
    for log_path in paths:
        if not log_path.exists():
            continue
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in CUDA_ERROR_PATTERNS:
            match = pattern.search(text)
            if match:
                findings.append(f"{log_path}: {match.group(0)}")
                break
    return findings


def main() -> int:
    parser = ArgumentParser(description="CI incident reporter")
    parser.add_argument("reason", nargs="*", help="失败原因描述")
    parser.add_argument("--cuda-log", action="append", default=[], help="需要扫描的 CUDA 日志路径")
    parser.add_argument("--fail-on-cuda-error", action="store_true", help="发现 CUDA 错误时返回非零")
    args = parser.parse_args()

    reason = " ".join(args.reason) if args.reason else "CI job failed"
    payload = gather_context(reason)

    cuda_logs = [Path(p) for p in args.cuda_log]
    if cuda_logs:
        cuda_findings = _scan_cuda_logs(cuda_logs)
        payload["cuda_logs"] = [str(p) for p in cuda_logs]
        payload["cuda_errors"] = cuda_findings
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ci_incident_report: wrote {OUTPUT_PATH}")

    if cuda_logs and args.fail_on_cuda_error and payload.get("cuda_errors"):
        print("Detected CUDA errors:")
        for item in payload["cuda_errors"]:
            print(f"  - {item}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
