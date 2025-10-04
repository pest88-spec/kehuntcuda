# SPDX-License-Identifier: GPL-3.0-or-later
# Source-Fusion Provenance: Original CI guardrail script authored for KeyHunt-CUDA (2024-09-29).

"""Incremental diff checks for AI-governed CI pipelines.

This script enforces repository rules for newly added or modified files:
  * Rejects filenames that look like anonymous copies (`*_new`, `*_copy`).
  * Flags mock artefacts accidentally committed (paths containing `mock/` etc.).
  * Optional diff 规模、占位符扫描、白名单校验，用来取代缺失的 shell 守卫脚本。

The diff base can be controlled via environment variables:
  * GITHUB_BASE_REF (branch name in pull requests)
  * CI_BASE_SHA (explicit commit SHA)
  * CI_DIFF_RANGE (explicit git diff range such as `origin/main...HEAD`)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


FORBIDDEN_SUFFIXES = ("_new", "_copy", "_backup")
FORBIDDEN_DIR_PATTERNS = (
    re.compile(r"(^|/)mocks?(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)__mocks?__(/|$)", re.IGNORECASE),
)

PLACEHOLDER_PATTERNS = (
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bFIXME\b", re.IGNORECASE),
    re.compile(r"mock_", re.IGNORECASE),
    re.compile(r"placeholder", re.IGNORECASE),
)

DEFAULT_ALLOWED_NEW_FILES = Path("docs/allowed_files.txt")


@dataclass
class DiffEntry:
    status: str
    path: str


@dataclass
class DiffContext:
    entries: List["DiffEntry"]
    range_args: Sequence[str]


def _run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def compute_diff_entries() -> DiffContext:
    diff_range = os.getenv("CI_DIFF_RANGE")
    if diff_range:
        cmd = ["git", "diff", "--name-status", diff_range]
        result = _run(cmd)
        return DiffContext(_parse_diff_output(result.stdout), (diff_range,))

    base_sha = os.getenv("CI_BASE_SHA")
    if base_sha:
        cmd = ["git", "diff", "--name-status", base_sha, "HEAD"]
        result = _run(cmd)
        return DiffContext(_parse_diff_output(result.stdout), (base_sha, "HEAD"))

    base_branch = os.getenv("GITHUB_BASE_REF")
    if base_branch:
        _run(["git", "fetch", "origin", base_branch, "--depth", "1"])
        merge_base_result = _run(["git", "merge-base", "HEAD", f"origin/{base_branch}"], check=False)
    else:
        _run(["git", "fetch", "origin", "main", "--depth", "1"])
        merge_base_result = _run(["git", "merge-base", "HEAD", "origin/main"], check=False)

    # If merge-base fails (no common ancestor), compare against HEAD~1 or skip check
    if merge_base_result.returncode != 0:
        # For direct pushes to main, compare against previous commit
        prev_commit_result = _run(["git", "rev-parse", "HEAD~1"], check=False)
        if prev_commit_result.returncode == 0:
            merge_base = prev_commit_result.stdout.strip()
        else:
            # First commit on branch, no diff to check
            return DiffContext([], ())
    else:
        merge_base = merge_base_result.stdout.strip()

    diff = _run(["git", "diff", "--name-status", merge_base, "HEAD"])
    return DiffContext(_parse_diff_output(diff.stdout), (merge_base, "HEAD"))


def _parse_diff_output(output: str) -> List[DiffEntry]:
    entries: List[DiffEntry] = []
    for line in output.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        status, path = parts
        entries.append(DiffEntry(status=status.strip(), path=path.strip()))
    return entries


def _violates_suffix(path: str) -> bool:
    lower = path.lower()
    return any(lower.endswith(suffix) or lower.endswith(suffix + os.path.splitext(lower)[1]) for suffix in FORBIDDEN_SUFFIXES)


def _violates_mock(path: str) -> bool:
    return any(pattern.search(path) for pattern in FORBIDDEN_DIR_PATTERNS)


def _load_allowed_paths(cli_allow: Iterable[str]) -> List[str]:
    allow: List[str] = list(cli_allow)
    if DEFAULT_ALLOWED_NEW_FILES.exists():
        try:
            with DEFAULT_ALLOWED_NEW_FILES.open("r", encoding="utf-8") as fh:
                for line in fh:
                    striped = line.strip()
                    if not striped or striped.startswith("#"):
                        continue
                    allow.append(striped)
        except OSError:
            pass
    return allow


def _compute_diff_stats(range_args: Sequence[str]) -> tuple[int, int]:
    if not range_args:
        return 0, 0
    cmd = ["git", "diff", "--numstat", *range_args]
    result = _run(cmd)
    added = 0
    deleted = 0
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            add = int(parts[0]) if parts[0].isdigit() else 0
            delete = int(parts[1]) if parts[1].isdigit() else 0
        except ValueError:
            continue
        added += add
        deleted += delete
    return added, deleted


def _scan_placeholders(entries: Iterable[DiffEntry]) -> List[str]:
    failures: List[str] = []
    for entry in entries:
        path = Path(entry.path)
        if not path.exists() or path.is_dir():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.search(text):
                failures.append(f"Forbidden placeholder `{pattern.pattern}` in {path}")
                break
    return failures


def main() -> int:
    parser = ArgumentParser(description="Incremental compliance checker")
    parser.add_argument("--max-diff-lines", type=int, default=None, help="总修改行数上限 (新增+删除)")
    parser.add_argument("--max-new-files", type=int, default=None, help="新增文件数量上限")
    parser.add_argument("--allow-new-file", action="append", default=[], help="额外允许的新增文件相对路径")
    parser.add_argument("--check-placeholders", action="store_true", help="检测 TODO/FIXME/mock 等占位符")
    args = parser.parse_args()

    context = compute_diff_entries()
    entries = context.entries
    failures: List[str] = []

    for entry in entries:
        if entry.status and entry.status[0] not in {"A", "M", "R"}:
            continue

        normalized = entry.path.strip()
        if entry.status.startswith("A") or entry.status.startswith("R"):
            if _violates_suffix(normalized):
                failures.append(f"Forbidden suffix detected in new file: {normalized}")

        if _violates_mock(normalized):
            failures.append(f"Mock artefact detected: {normalized}")

    if args.max_new_files is not None:
        new_files = [entry.path for entry in entries if entry.status.startswith("A")]
        allow = _load_allowed_paths(args.allow_new_file)
        for new_path in new_files:
            if str(new_path) not in allow:
                failures.append(f"New file not in allow list: {new_path}")
        if len(new_files) > args.max_new_files:
            failures.append(f"Too many new files: {len(new_files)} > {args.max_new_files}")

    if args.max_diff_lines is not None:
        added, deleted = _compute_diff_stats(context.range_args)
        total = added + deleted
        if total > args.max_diff_lines:
            failures.append(f"Diff size {total} lines exceeds limit {args.max_diff_lines}")

    if args.check_placeholders:
        failures.extend(
            _scan_placeholders(
                entry
                for entry in entries
                if entry.status and entry.status[0] in {"A", "M", "R"}
            )
        )

    if failures:
        print("Incremental integrity check failed:\n" + "\n".join(f"  - {msg}" for msg in failures), file=sys.stderr)
        return 1

    print("check_incremental: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
