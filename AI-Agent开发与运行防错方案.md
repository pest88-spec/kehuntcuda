# AI Agent 开发与运行防错方案（工业级增强版）

**文档版本**：v2.0  
**适用项目**：Keyhunt-CUDA 及其配套项目  
**强制执行等级**：P0（所有规则为必须遵守，违反任何一条导致工作回滚）

---

## 0. 适用范围与执行优先级

### 0.1 适用对象
本方案约束以下所有参与者（人类与AI）：
- **Developer Agent**：负责代码编写、修改、重构
- **Reviewer Agent**：负责代码审查、质量把关、安全审计
- **Executor Agent**：负责构建、测试、基准测试、静态分析
- **Fixer Agent**：负责根据失败日志进行缺陷修复
- **Human Developer**：人类开发者在审查AI产出时必须参照本方案

### 0.2 强制执行层级
本方案分为三个执行层级，所有层级均为强制性：

| 层级 | 名称 | 违反后果 | 检测方式 |
|------|------|---------|---------|
| L1 | 铁律层 | 立即回滚，重新开始 | 自动化CI门禁 |
| L2 | 工程层 | 标记警告，人工复核 | 自动化扫描+人工审查 |
| L3 | 质量层 | 记录偏差，定期改进 | 夜间批处理分析 |

---

## 1. 术语与核心概念

### 1.1 基础术语

**Baseline（基线版本）**  
项目核心参考版本，包含以下要素：
- 完整的源代码快照
- 快照的SHA256哈希值（用于完整性验证）
- 性能基准数据（GPU throughput、延迟、内存占用）
- 测试覆盖率报告
- 所有依赖库的版本锁定清单

**Provenance Header（溯源头部）**  
每个源文件开头的标准化注释块，记录：
```cpp
/**
 * @file ec_operations.cu
 * @origin https://github.com/bitcoin-core/secp256k1
 * @origin_path src/ecmult_impl.h
 * @origin_commit a1b2c3d4e5f6
 * @origin_license MIT
 * @modified_by keyhunt-cuda-integration
 * @modifications "Adapted for CUDA kernel, optimized memory layout"
 * @fusion_date 2025-09-30
 * @spdx_license_identifier MIT
 */
```

**Incremental Edit（增量编辑）**  
对现有文件进行的最小化修改，必须满足：
- 可以用git diff清晰表达
- 修改行数占文件总行数≤80%
- 不改变文件的核心职责
- 保留原有函数签名（除非有充分理由）

**SoT（Source of Truth）**  
项目依赖的权威参考实现，分为：
- **SoT-CRYPTO**：密码学算法正确性参考（libsecp256k1、OpenSSL）
- **SoT-PERF**：性能基准参考（BitCrack、VanitySearch）
- **SoT-BUILD**：构建系统参考（CMake官方文档、CUDA Toolkit文档）
- **SoT-TEST**：测试框架参考（Google Test、CUDA Sample Tests）

### 1.2 角色与职责矩阵

| 角色 | 主要职责 | 输入 | 输出 | 审计要求 |
|------|---------|------|------|---------|
| Developer | 编写增量修改 | 需求说明、现有代码 | Patch文件、测试用例 | 必须通过Reviewer审查 |
| Reviewer | 质量把关 | Patch、测试报告 | 审查报告、改进建议 | 必须填写审查检查表 |
| Executor | 执行验证 | Patch、构建脚本 | CI日志、性能报告 | 所有日志归档90天 |
| Fixer | 缺陷修复 | 失败日志、错误堆栈 | 修复Patch | 必须附带根因分析 |

---

## 2. 铁律层约束（L1：自动化强制执行）

### 2.1 ZERO-NEW-FILES 原则

**规则**：禁止创建任何未经预先批准的新文件。

**实施细节**：
1. 项目维护一份白名单文件 `docs/allowed_files.txt`，记录所有允许存在的文件路径
2. CI在每次提交时执行：
   ```bash
   git diff --name-status origin/main | grep "^A" | while read status file; do
       if ! grep -q "^${file}$" docs/allowed_files.txt; then
           echo "ERROR: Unauthorized new file detected: ${file}"
           exit 1
       fi
   done
   ```
3. 新增必要文件的流程：
   - 在设计文档中声明文件路径、用途、预期行数
   - 通过人工审查后将路径添加到白名单
   - 白名单变更需要两名人类审查者批准

**禁止文件类型（黑名单）**：
```
# 脚本文件（除非在tools/目录且已预先批准）
*.sh (除外: tools/sync_reference_sources.sh, scripts/trace_snapshot.sh)
*.bat
*.ps1
*.py (除外: scripts/detect_truncation.py, tools/code_metrics.py)
*.js
*.rb
*.pl

# 临时文件标识
*_temp.*
*_copy.*
*_backup.*
*_new.*
*_old.*
*.tmp
*.bak

# 配置生成器（除非使用CMake官方模板）
config.h.in (除外: cmake/config.h.in - 必须基于CMake模板)
*.in (除非在cmake/目录)

# 不明来源的二进制或数据文件
*.bin (除非在tests/data/且有SHA256记录)
*.dat (除非在benchmark/baseline/且有溯源)
```

**CI实现**（伪代码）：
```yaml
# .github/workflows/gate-files.yml
name: File Creation Gate
on: [pull_request]
jobs:
  check-new-files:
    runs-on: ubuntu-latest
    steps:
      - name: Check unauthorized files
        run: |
          ./ci/check_new_files.sh || exit 1
      - name: Check blacklist patterns
        run: |
          git diff --name-only origin/main | grep -E '\.(sh|bat|ps1|py|js)$' | while read f; do
            if ! grep -q "^${f}$" docs/allowed_files.txt; then
              echo "BLOCKED: Script file ${f} not in whitelist"
              exit 1
            fi
          done
```

---

### 2.2 NO-PLACEHOLDERS 原则

**规则**：禁止任何形式的占位实现、待办标记、虚拟函数体。

**禁止模式库**（完整版）：
```python
# ci/forbidden_patterns.py
FORBIDDEN_PATTERNS = {
    # 待办标记
    r'//\s*(TODO|FIXME|HACK|XXX|NOTE:\s*implement)',
    r'#\s*(TODO|FIXME)',
    r'/\*\s*(TODO|FIXME)',
    
    # 占位标识
    r'\b(placeholder|temporary|temp_impl|stub)\b',
    r'\btemp_[a-zA-Z_]+\b',  # temp_function, temp_variable
    
    # 简化标识
    r'\b(simple|basic|simplified|dummy|mock|fake)\b',
    r'\bsimple_[a-zA-Z_]+\(',  # simple_version(), simple_impl()
    
    # 空实现
    r'{\s*//\s*not implemented',
    r'return\s+0;\s*//\s*(placeholder|todo|fixme)',
    r'return\s+nullptr;\s*//\s*not implemented',
    r'pass\s*#\s*TODO',  # Python
    
    # 注释掉的代码块（超过3行）
    r'(/\*([^*]|\*[^/]){100,}\*/)',  # 100+ chars of commented code
    r'(//.*\n){4,}',  # 4+ consecutive comment lines (疑似注释代码)
    
    # 测试用的临时main
    r'int\s+main.*--test',
    r'if\s*\(__name__\s*==\s*["\']__main__["\']\):\s*#\s*temp',
    
    # 占位返回值
    r'return\s+(-1|999|0xDEADBEEF);\s*//',  # Magic number placeholders
}

FORBIDDEN_FUNCTION_NAMES = [
    r'.*_temp\(',
    r'.*_mock\(',
    r'.*_dummy\(',
    r'.*_stub\(',
    r'test_placeholder\(',
    r'quick_hack\(',
]
```

**扫描工具**：
```bash
#!/bin/bash
# ci/scan_placeholders.sh

ERRORS=0

# 扫描禁止模式
for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
    if grep -rn --include="*.cpp" --include="*.cu" --include="*.h" -E "$pattern" src/; then
        echo "ERROR: Found forbidden pattern: $pattern"
        ERRORS=$((ERRORS + 1))
    fi
done

# 扫描空函数体（除了显式声明为deleted的）
find src/ -name "*.cpp" -o -name "*.cu" | while read file; do
    if grep -Pzo '(?s)\w+\s+\w+\([^)]*\)\s*{\s*}' "$file" | grep -v "= delete"; then
        echo "ERROR: Empty function body in $file"
        ERRORS=$((ERRORS + 1))
    fi
done

exit $ERRORS
```

**违反示例与修复对照**：

| 违规代码 | 问题 | 正确做法 |
|---------|------|---------|
| `void compute() { /* TODO */ }` | 空实现+TODO | 要么实现完整功能，要么删除函数 |
| `int basic_version() { return 0; }` | "basic"标识 | 重命名为描述性名称或删除 |
| `result = mock_gpu_call();` | "mock"标识 | 调用真实GPU函数或使用条件编译 |
| `#if 0 ... old code ... #endif` | 注释代码超过10行 | 删除旧代码，依赖git历史 |

---

### 2.3 INCREMENTAL-EDIT-ONLY 原则

**规则**：所有代码修改必须以增量形式呈现，单文件修改幅度受严格限制。

**修改幅度阈值**：

| 文件类型 | 允许修改行数占比 | 允许新增行数 | 超出后处理 |
|---------|-----------------|-------------|-----------|
| 核心算法（*.cu, ec_*.cpp） | ≤30% | ≤100行 | 强制人工复核+设计文档 |
| 接口文件（*_api.h） | ≤20% | ≤50行 | 强制人工复核 |
| 工具函数（util_*.cpp） | ≤50% | ≤150行 | Reviewer审查 |
| 测试文件（test_*.cpp） | ≤80% | 不限 | 自动通过（测试除外） |
| 构建脚本（CMakeLists.txt） | ≤40% | ≤80行 | 强制人工复核 |

**CI检测脚本**：
```bash
#!/bin/bash
# ci/check_diff_size.sh

git diff --numstat origin/main | while read added deleted file; do
    # 跳过测试文件
    if [[ "$file" =~ ^tests/ ]]; then
        continue
    fi
    
    # 计算原文件行数
    if [ -f "$file" ]; then
        total=$(wc -l < "$file")
        changed=$((added + deleted))
        ratio=$((changed * 100 / total))
        
        # 判断文件类型
        if [[ "$file" =~ \.(cu|cpp)$ ]] && [[ "$file" =~ (ec_|scalar_|field_) ]]; then
            threshold=30
        elif [[ "$file" =~ _api\.h$ ]]; then
            threshold=20
        elif [[ "$file" =~ CMakeLists\.txt$ ]]; then
            threshold=40
        else
            threshold=50
        fi
        
        if [ $ratio -gt $threshold ]; then
            echo "WARNING: $file changed ${ratio}% (threshold: ${threshold}%)"
            echo "  Added: $added, Deleted: $deleted, Total: $total"
            echo "  This file requires MANDATORY human review."
            echo "$file" >> /tmp/large_diff_files.txt
        fi
    fi
done

# 如果有超大修改文件，标记PR
if [ -f /tmp/large_diff_files.txt ]; then
    echo "::set-output name=large_diffs::true"
    cat /tmp/large_diff_files.txt
fi
```

**Patch提交规范**：
```bash
# 正确的提交方式（增量Patch）
git diff HEAD^ HEAD -- src/ec_operations.cu > patches/ec_ops_optimization.patch

# Patch文件必须包含的元信息
cat > patches/ec_ops_optimization.patch.meta << EOF
{
  "target_file": "src/ec_operations.cu",
  "change_type": "optimization",
  "lines_changed": 45,
  "total_lines": 320,
  "change_ratio": 14.0,
  "reviewer_required": false,
  "test_coverage_delta": +2.5,
  "performance_impact": "Expected +8% throughput",
  "sot_references": ["libsecp256k1:ecmult_impl.h:L234-L267"],
  "approved_by": null
}
EOF
```

---

### 2.4 REUSE-FIRST 原则

**规则**：必须按优先级顺序检查复用可能性，不可跳级。

**强制复用检查流程**（5级瀑布）：

```
[开始编写代码]
    ↓
[第1级] 当前项目已有实现？
    ├─ 是 → 直接复用（调整参数/配置）
    └─ 否 → 进入第2级
        ↓
[第2级] SoT参考仓库有直接可用代码？
    ├─ 是 → 复制并添加Provenance Header
    └─ 否 → 进入第3级
        ↓
[第3级] SoT参考仓库有可改造代码（需修改≤30%）？
    ├─ 是 → 复制+增量修改+标注SoT引用
    └─ 否 → 进入第4级
        ↓
[第4级] 标准库/CUDA SDK有原生支持？
    ├─ 是 → 使用标准API
    └─ 否 → 进入第5级
        ↓
[第5级] 确认无法复用？
    ├─ 是 → 编写最小实现（需设计文档+人工审批）
    └─ 否 → 返回第1级重新检查
```

**强制文档化要求**：
每次新增函数必须在注释中填写复用检查记录：

```cpp
/**
 * @brief 批量标量乘法（GPU优化版本）
 * 
 * @reuse_check_L1 当前项目: src/ec_operations.cu::scalar_mul_single (不支持批量)
 * @reuse_check_L2 libsecp256k1: src/ecmult_impl.h::secp256k1_ecmult (CPU实现)
 * @reuse_check_L3 BitCrack: src/KeySearchDevice.cu::generate_points (需大幅改造)
 * @reuse_check_L4 CUDA SDK: 无直接API
 * @reuse_check_L5 新增原因: 需要GPU批量处理+链式逆元优化，无现成实现
 * 
 * @sot_ref SOT-CRYPTO: libsecp256k1/ecmult_impl.h L456-L678 (算法正确性)
 * @sot_ref SOT-PERF: BitCrack/KeySearchDevice.cu L234-L289 (批量模式)
 * 
 * @param d_scalars 设备端标量数组
 * @param d_points 设备端输出点数组
 * @param count 批量大小
 */
__global__ void batch_scalar_mul(const uint256_t* d_scalars, ec_point_t* d_points, size_t count);
```

**CI自动检查**：
```bash
#!/bin/bash
# ci/verify_reuse_check.sh

# 检查所有新增函数是否有复用检查记录
git diff origin/main --unified=0 -- '*.cpp' '*.cu' '*.h' | \
grep -E '^\+.*\b(void|int|bool|__global__|__device__).*\(' | \
while read line; do
    func_name=$(echo "$line" | sed 's/^+//' | grep -oP '\b\w+\s*\(')
    if ! git diff origin/main | grep -B20 "$func_name" | grep -q "@reuse_check"; then
        echo "ERROR: Function $func_name missing @reuse_check documentation"
        exit 1
    fi
done
```

---

## 3. 引用与溯源控制（L1+L2）

### 3.1 参考源同步机制

**自动同步工具**：`tools/sync_reference_sources.sh`

```bash
#!/bin/bash
# tools/sync_reference_sources.sh

set -e

ACTION=${1:-check}  # check | apply | verify

# 参考源定义（从配置文件读取）
source config/reference_repos.conf

sync_reference() {
    local repo_name=$1
    local repo_url=$2
    local target_commit=$3
    local snapshot_dir="snapshots/${repo_name}"
    
    echo "Syncing ${repo_name}..."
    
    if [ ! -d "$snapshot_dir" ]; then
        git clone "$repo_url" "$snapshot_dir"
    fi
    
    cd "$snapshot_dir"
    git fetch --all
    git checkout "$target_commit"
    
    # 计算快照SHA256
    find . -type f -not -path './.git/*' | sort | \
    xargs sha256sum | sha256sum | \
    awk '{print $1}' > ../snapshots/${repo_name}.sha256
    
    cd ../..
    
    echo "${repo_name}: $(cat snapshots/${repo_name}.sha256)"
}

# 主流程
case $ACTION in
    check)
        echo "Checking reference source integrity..."
        verify_all_snapshots
        ;;
    apply)
        echo "Syncing all reference sources..."
        sync_reference "libsecp256k1" \
            "https://github.com/bitcoin-core/secp256k1.git" \
            "v0.4.0"
        sync_reference "BitCrack" \
            "https://github.com/brichard19/BitCrack.git" \
            "main"
        update_fusion_report
        ;;
    verify)
        echo "Verifying snapshot integrity..."
        for snapshot in snapshots/*.sha256; do
            repo=$(basename "$snapshot" .sha256)
            stored_hash=$(cat "$snapshot")
            computed_hash=$(compute_snapshot_hash "snapshots/$repo")
            if [ "$stored_hash" != "$computed_hash" ]; then
                echo "ERROR: Snapshot integrity check failed for $repo"
                exit 1
            fi
        done
        ;;
esac
```

**融合报告模板**：`docs/source_fusion_report.md`

```markdown
# 源代码融合追溯报告

## 基线版本信息
- **项目版本**: v1.2.3
- **报告生成时间**: 2025-09-30 14:32:15 UTC
- **CI构建ID**: #12345

## 参考源快照清单

| 仓库名称 | 版本/Commit | 快照SHA256 | 同步时间 | 用途 |
|---------|------------|-----------|---------|------|
| libsecp256k1 | v0.4.0 / a1b2c3d | e4f5a6b7... | 2025-09-28 | 密码学算法参考 |
| BitCrack | main / x7y8z9 | c1d2e3f4... | 2025-09-29 | 性能优化参考 |
| CUDA Samples | 12.0 / m9n8o7 | g5h6i7j8... | 2025-09-27 | CUDA最佳实践 |

## 文件溯源映射

| 目标文件 | 源仓库 | 源路径 | 修改幅度 | 许可证兼容性 |
|---------|--------|--------|---------|-------------|
| src/ec_operations.cu | libsecp256k1 | src/ecmult_impl.h | 35% | MIT → MIT ✓ |
| src/field_ops.cu | libsecp256k1 | src/field_impl.h | 20% | MIT → MIT ✓ |
| src/batch_kernel.cu | BitCrack | KeySearchDevice.cu | 60% | MIT → MIT ✓ |

## 许可证合规性检查

✓ 所有源文件已添加SPDX标识符  
✓ 所有修改已在Provenance Header中记录  
✓ 无GPL污染风险（所有依赖为MIT/BSD/Apache 2.0）  

## 审计追踪
- 最后审计人: @reviewer-bot
- 审计时间: 2025-09-30 10:00:00 UTC
- 下次审计: 2025-10-07 10:00:00 UTC (每周自动)
```

---

### 3.2 Provenance Header 强制检查

**CI检查脚本**：
```bash
#!/bin/bash
# ci/check_provenance.sh

MISSING_PROVENANCE=0

# 检查所有源文件是否有Provenance Header
find src/ -name "*.cpp" -o -name "*.cu" -o -name "*.h" | while read file; do
    # 检查是否有@origin标记
    if ! head -n 20 "$file" | grep -q "@origin"; then
        # 检查是否是新创建的文件（需要溯源）
        if git log --diff-filter=A --pretty=format: --name-only | grep -q "$file"; then
            echo "ERROR: $file missing Provenance Header"
            MISSING_PROVENANCE=$((MISSING_PROVENANCE + 1))
        fi
    fi
    
    # 验证SPDX标识符
    if ! head -n 30 "$file" | grep -q "SPDX-License-Identifier:"; then
        echo "ERROR: $file missing SPDX-License-Identifier"
        MISSING_PROVENANCE=$((MISSING_PROVENANCE + 1))
    fi
done

exit $MISSING_PROVENANCE
```

**追溯工具**：`scripts/trace_snapshot.sh`

```bash
#!/bin/bash
# scripts/trace_snapshot.sh - 追溯文件的所有上游来源

FILE=$1

if [ -z "$FILE" ]; then
    echo "Usage: $0 <file_path>"
    exit 1
fi

echo "Tracing provenance for: $FILE"
echo "========================================"

# 从Provenance Header提取信息
ORIGIN=$(grep "@origin " "$FILE" | sed 's/.*@origin //')
ORIGIN_PATH=$(grep "@origin_path " "$FILE" | sed 's/.*@origin_path //')
ORIGIN_COMMIT=$(grep "@origin_commit " "$FILE" | sed 's/.*@origin_commit //')

echo "Origin Repository: $ORIGIN"
echo "Origin Path: $ORIGIN_PATH"
echo "Origin Commit: $ORIGIN_COMMIT"
echo ""

# 在本地快照中查找源文件
REPO_NAME=$(echo "$ORIGIN" | awk -F'/' '{print $NF}' | sed 's/\.git//')
SNAPSHOT_PATH="snapshots/${REPO_NAME}/${ORIGIN_PATH}"

if [ -f "$SNAPSHOT_PATH" ]; then
    echo "Found in local snapshot: $SNAPSHOT_PATH"
    
    # 计算差异
    echo ""
    echo "Modifications from original:"
    echo "----------------------------"
    diff -u "$SNAPSHOT_PATH" "$FILE" | head -n 50
    
    # 统计修改幅度
    TOTAL_LINES=$(wc -l < "$SNAPSHOT_PATH")
    DIFF_LINES=$(diff -u "$SNAPSHOT_PATH" "$FILE" | grep -E '^\+' | grep -v '^\+++' | wc -l)
    RATIO=$((DIFF_LINES * 100 / TOTAL_LINES))
    
    echo ""
    echo "Modification Statistics:"
    echo "  Total lines: $TOTAL_LINES"
    echo "  Changed lines: $DIFF_LINES"
    echo "  Change ratio: ${RATIO}%"
else
    echo "WARNING: Original file not found in snapshots"
    echo "  Expected: $SNAPSHOT_PATH"
fi
```

---

## 4. 代码质量与反退化机制（L1+L2）

### 4.1 CUDA错误处理守护

**强制宏包装**：所有CUDA API调用必须使用以下宏：

```cpp
// include/cuda_utils.h

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d - %s (code: %d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err), err); \
        fprintf(stderr, "  Failed call: %s\n", #call); \
        abort(); \
    } \
} while(0)

#define CUDA_CHECK_RETURN(call, retval) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d - %s (code: %d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err), err); \
        return retval; \
    } \
} while(0)

// 内核启动专用宏
#define CUDA_KERNEL_CHECK() \
do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA KERNEL ERROR] %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        abort(); \
    } \
    CUDA_CHECK(cudaDeviceSynchronize()); \
} while(0)
```

**静态检查器**：
```bash
#!/bin/bash
# ci/check_cuda_errors.sh

VIOLATIONS=0

# 查找所有未使用CUDA_CHECK包装的API调用
CUDA_APIS=(
    "cudaMalloc" "cudaFree" "cudaMemcpy" "cudaMemset"
    "cudaGetDeviceCount" "cudaSetDevice" "cudaDeviceSynchronize"
    "cudaStreamCreate" "cudaStreamDestroy" "cudaStreamSynchronize"
    "cudaEventCreate" "cudaEventDestroy" "cudaEventRecord"
)

for api in "${CUDA_APIS[@]}"; do
    # 查找裸调用（不在CUDA_CHECK内）
    if grep -rn --include="*.cpp" --include="*.cu" "\b${api}\s*(" src/ | \
       grep -v "CUDA_CHECK\|CUDA_CHECK_RETURN" | \
       grep -v "^\s*//" | grep -v "^\s*\*"; then
        echo "ERROR: Found unwrapped ${api} calls"
        VIOLATIONS=$((VIOLATIONS + 1))
    fi
done

# 检查内核启动后是否有错误检查
find src/ -name "*.cu" | while read file; do
    # 查找内核启动（<<<...>>>）
    grep -n "<<<.*>>>" "$file" | while read line_num kernel_launch; do
        line_num=$(echo "$line_num" | cut -d: -f1)
        next_lines=$(sed -n "$((line_num+1)),$((line_num+5))p" "$file")
        
        if ! echo "$next_lines" | grep -q "CUDA_KERNEL_CHECK\|cudaGetLastError"; then
            echo "ERROR: $file:$line_num - Kernel launch without error check"
            VIOLATIONS=$((VIOLATIONS + 1))
        fi
    done
done

exit $VIOLATIONS
```

---

### 4.2 性能反退化监控

**基准测试框架**：`benchmark/bench_gpu.sh`

```bash
#!/bin/bash
# benchmark/bench_gpu.sh

set -e

OUTPUT_DIR="benchmark/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 测试配置
BATCH_SIZES=(1024 4096 8192 16384 32768)
SM_ARCHITECTURES=(75 80 86 89)  # Turing, Ampere, Ada Lovelace
ITERATIONS=10

run_benchmark() {
    local batch_size=$1
    local sm_arch=$2
    local output_file="$OUTPUT_DIR/batch${batch_size}_sm${sm_arch}.json"
    
    echo "Running: batch_size=$batch_size, sm=$sm_arch"
    
    ./build/keyhunt_bench \
        --batch-size "$batch_size" \
        --iterations "$ITERATIONS" \
        --sm-arch "$sm_arch" \
        --output-json "$output_file"
}

# 执行所有配置组合
for batch in "${BATCH_SIZES[@]}"; do
    for sm in "${SM_ARCHITECTURES[@]}"; do
        run_benchmark "$batch" "$sm"
    done
done

# 生成综合报告
python3 tools/generate_perf_report.py \
    --input-dir "$OUTPUT_DIR" \
    --output-html "$OUTPUT_DIR/report.html" \
    --compare-baseline "benchmark/baseline/latest.json"

# 检查性能退化
BASELINE_THROUGHPUT=$(jq '.avg_throughput' benchmark/baseline/latest.json)
CURRENT_THROUGHPUT=$(jq '.avg_throughput' "$OUTPUT_DIR"/batch8192_sm80.json)

THRESHOLD=0.95  # 95%阈值
RATIO=$(echo "scale=2; $CURRENT_THROUGHPUT / $BASELINE_THROUGHPUT" | bc)

if (( $(echo "$RATIO < $THRESHOLD" | bc -l) )); then
    echo "ERROR: Performance regression detected!"
    echo "  Baseline: ${BASELINE_THROUGHPUT} keys/s"
    echo "  Current:  ${CURRENT_THROUGHPUT} keys/s"
    echo "  Ratio:    ${RATIO} (threshold: ${THRESHOLD})"
    exit 1
fi

echo "Performance check passed: ${RATIO}x of baseline"
```

**性能趋势监控**：`tools/plot_perf_trend.py`

```python
#!/usr/bin/env python3
# tools/plot_perf_trend.py

import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def load_historical_data(results_dir):
    """加载历史性能数据"""
    data = []
    for result_file in sorted(Path(results_dir).glob("*/batch8192_sm80.json")):
        with open(result_file) as f:
            entry = json.load(f)
            timestamp = datetime.strptime(
                result_file.parent.name, "%Y%m%d_%H%M%S"
            )
            data.append({
                "timestamp": timestamp,
                "throughput": entry["avg_throughput"],
                "latency": entry["avg_latency_ms"],
                "commit": entry.get("git_commit", "unknown")
            })
    return data

def plot_trend(data, output_path):
    """绘制性能趋势图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    timestamps = [d["timestamp"] for d in data]
    throughputs = [d["throughput"] for d in data]
    latencies = [d["latency"] for d in data]
    
    # 吞吐量趋势
    ax1.plot(timestamps, throughputs, marker='o', label='Throughput')
    ax1.axhline(y=throughputs[0] * 0.95, color='r', linestyle='--', 
                label='95% Baseline Threshold')
    ax1.set_ylabel('Throughput (keys/s)')
    ax1.set_title('GPU Performance Trend')
    ax1.legend()
    ax1.grid(True)
    
    # 延迟趋势
    ax2.plot(timestamps, latencies, marker='s', color='orange', label='Latency')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Latency (ms)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Trend plot saved to {output_path}")

if __name__ == "__main__":
    data = load_historical_data("benchmark/results")
    plot_trend(data, "benchmark/performance_trend.png")
```

---

### 4.3 密码学正确性验证

**CPU-GPU一致性测试**：

```cpp
// tests/validation/test_crypto_consistency.cpp

#include <gtest/gtest.h>
#include "ec_operations.h"
#include "ec_operations_gpu.cuh"
#include <secp256k1.h>  // libsecp256k1作为baseline

class CryptoConsistencyTest : public ::testing::Test {
protected:
    secp256k1_context* ctx_baseline;
    
    void SetUp() override {
        ctx_baseline = secp256k1_context_create(
            SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY
        );
    }
    
    void TearDown() override {
        secp256k1_context_destroy(ctx_baseline);
    }
};

TEST_F(CryptoConsistencyTest, ScalarMultiplication_MatchesLibsecp256k1) {
    constexpr size_t TEST_COUNT = 1000;
    
    // 生成随机标量
    std::vector<uint8_t> scalars(TEST_COUNT * 32);
    ASSERT_TRUE(secp256k1_rand256_test(scalars.data()));
    
    // Baseline: libsecp256k1 CPU计算
    std::vector<secp256k1_pubkey> cpu_results(TEST_COUNT);
    for (size_t i = 0; i < TEST_COUNT; i++) {
        ASSERT_TRUE(secp256k1_ec_pubkey_create(
            ctx_baseline,
            &cpu_results[i],
            &scalars[i * 32]
        ));
    }
    
    // GPU计算
    std::vector<ec_point_t> gpu_results(TEST_COUNT);
    batch_scalar_mul_gpu(
        reinterpret_cast<const scalar256_t*>(scalars.data()),
        gpu_results.data(),
        TEST_COUNT
    );
    
    // 逐个比对
    for (size_t i = 0; i < TEST_COUNT; i++) {
        uint8_t cpu_serialized[65];
        size_t cpu_len = 65;
        ASSERT_TRUE(secp256k1_ec_pubkey_serialize(
            ctx_baseline,
            cpu_serialized,
            &cpu_len,
            &cpu_results[i],
            SECP256K1_EC_UNCOMPRESSED
        ));
        
        uint8_t gpu_serialized[65];
        serialize_ec_point(&gpu_results[i], gpu_serialized);
        
        EXPECT_EQ(0, memcmp(cpu_serialized, gpu_serialized, 65))
            << "Mismatch at index " << i;
    }
}

TEST_F(CryptoConsistencyTest, EdgeCases_ZeroAndOne) {
    // k=0应输出无穷远点
    scalar256_t zero = {0};
    ec_point_t point_zero;
    scalar_mul_gpu(&zero, &point_zero);
    EXPECT_TRUE(is_infinity(&point_zero));
    
    // k=1应输出生成元G
    scalar256_t one = {1};
    ec_point_t point_one;
    scalar_mul_gpu(&one, &point_one);
    
    // 与secp256k1常量对比
    EXPECT_EQ(0, memcmp(&point_one.x, SECP256K1_G_X, 32));
    EXPECT_EQ(0, memcmp(&point_one.y, SECP256K1_G_Y, 32));
}
```

---

## 5. 测试驱动开发与CI门禁（L1）

### 5.1 完整CI流水线

```yaml
# .github/workflows/ci-gate.yml
name: CI Gate - Full Verification

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

jobs:
  # 阶段1: 源代码溯源验证
  provenance-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Sync reference sources
        run: ./tools/sync_reference_sources.sh --apply
      - name: Verify provenance headers
        run: ./ci/check_provenance.sh
      - name: Check license compliance
        run: ./ci/check_licenses.sh

  # 阶段2: 静态分析
  static-analysis:
    runs-on: ubuntu-latest
    needs: provenance-check
    steps:
      - uses: actions/checkout@v3
      - name: Scan for placeholders
        run: ./ci/scan_placeholders.sh
      - name: Check CUDA error handling
        run: ./ci/check_cuda_errors.sh
      - name: Check diff size
        run: ./ci/check_diff_size.sh
      - name: Verify new files
        run: ./ci/check_new_files.sh

  # 阶段3: 编译与单元测试
  build-and-test:
    runs-on: ubuntu-latest
    needs: static-analysis
    steps:
      - uses: actions/checkout@v3
      - name: Build project
        run: |
          mkdir build && cd build
          cmake -DCMAKE_BUILD_TYPE=Release ..
          make -j$(nproc)
      - name: Run unit tests
        run: |
          cd build
          ctest --output-on-failure --timeout 300
      - name: Check test coverage
        run: |
          ./tools/check_coverage.sh
          # 要求覆盖率≥80%
          coverage=$(cat build/coverage.txt | grep 'lines' | awk '{print $2}' | tr -d '%')
          if [ "$coverage" -lt 80 ]; then
            echo "ERROR: Coverage $coverage% < 80%"
            exit 1
          fi

  # 阶段4: CUDA内存检查
  cuda-memcheck:
    runs-on: [self-hosted, gpu]
    needs: build-and-test
    steps:
      - uses: actions/checkout@v3
      - name: Build CUDA tests
        run: |
          mkdir build && cd build
          cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA_TESTS=ON ..
          make -j$(nproc)
      - name: Run cuda-memcheck
        run: |
          cd build
          cuda-memcheck --leak-check full ./tests/test_gpu_kernels
          # 任何内存错误都会导致失败

  # 阶段5: 密码学一致性验证
  crypto-validation:
    runs-on: [self-hosted, gpu]
    needs: cuda-memcheck
    steps:
      - uses: actions/checkout@v3
      - name: Build with validation tests
        run: |
          mkdir build && cd build
          cmake -DENABLE_VALIDATION_TESTS=ON ..
          make -j$(nproc)
      - name: Run CPU-GPU consistency tests
        run: |
          cd build
          ./tests/test_crypto_consistency --gtest_filter="*Consistency*"

  # 阶段6: 性能基准测试
  performance-benchmark:
    runs-on: [self-hosted, gpu]
    needs: crypto-validation
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmark
        run: |
          mkdir build && cd build
          cmake -DCMAKE_BUILD_TYPE=Release ..
          make -j$(nproc)
          ../benchmark/bench_gpu.sh
      - name: Check performance regression
        run: |
          python3 tools/check_perf_regression.py \
            --current build/benchmark/results/latest.json \
            --baseline benchmark/baseline/latest.json \
            --threshold 0.95

  # 阶段7: 最终门禁
  gate-summary:
    runs-on: ubuntu-latest
    needs: [provenance-check, static-analysis, build-and-test, 
            cuda-memcheck, crypto-validation, performance-benchmark]
    steps:
      - name: Generate gate report
        run: |
          echo "# CI Gate Summary" > gate_report.md
          echo "All checks passed ✓" >> gate_report.md
      - name: Post to PR
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: require('fs').readFileSync('gate_report.md', 'utf8')
            })
```

---

### 5.2 测试驱动开发工作流

**TDD强制流程**：

```
[收到新需求]
    ↓
[步骤1] 编写失败的测试用例
    ├─ 测试文件命名: tests/unit/test_<feature>.cpp
    ├─ 测试用例覆盖: 正常路径、边界条件、错误处理
    └─ 运行测试确认失败（红灯）
    ↓
[步骤2] 编写最小实现代码
    ├─ 仅实现让测试通过的代码
    ├─ 不添加任何额外功能
    └─ 运行测试确认通过（绿灯）
    ↓
[步骤3] 重构优化
    ├─ 消除重复代码
    ├─ 优化性能
    └─ 运行测试确认仍通过（保持绿灯）
    ↓
[步骤4] 提交代码
    ├─ 提交测试代码
    ├─ 提交实现代码
    └─ CI自动验证
```

**测试用例模板**：

```cpp
// tests/unit/test_batch_scalar_mul.cpp

#include <gtest/gtest.h>
#include "ec_operations_gpu.cuh"
#include "test_utils.h"

class BatchScalarMulTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化GPU设备
        CUDA_CHECK(cudaSetDevice(0));
    }
    
    void TearDown() override {
        // 清理
        CUDA_CHECK(cudaDeviceReset());
    }
};

// 正常路径测试
TEST_F(BatchScalarMulTest, NormalInput_ReturnsCorrectResults) {
    constexpr size_t BATCH_SIZE = 1024;
    
    // 准备输入
    std::vector<scalar256_t> scalars(BATCH_SIZE);
    generate_random_scalars(scalars.data(), BATCH_SIZE);
    
    // 执行GPU计算
    std::vector<ec_point_t> results(BATCH_SIZE);
    int ret = batch_scalar_mul_gpu(scalars.data(), results.data(), BATCH_SIZE);
    
    ASSERT_EQ(0, ret) << "GPU computation failed";
    
    // 验证每个结果
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        EXPECT_TRUE(is_on_curve(&results[i])) 
            << "Point " << i << " not on curve";
    }
}

// 边界条件测试
TEST_F(BatchScalarMulTest, ZeroScalar_ReturnsInfinity) {
    scalar256_t zero = {0};
    ec_point_t result;
    
    int ret = batch_scalar_mul_gpu(&zero, &result, 1);
    
    ASSERT_EQ(0, ret);
    EXPECT_TRUE(is_infinity(&result)) << "k=0 should give point at infinity";
}

TEST_F(BatchScalarMulTest, OneScalar_ReturnsGenerator) {
    scalar256_t one = {1};
    ec_point_t result;
    
    int ret = batch_scalar_mul_gpu(&one, &result, 1);
    
    ASSERT_EQ(0, ret);
    EXPECT_EQ(0, memcmp(&result.x, SECP256K1_G_X, 32));
    EXPECT_EQ(0, memcmp(&result.y, SECP256K1_G_Y, 32));
}

// 错误处理测试
TEST_F(BatchScalarMulTest, NullPointer_ReturnsError) {
    scalar256_t scalar = {1};
    
    int ret = batch_scalar_mul_gpu(nullptr, nullptr, 1);
    
    EXPECT_NE(0, ret) << "Should return error for null pointer";
}

TEST_F(BatchScalarMulTest, ZeroCount_ReturnsError) {
    scalar256_t scalar = {1};
    ec_point_t result;
    
    int ret = batch_scalar_mul_gpu(&scalar, &result, 0);
    
    EXPECT_NE(0, ret) << "Should return error for zero count";
}

// 性能测试
TEST_F(BatchScalarMulTest, Performance_MeetsThreshold) {
    constexpr size_t BATCH_SIZE = 8192;
    constexpr int ITERATIONS = 10;
    
    std::vector<scalar256_t> scalars(BATCH_SIZE);
    std::vector<ec_point_t> results(BATCH_SIZE);
    generate_random_scalars(scalars.data(), BATCH_SIZE);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < ITERATIONS; i++) {
        batch_scalar_mul_gpu(scalars.data(), results.data(), BATCH_SIZE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double throughput = (BATCH_SIZE * ITERATIONS) / (duration.count() / 1000.0);
    
    EXPECT_GE(throughput, 400000.0) 
        << "Throughput " << throughput << " keys/s < 400K keys/s threshold";
}
```

---

## 6. Nightly Routine与熔断机制（L2+L3）

### 6.1 夜间自动化流水线

```bash
#!/bin/bash
# ci/nightly_build.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/nightly/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=== Nightly Build Started: $TIMESTAMP ===" | tee "$LOG_DIR/summary.log"

# 步骤1: 完整清理构建
echo "[1/7] Clean build..." | tee -a "$LOG_DIR/summary.log"
rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_ALL_TESTS=ON .. \
    2>&1 | tee "$LOG_DIR/build.log"
make -j$(nproc) 2>&1 | tee -a "$LOG_DIR/build.log"
BUILD_STATUS=$?
cd ..

if [ $BUILD_STATUS -ne 0 ]; then
    echo "ERROR: Build failed" | tee -a "$LOG_DIR/summary.log"
    trigger_alert "nightly_build_failed" "$LOG_DIR/build.log"
    exit 1
fi

# 步骤2: 单元测试
echo "[2/7] Running unit tests..." | tee -a "$LOG_DIR/summary.log"
cd build
ctest --output-on-failure --timeout 600 \
    2>&1 | tee "$LOG_DIR/unit_tests.log"
UNIT_TEST_STATUS=$?
cd ..

# 步骤3: 随机Fuzz验证（2^20个输入）
echo "[3/7] Running validation tests..." | tee -a "$LOG_DIR/summary.log"
./scripts/run_validation.sh --fuzz-count 1048576 \
    2>&1 | tee "$LOG_DIR/validation.log"
VALIDATION_STATUS=$?

# 步骤4: 性能基准测试
echo "[4/7] Running performance benchmark..." | tee -a "$LOG_DIR/summary.log"
./benchmark/bench_gpu.sh --sm-list "75,80,86,89" \
    2>&1 | tee "$LOG_DIR/benchmark.log"
BENCH_STATUS=$?

# 步骤5: CUDA内存检查（随机内核）
echo "[5/7] Running CUDA memcheck..." | tee -a "$LOG_DIR/summary.log"
RANDOM_KERNEL=$(find build/tests -name "test_*" | shuf -n 1)
cuda-memcheck --leak-check full "$RANDOM_KERNEL" \
    2>&1 | tee "$LOG_DIR/memcheck.log"
MEMCHECK_STATUS=$?

# 步骤6: 生成性能趋势图
echo "[6/7] Updating performance trends..." | tee -a "$LOG_DIR/summary.log"
python3 tools/plot_perf_trend.py \
    --input-dir benchmark/results \
    --output-png "$LOG_DIR/perf_trend.png"

# 步骤7: 打包发布构建
echo "[7/7] Packaging release..." | tee -a "$LOG_DIR/summary.log"
./scripts/package_release.sh --output "releases/keyhunt_cuda_${TIMESTAMP}.tar.gz" \
    2>&1 | tee "$LOG_DIR/package.log"

# 汇总状态
TOTAL_FAILURES=0
for status in $BUILD_STATUS $UNIT_TEST_STATUS $VALIDATION_STATUS \
              $BENCH_STATUS $MEMCHECK_STATUS; do
    if [ $status -ne 0 ]; then
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
    fi
done

echo "" | tee -a "$LOG_DIR/summary.log"
echo "=== Nightly Build Summary ===" | tee -a "$LOG_DIR/summary.log"
echo "Build:      $(status_emoji $BUILD_STATUS)" | tee -a "$LOG_DIR/summary.log"
echo "Unit Tests: $(status_emoji $UNIT_TEST_STATUS)" | tee -a "$LOG_DIR/summary.log"
echo "Validation: $(status_emoji $VALIDATION_STATUS)" | tee -a "$LOG_DIR/summary.log"
echo "Benchmark:  $(status_emoji $BENCH_STATUS)" | tee -a "$LOG_DIR/summary.log"
echo "Memcheck:   $(status_emoji $MEMCHECK_STATUS)" | tee -a "$LOG_DIR/summary.log"
echo "Total Failures: $TOTAL_FAILURES" | tee -a "$LOG_DIR/summary.log"

# 熔断检查
check_circuit_breaker "$TOTAL_FAILURES"

exit $TOTAL_FAILURES
```

---

### 6.2 三级熔断机制

```bash
#!/bin/bash
# ci/circuit_breaker.sh

FAILURE_COUNT=$1
FAILURE_HISTORY_FILE=".ci/failure_history.txt"

# 记录失败
echo "$(date +%Y-%m-%d) $FAILURE_COUNT" >> "$FAILURE_HISTORY_FILE"

# 获取最近3天的失败记录
RECENT_FAILURES=$(tail -n 3 "$FAILURE_HISTORY_FILE" | awk '{sum+=$2} END {print sum}')

echo "Recent failures (last 3 nights): $RECENT_FAILURES"

# 熔断阈值
if [ "$RECENT_FAILURES" -ge 3 ]; then
    echo "=== CIRCUIT BREAKER TRIGGERED ===" | tee circuit_breaker.alert
    echo "Consecutive failures detected: $RECENT_FAILURES" | tee -a circuit_breaker.alert
    
    # 1. 锁定主分支
    echo "Locking main branch merges..." | tee -a circuit_breaker.alert
    gh api \
        --method PUT \
        /repos/:owner/:repo/branches/main/protection/required_pull_request_reviews \
        -f required_approving_review_count=2 \
        -F dismiss_stale_reviews=true \
        -F restrict_dismissals=true
    
    # 2. 创建紧急Issue
    gh issue create \
        --title "🚨 Circuit Breaker: Build Failures Detected" \
        --body "$(cat circuit_breaker.alert)" \
        --label "P0,circuit-breaker" \
        --assignee "@maintainers"
    
    # 3. 发送告警
    trigger_alert "circuit_breaker" circuit_breaker.alert
    
    # 4. 生成诊断报告
    generate_diagnostic_report > incident_$(date +%Y%m%d).md
    
    echo "Main branch is now LOCKED. Resolve the issue to unlock." | tee -a circuit_breaker.alert
    
    exit 2
elif [ "$RECENT_FAILURES" -ge 2 ]; then
    echo "WARNING: 2 failures in last 3 nights - one more triggers circuit breaker"
    trigger_alert "circuit_breaker_warning" "Approaching circuit breaker threshold"
fi

exit 0
```

---

## 7. Developer与Reviewer检查清单

### 7.1 Developer提交前检查表

```markdown
# Developer Pre-Commit Checklist

## 开发准备（在编写代码前）
- [ ] 已阅读并理解本次修改的需求文档
- [ ] 已运行 `tools/sync_reference_sources.sh --apply` 更新参考源
- [ ] 已检查5级复用优先级（见第2.4节）
- [ ] 已在设计文档中声明需要修改的文件列表
- [ ] 确认不需要创建任何新文件（如需要，已走白名单审批流程）

## 代码编写（TDD流程）
- [ ] 已编写测试用例并确认失败（红灯）
- [ ] 已编写最小实现代码
- [ ] 所有测试通过（绿灯）
- [ ] 已添加@reuse_check和@sot_ref注释
- [ ] 已添加Provenance Header（如修改了来自参考源的文件）
- [ ] 所有CUDA API调用使用CUDA_CHECK包装
- [ ] 无任何TODO/FIXME/placeholder标记

## 本地验证
- [ ] 编译通过（零警告）：`make CXXFLAGS="-Wall -Wextra -Werror"`
- [ ] 单元测试通过：`ctest --output-on-failure`
- [ ] 禁止模式扫描通过：`./ci/scan_placeholders.sh`
- [ ] CUDA错误检查通过：`./ci/check_cuda_errors.sh`
- [ ] Diff规模检查通过：`./ci/check_diff_size.sh`
- [ ] 代码覆盖率≥80%：`./tools/check_coverage.sh`

## 性能验证（如涉及性能关键路径）
- [ ] 运行基准测试：`./benchmark/bench_gpu.sh`
- [ ] 性能不低于baseline的95%
- [ ] 已更新性能趋势图

## 提交准备
- [ ] Commit message遵循约定格式（见下方）
- [ ] 已将Patch文件及其.meta元数据添加到patches/目录
- [ ] 已更新CHANGELOG.md
- [ ] 已填写本检查表并附在PR描述中

## Commit Message格式
```
<type>(<scope>): <subject>

<body>

Refs: #<issue_number>
Reviewed-by: @<reviewer>
Tested: <test_details>
Performance: <benchmark_result>
```

类型（type）：
- feat: 新功能
- fix: 缺陷修复
- perf: 性能优化
- refactor: 重构
- test: 测试相关
- docs: 文档更新
- build: 构建系统
```
```

---

### 7.2 Reviewer审查检查表

```markdown
# Reviewer Checklist

## 第一轮：形式审查
- [ ] PR描述包含Developer检查表，所有项已勾选
- [ ] Commit message格式正确
- [ ] 文件修改列表在预期范围内（无意外新文件）
- [ ] Diff规模在阈值内（核心算法≤30%，接口≤20%）

## 第二轮：溯源审查
- [ ] 所有修改文件保留了Provenance Header
- [ ] @reuse_check注释记录完整（5级检查）
- [ ] @sot_ref引用了正确的参考源章节
- [ ] 无重复实现已有算法（检查代码重复率）

## 第三轮：质量审查
- [ ] 无TODO/FIXME/placeholder标记
- [ ] 所有CUDA API调用使用CUDA_CHECK
- [ ] 所有函数有Doxygen格式注释
- [ ] 无magic number（使用命名常量）
- [ ] 错误处理完整（所有分支有错误处理）

## 第四轮：测试审查
- [ ] 新增功能有对应测试用例
- [ ] 测试覆盖正常路径、边界条件、错误处理
- [ ] 测试通过率100%
- [ ] 测试覆盖率≥80%（整体）

## 第五轮：性能审查（如相关）
- [ ] 性能基准测试已运行
- [ ] 性能满足≥95% baseline阈值
- [ ] 无明显性能退化
- [ ] 已更新性能趋势图

## 第六轮：安全审查
- [ ] 无内存泄漏（通过cuda-memcheck）
- [ ] 无越界访问
- [ ] 无未初始化变量
- [ ] 密码学操作正确性（通过CPU-GPU一致性测试）

## 审查决策
- [ ] APPROVE（通过）
- [ ] REQUEST_CHANGES（需修改）
- [ ] COMMENT（建议）

## 审查报告模板
```markdown
# Code Review Report

**PR**: #<pr_number>
**Reviewer**: @<username>
**Date**: YYYY-MM-DD

## Summary
[总体评价]

## Findings

### Critical Issues (Must Fix)
- [ ] Issue 1: [描述]
- [ ] Issue 2: [描述]

### Major Issues (Should Fix)
- [ ] Issue 1: [描述]

### Minor Issues (Nice to Have)
- Issue 1: [描述]

## Performance Impact
- Baseline: XXX keys/s
- Current: YYY keys/s
- Ratio: Z.ZZ (threshold: 0.95)

## Test Coverage
- Previous: XX.X%
- Current: YY.Y%
- Delta: +Z.Z%

## Decision
- [x] APPROVE
- [ ] REQUEST_CHANGES
- [ ] COMMENT

## Additional Comments
[其他意见]
```
```

---

## 8. 故障应急与事故处理

### 8.1 故障分级

| 级别 | 定义 | 响应时间 | 处理流程 |
|------|------|---------|---------|
| P0 | 数据损坏、截断、密码学错误 | 立即 | 触发熔断，创建hotfix分支 |
| P1 | 性能严重退化（<50% baseline） | 2小时 | 创建高优先级Issue |
| P2 | CI完全失败（所有测试红） | 8小时 | 分配给Fixer Agent |
| P3 | 单个测试失败或警告 | 24小时 | 常规修复流程 |

---

### 8.2 P0故障处理流程

```bash
#!/bin/bash
# ci/handle_p0_incident.sh

ISSUE_TYPE=$1  # truncation | crypto_error | data_corruption
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
INCIDENT_ID="INC_${TIMESTAMP}"
INCIDENT_DIR="docs/incidents/${INCIDENT_ID}"

mkdir -p "$INCIDENT_DIR"

echo "=== P0 INCIDENT DETECTED: $ISSUE_TYPE ===" | tee "$INCIDENT_DIR/incident_report.md"
echo "Incident ID: $INCIDENT_ID" | tee -a "$INCIDENT_DIR/incident_report.md"
echo "Timestamp: $(date)" | tee -a "$INCIDENT_DIR/incident_report.md"

# 1. 立即锁定主分支
echo "Step 1: Locking main branch..." | tee -a "$INCIDENT_DIR/incident_report.md"
gh api \
    --method PUT \
    /repos/:owner/:repo/branches/main/protection \
    -f required_status_checks='{}' \
    -f enforce_admins=true \
    -f required_pull_request_reviews='{required_approving_review_count:2}'

# 2. 创建hotfix分支
echo "Step 2: Creating hotfix branch..." | tee -a "$INCIDENT_DIR/incident_report.md"
HOTFIX_BRANCH="hotfix/${INCIDENT_ID}_${ISSUE_TYPE}"
git checkout -b "$HOTFIX_BRANCH"

# 3. 收集诊断信息
echo "Step 3: Collecting diagnostics..." | tee -a "$INCIDENT_DIR/incident_report.md"

# Git历史
git log --oneline -20 > "$INCIDENT_DIR/recent_commits.txt"

# 性能趋势
cp benchmark/performance_trend.png "$INCIDENT_DIR/"

# 最近的测试日志
cp -r logs/nightly/$(ls -t logs/nightly/ | head -n 1) "$INCIDENT_DIR/latest_logs/"

# 内存检查日志
if [ -f logs/memcheck/latest.log ]; then
    cp logs/memcheck/latest.log "$INCIDENT_DIR/"
fi

# 4. 根据故障类型启用防护措施
case $ISSUE_TYPE in
    truncation)
        echo "Enabling truncation protection..." | tee -a "$INCIDENT_DIR/incident_report.md"
        # 修改CMakeLists.txt启用运行时检查
        sed -i 's/#define ENABLE_TRUNCATION_GUARD 0/#define ENABLE_TRUNCATION_GUARD 1/' src/config.h
        ;;
    crypto_error)
        echo "Enabling crypto validation..." | tee -a "$INCIDENT_DIR/incident_report.md"
        # 启用CPU-GPU双路验证
        sed -i 's/#define ENABLE_DUAL_PATH_VALIDATION 0/#define ENABLE_DUAL_PATH_VALIDATION 1/' src/config.h
        ;;
    data_corruption)
        echo "Enabling data integrity checks..." | tee -a "$INCIDENT_DIR/incident_report.md"
        # 启用所有数据校验和
        sed -i 's/#define ENABLE_CHECKSUM 0/#define ENABLE_CHECKSUM 1/' src/config.h
        ;;
esac

# 5. 运行历史数据检查（检测污染范围）
echo "Step 4: Scanning historical results..." | tee -a "$INCIDENT_DIR/incident_report.md"
./scripts/detect_truncation.py --scan-dir results/archive \
    --output "$INCIDENT_DIR/pollution_report.json"

# 6. 创建紧急Issue
echo "Step 5: Creating emergency issue..." | tee -a "$INCIDENT_DIR/incident_report.md"
gh issue create \
    --title "🚨 P0 INCIDENT: ${ISSUE_TYPE}" \
    --body "$(cat $INCIDENT_DIR/incident_report.md)" \
    --label "P0,incident,${ISSUE_TYPE}" \
    --assignee "@security-team" \
    --milestone "Hotfix"

# 7. 发送告警
echo "Step 6: Triggering alerts..." | tee -a "$INCIDENT_DIR/incident_report.md"
trigger_alert "p0_incident" "$INCIDENT_DIR/incident_report.md"

# 8. 生成修复指引
cat > "$INCIDENT_DIR/fix_guidance.md" << EOF
# Hotfix Guidance for ${INCIDENT_ID}

## Issue Type
${ISSUE_TYPE}

## Root Cause Analysis
[待填写]

## Affected Versions
[待填写]

## Fix Strategy
1. [具体修复步骤]
2. [验证方法]
3. [回归测试]

## Rollback Plan
\`\`\`bash
git checkout main
git revert <commit_hash>
\`\`\`

## Verification Checklist
- [ ] 问题已修复
- [ ] 所有测试通过
- [ ] 性能无退化
- [ ] 历史数据已清理/标记
- [ ] 防护措施已就位
- [ ] 文档已更新

## Sign-off
- Fixed by: @
- Reviewed by: @
- Approved by: @
EOF

echo "" | tee -a "$INCIDENT_DIR/incident_report.md"
echo "Hotfix branch created: $HOTFIX_BRANCH" | tee -a "$INCIDENT_DIR/incident_report.md"
echo "Fix guidance: $INCIDENT_DIR/fix_guidance.md" | tee -a "$INCIDENT_DIR/incident_report.md"
echo "" | tee -a "$INCIDENT_DIR/incident_report.md"
echo "Main branch is LOCKED. Follow fix guidance to resolve." | tee -a "$INCIDENT_DIR/incident_report.md"
```

---

## 9. Prompt套件与AI Agent交互协议

### 9.1 标准Prompt模板

```markdown
# AI Agent Standard Prompt Template

You are working on the Keyhunt-CUDA project under strict industrial-grade constraints.

## MANDATORY RULES (铁律)

1. **ZERO-NEW-FILES**: Do NOT create any new files unless pre-approved in `docs/allowed_files.txt`
2. **NO-PLACEHOLDERS**: Do NOT use TODO, FIXME, mock_, dummy_, stub_, placeholder, or any temporary implementations
3. **INCREMENTAL-EDIT-ONLY**: All modifications must be expressed as minimal diffs (≤80% of file for core algorithms)
4. **REUSE-FIRST**: Before writing any code, complete the 5-level reuse check:
   - L1: Current project
   - L2: SoT reference repos (libsecp256k1, BitCrack)
   - L3: Modified SoT code (≤30% changes)
   - L4: Standard library / CUDA SDK
   - L5: New implementation (requires design doc + human approval)

## BEFORE YOU START

Answer these questions:
1. Have you checked all 5 levels of reuse priority?
2. Does this feature require creating new files? (If yes, STOP and request human approval)
3. Can this be achieved by modifying existing function parameters/configs only?
4. Have you read the Provenance Header of the file you're about to modify?

## CODE MODIFICATION REQUIREMENTS

- Use `CUDA_CHECK()` macro for ALL CUDA API calls
- Add `@reuse_check` comments documenting all 5 levels
- Add `@sot_ref` comments referencing specific source lines
- Preserve existing Provenance Headers
- Include complete Doxygen comments for all functions
- Write tests BEFORE implementation (TDD)

## OUTPUT FORMAT

Provide your changes as:
1. **Reuse Check Summary**: Which levels you checked and findings
2. **Design Rationale**: Why this approach is minimal
3. **Diff/Patch**: Exact changes in unified diff format
4. **Test Cases**: New or modified tests
5. **Performance Impact**: Expected change in throughput

## EXAMPLE OUTPUT

```
### Reuse Check Summary
- L1 (Current): src/ec_operations.cu::scalar_mul_single exists but doesn't support batching
- L2 (libsecp256k1): src/ecmult_impl.h provides algorithm reference (CPU only)
- L3 (BitCrack): KeySearchDevice.cu has batch mode but requires >60% modifications
- L4 (CUDA SDK): No direct API for batch scalar multiplication
- L5 (New): Justified - need GPU batch processing with shared memory optimization

### Design Rationale
Implementing batch_scalar_mul() as a new __global__ function because:
1. No existing function supports GPU batching
2. Will reuse field_mul() from existing field_ops.cu
3. Total new code: ~120 lines (within budget)
4. Expected performance: 450K keys/s (>400K threshold)

### Diff/Patch
```diff
--- a/src/ec_operations.cu
+++ b/src/ec_operations.cu
@@ -245,6 +245,18 @@
 
+/**
+ * @brief Batch scalar multiplication on GPU
+ * @reuse_check_L1 Current: scalar_mul_single (no batch support)
+ * @reuse_check_L2 libsecp256k1: ecmult_impl.h L456-L678
+ * @sot_ref SOT-CRYPTO: libsecp256k1/ecmult_impl.h
+ */
+__global__ void batch_scalar_mul(
+    const scalar256_t* d_scalars,
+    ec_point_t* d_points,
+    size_t count
+) {
+    // Implementation...
+}
```

### Test Cases
```cpp
TEST(BatchScalarMulTest, ReturnsCorrectResults) {
    // Test implementation...
}
```

### Performance Impact
- Estimated throughput: 450K keys/s
- Baseline: 420K keys/s
- Ratio: 1.07x (above 0.95 threshold ✓)
```

---

### 9.2 Fixer Agent专用Prompt

```markdown
# Fixer Agent Prompt - CI Failure Recovery

You are a Fixer Agent responding to CI failures. Your role is to analyze logs and propose minimal fixes.

## INPUT
- CI failure logs (provided below)
- Failed test cases
- Error stack traces

## CONSTRAINTS
- Do NOT rewrite entire files
- Do NOT introduce new dependencies
- Do NOT bypass tests
- Fix the root cause, not symptoms

## ANALYSIS PROCESS
1. **Classify Failure Type**:
   - Compilation error
   - Test failure
   - Performance regression
   - Memory error
   - Placeholder detection

2. **Root Cause Analysis**:
   - Quote the exact error message
   - Identify the failing code location
   - Explain why it failed

3. **Minimal Fix Strategy**:
   - What's the smallest change to fix this?
   - Can we adjust a parameter instead of code?
   - Is there an existing function we missed?

4. **Verification Plan**:
   - Which tests will confirm the fix?
   - What's the expected output?

## OUTPUT FORMAT

```markdown
# Fix Analysis Report

## Failure Classification
[Type]: Compilation Error / Test Failure / Performance Regression / Memory Error

## Root Cause
[Exact error message from logs]
[Code location: file:line]
[Explanation]

## Minimal Fix
```diff
[Provide exact patch]
```

## Rationale
[Why this is the minimal necessary change]

## Verification
- Run: `[command]`
- Expected: `[output]`

## Risk Assessment
- Low / Medium / High
- Affected modules: [list]
```

## EXAMPLE

Given this CI failure:
```
ERROR: test_batch_scalar_mul.cpp:45
  Expected: is_on_curve(&result)
    Actual: false
```

Your response should be:

```markdown
# Fix Analysis Report

## Failure Classification
Test Failure - Incorrect curve point

## Root Cause
```
ERROR: test_batch_scalar_mul.cpp:45
  Expected: is_on_curve(&result)
    Actual: false
```

The batch_scalar_mul() function is not properly initializing the Z coordinate in Jacobian representation.

## Minimal Fix
```diff
--- a/src/batch_kernel.cu
+++ b/src/batch_kernel.cu
@@ -123,6 +123,7 @@
     point.x = /* ... */;
     point.y = /* ... */;
+    point.z = field_one();  // Initialize Z coordinate
```

## Rationale
Jacobian points require Z=1 for affine conversion. The original code left Z uninitialized.

## Verification
- Run: `./build/tests/test_batch_scalar_mul`
- Expected: All tests pass

## Risk Assessment
- Low - Single line addition, no logic changes
- Affected modules: batch_kernel.cu only
```
```

---

## 10. 文档版本与变更记录

| 版本 | 日期 | 主要变更 | 作者 |
|------|------|---------|------|
| v1.0 | 2025-09-XX | 初始版本 | Original Team |
| v2.0 | 2025-09-30 | **工业级增强** | Enhanced Team |
|      |            | - 增加四级约束机制（L1铁律层、L2工程层、L3质量层） | |
|      |            | - 新增ZERO-NEW-FILES原则与白名单机制 | |
|      |            | - 新增NO-PLACEHOLDERS完整禁止模式库 | |
|      |            | - 新增INCREMENTAL-EDIT-ONLY阈值体系 | |
|      |            | - 新增REUSE-FIRST五级瀑布检查流程 | |
|      |            | - 完善Provenance Header与溯源追踪工具 | |
|      |            | - 新增完整CI流水线（7阶段门禁） | |
|      |            | - 新增TDD工作流与测试模板 | |
|      |            | - 新增夜间流水线与三级熔断机制 | |
|      |            | - 新增P0故障应急处理流程 | |
|      |            | - 新增标准Prompt模板与Fixer Agent协议 | |
|      |            | - 补充Developer与Reviewer完整检查表 | |
|      |            | - 增加性能反退化监控与趋势图 | |
|      |            | - 增加密码学一致性自动验证 | |

---

## 附录A：工具脚本索引

| 脚本路径 | 用途 | 执行频率 |
|---------|------|---------|
| `tools/sync_reference_sources.sh` | 同步参考源快照 | 每次工作前 |
| `scripts/trace_snapshot.sh` | 追溯文件溯源 | 按需 |
| `ci/scan_placeholders.sh` | 扫描禁止模式 | 每次CI |
| `ci/check_cuda_errors.sh` | 检查CUDA错误处理 | 每次CI |
| `ci/check_diff_size.sh` | 检查修改幅度 | 每次CI |
| `ci/check_new_files.sh` | 验证新文件 | 每次CI |
| `ci/check_provenance.sh` | 检查溯源头部 | 每次CI |
| `ci/nightly_build.sh` | 夜间完整构建 | 每晚 |
| `ci/circuit_breaker.sh` | 熔断机制 | 每晚 |
| `ci/handle_p0_incident.sh` | P0故障处理 | 故障时 |
| `benchmark/bench_gpu.sh` | GPU性能基准 | 每次CI + 每晚 |
| `tools/plot_perf_trend.py` | 性能趋势图 | 每晚 |
| `tools/check_coverage.sh` | 测试覆盖率 | 每次CI |
| `tools/generate_perf_report.py` | 性能报告 | 每次基准 |

---

## 附录B：快速参考卡

### 开发者速查

```bash
# 工作前准备
./tools/sync_reference_sources.sh --apply

# 本地验证（提交前必做）
make clean && make CXXFLAGS="-Wall -Wextra -Werror"
ctest --output-on-failure
./ci/scan_placeholders.sh
./ci/check_cuda_errors.sh
./ci/check_diff_size.sh
./tools/check_coverage.sh

# 提交
git add <files>
git commit -m "feat(scope): description"
git push origin <branch>

# 性能测试
./benchmark/bench_gpu.sh
python3 tools/check_perf_regression.py \
    --current build/benchmark/results/latest.json \
    --baseline benchmark/baseline/latest.json
```

### 审查者速查

```bash
# 检查PR
gh pr checkout <pr_number>

# 形式检查
./ci/check_new_files.sh
./ci/check_diff_size.sh

# 质量检查
./ci/scan_placeholders.sh
./ci/check_provenance.sh

# 运行测试
make clean && make && ctest

# 性能检查
./benchmark/bench_gpu.sh
```

---

**文档结束。所有规则强制执行，违反任何一条导致工作回滚。**
