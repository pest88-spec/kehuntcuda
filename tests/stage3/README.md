# Stage 3 测试说明

## 快速开始

### 安装依赖

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libgtest-dev cmake build-essential

# 编译gtest (如果未预编译)
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib
```

### 编译测试

```bash
cd tests/stage3
chmod +x build_test.sh
./build_test.sh
```

### 运行测试

```bash
# 运行所有测试
./test_scan_kernel_correctness

# 运行特定测试
./test_scan_kernel_correctness --gtest_filter="*ScanIsInfinity*"

# 详细输出
./test_scan_kernel_correctness --gtest_verbose
```

## 测试清单

### ✅ 已实现测试（5个）

1. **ScanIsInfinity_ZeroPoint_ReturnsTrue**
   - 验证零点被识别为无穷远点

2. **ScanIsInfinity_NonZeroPoint_ReturnsFalse**
   - 验证非零点不是无穷远点

3. **ScanMatchHash160_IdenticalHashes_ReturnsTrue**
   - 验证相同hash160匹配

4. **ScanMatchHash160_DifferentHashes_ReturnsFalse**
   - 验证不同hash160不匹配

5. **ScanHash160Compressed_GeneratorPoint_KnownVector**
   - 验证生成元G的hash160计算正确性
   - 预期: 751e76e8199196d454941c45d1b3a323f1433bd6
   - 地址: 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH

### ⏭️ 跳过的测试（3个）

- `HelperFunctions_Hash160Compressed_PlaceholderForPhase6`
- `HelperFunctions_Hash160Uncompressed_PlaceholderForPhase6`
- `ScanBloomCheck_WrapperCorrectness_Pending`

这些测试等待Phase 7实现。

## 测试覆盖

| Helper函数 | 测试状态 |
|-----------|---------|
| scan_is_infinity | ✅ 2个测试 |
| scan_match_hash160 | ✅ 2个测试 |
| scan_hash160_compressed | ✅ 1个测试 |
| scan_hash160_uncompressed | ⏭️ 待实现 |
| scan_point_add | ⏭️ 待实现 |
| scan_bloom_check | ⏭️ 待实现 |

## 预期结果

```
[==========] Running 8 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 8 tests from ScanKernelCorrectnessTest
[ RUN      ] ScanKernelCorrectnessTest.ScanIsInfinity_ZeroPoint_ReturnsTrue
[       OK ] ScanKernelCorrectnessTest.ScanIsInfinity_ZeroPoint_ReturnsTrue (X ms)
[ RUN      ] ScanKernelCorrectnessTest.ScanIsInfinity_NonZeroPoint_ReturnsFalse
[       OK ] ScanKernelCorrectnessTest.ScanIsInfinity_NonZeroPoint_ReturnsFalse (X ms)
...
[  PASSED  ] 5 tests.
[  SKIPPED ] 3 tests.
```

## 故障排除

### 编译错误: "fatal error: gtest/gtest.h"
```bash
sudo apt-get install libgtest-dev
```

### 编译错误: "undefined reference to pthread_create"
```bash
# build_test.sh已包含 -lpthread，检查gtest安装
```

### 运行错误: "No CUDA device available"
- 确保有可用的CUDA GPU
- 检查CUDA驱动: `nvidia-smi`

### 测试失败: Hash160不匹配
- 检查字节序（大端/小端）
- 验证EC点坐标
- 查看printf输出的详细mismatch信息

## 文件说明

- `test_scan_kernel_correctness.cu` - 测试源代码（268行）
- `build_test.sh` - 编译脚本
- `README.md` - 本文档

## 参考

- PROJECT_CHARTER.md - 项目规范
- docs/stage3_implementation_plan.md - Stage 3计划
- docs/stage3_phase6_completion_summary.md - Phase 6完成报告
