# MatMul QHPI HVX 8-Row + Vector Convert

本目录是在 `../matmul_qhpi_hvx_multi_row_vector_convert` 基础上的累计优化：

```text
原版本 tile: 4x64
当前版本 tile: 8x64
```

目标是在 8 个输出行之间复用同一个 RHS Q13 vector，进一步减少 RHS
FP16 -> Q13 向量转换和加载次数。

## 1. 为什么选择 8-Row

前一版 `4x64` kernel 对每个 reduction 执行：

```text
1 次 64-lane RHS vector conversion
4 次 LHS scalar conversion
4 次 HVX multiply-accumulate
```

本版本改为：

```text
1 次 64-lane RHS vector conversion
8 次 LHS scalar conversion
8 次 HVX multiply-accumulate
```

计算 8 行时，RHS conversion 次数从 2 次减少为 1 次。矩阵 `M=128` 可被
8 整除，因此正式 benchmark 不进入 row tail。

## 2. 为什么没有先做 RHS Prepack

曾计划在 Execute 开始时将完整 RHS 转换为 Q13。当前 shape 需要：

```text
K * N * sizeof(int16_t)
= 256 * 256 * 2
= 131,072 bytes
```

但 QHPI `RuntimeHandle` 没有公开通用临时 allocator：

- 128 KiB 不适合放在 DSP stack；
- 全局可变 buffer 不支持并发 graph/context；
- 普通 heap allocation 会引入生命周期、对齐和实时性问题；
- QHPI precomputed data 在 graph load 阶段生成；
- 当前 RHS 是 APP_WRITE runtime input，graph load 时没有实际 RHS 数据。

因此没有用危险的内存方案伪装成 prepack 优化。

如果未来模型将 RHS 变为 static weight，可以重新研究：

```text
precomputed_data_size
do_precomputation_function
function_with_precomputed_data
```

## 3. 为什么没有保留 Vector Store 版本

曾尝试将 scalar 输出写回替换为：

```text
int32 accumulator
  -> Q6_Vsf_equals_Vw
  -> Q6_Vsf_vmpy_VsfVsf
  -> Q6_Vhf_vcvt_VsfVsf
  -> vector store
```

该版本可以编译、注册和执行，但设备输出全部为零。这说明当前 SM8650/v75
环境的 direct FP32 -> FP16 HVX conversion 路径不可直接依赖。

失败版本已删除，详细记录保存在：

```text
../matmul_qhpi_hvx_multi_row_vector_convert/README.md
```

本目录继续使用经过验证的 scalar 输出转换。

## 4. Kernel 结构

每个 `8x64` tile 维护 8 个 `HVX_VectorPair` accumulator：

```cpp
HVX_VectorPair acc0;
HVX_VectorPair acc1;
HVX_VectorPair acc2;
HVX_VectorPair acc3;
HVX_VectorPair acc4;
HVX_VectorPair acc5;
HVX_VectorPair acc6;
HVX_VectorPair acc7;
```

每个 reduction 只转换一次 RHS：

```cpp
const HVX_Vector rhs_vec =
    convert64Fp16ToQ13(&rhs[rhs_row_base]);
```

然后更新 8 行：

```cpp
acc0 = Q6_Ww_vmpyacc_WwVhVh(acc0, lhs0, rhs_vec);
// ...
acc7 = Q6_Ww_vmpyacc_WwVhVh(acc7, lhs7, rhs_vec);
```

不足 8 行时仍使用 `1x64` fallback，剩余不足 64 列时仍使用 scalar tail。

## 5. 独立名称

```text
Directory:
  matmul_qhpi_hvx_8row_vector_convert

QHPI Op:
  MatMulQhpiHvx8RowVectorConvert

OpPackage:
  MatMulQhpiHvx8RowVectorConvertOpPackage

Interface provider:
  MatMulQhpiHvx8RowVectorConvertOpPackageInterfaceProvider

Model:
  libcustom_matmul_qhpi_hvx_8row_vector_convert_model.so
```

## 6. 编译

```bash
PKG=qnn_custom_ops/matmul_qhpi_hvx_8row_vector_convert/htp/MatMulQhpiHvx8RowVectorConvertOpPackage
```

HTP：

```bash
make -C "$PKG" htp_v75 \
  QNN_INCLUDE=/home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN \
  HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  HEXAGON_SDK_ROOT_V75=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  HEXAGON_TOOLS_VERSION_V75=8.7.06
```

ARM：

```bash
make -C "$PKG" htp_aarch64 \
  QNN_INCLUDE=/home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN \
  QNN_TARGET_LIB=/home/lingbok/Qualcomm/qairt/2.47.0.260601/lib/aarch64-android \
  HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  X86_LIBNATIVE_RELEASE_DIR=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0/tools/HEXAGON_Tools/8.7.06/Tools \
  ANDROID_NDK_ROOT=/home/lingbok/android/android-ndk-r28
```

Model：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qnn_custom_ops/matmul_qhpi_hvx_8row_vector_convert/model/custom_matmul_qhpi_hvx_8row_vector_convert_model.cpp \
  -t aarch64-android \
  -l custom_matmul_qhpi_hvx_8row_vector_convert_model \
  -o qnn_custom_ops/matmul_qhpi_hvx_8row_vector_convert/model_libs
```

## 7. 正确性

```text
max_abs_error:  0.0009766817092895508
mean_abs_error: 0.00007510452996939421
allclose(1e-3): True
NaN / Inf:      0 / 0
```

结果与 4-row vector-convert 版本一致。

## 8. Profiling 方法

使用自编译的 profile SampleApp：

```text
/data/local/tmp/qnn/bin/qnn-sample-app-profile
--profiling_level detailed
--num_inferences 20
```

过滤节点：

```bash
grep 'identifier=MatMulQhpiHvx8RowVectorConvert_0.*cycles'
```

丢弃第一轮，剩余 19 次取中位数。

## 9. 性能结果

| 实现 | 中位数 cycles | 最小 cycles | 最大 cycles |
| --- | ---: | ---: | ---: |
| Baseline `1x64` | 793,889,648 | 782,945,312 | 801,008,396 |
| 4-row + vector convert | 15,042,197 | 14,978,439 | 15,107,041 |
| 8-row + vector convert | 14,554,119 | 14,526,149 | 14,624,822 |

相对 4-row：

```text
speedup = 15,042,197 / 14,554,119 = 1.0335x
cycles reduction = 3.2447%
```

相对 Baseline：

```text
speedup = 793,889,648 / 14,554,119 = 54.5474x
```

## 10. 结论

8-row 是有效优化，但收益只有约 `3.35%`。原因是 RHS conversion 在向量化后
已经不再占绝对主导，扩大行 tile 只能减少其中一部分；与此同时 8 个
accumulator pair 增加了寄存器压力和写回开销。

下一步不应直接扩展到 16-row。16 个 accumulator pair 本身就会占满 32 个
HVX vector registers，几乎必然导致 spill。更合理的后续方向是：

- 对 `FBITS=13` Q13 vector conversion 做完整输入域验证；
- 分析 8-row 反汇编中的寄存器 spill；
- 研究可用的非 direct-HF 输出转换方式；
- 在 RHS 为 static weight 的模型中测试 QHPI precomputation；
- 研究 QHPI 多线程切分不同 M tile。
