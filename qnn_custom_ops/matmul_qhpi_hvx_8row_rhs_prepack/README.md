# MatMul QHPI HVX 8-Row RHS Prepack

本目录在 `../matmul_qhpi_hvx_8row_vector_convert` 基础上，将 RHS 转换拆成
独立 QHPI Op：

```text
RHS FP32
  -> QNN Cast FP16
  -> MatMulQhpiHvxPackRhs
  -> Q13 bit-carrier tensor
  -> MatMulQhpiHvx8RowRhsPrepack
```

目标是让整张 RHS 每次 inference 只执行一次 FP16 -> Q13 转换，而不是在
每个 8-row tile 中重复转换。

## 1. 两个 QHPI Op

同一个 OpPackage 注册：

```text
MatMulQhpiHvxPackRhs
MatMulQhpiHvx8RowRhsPrepack
```

### PackRhs

输入：

```text
FP16 [B,H,K,N]
```

输出：

```text
16-bit Q13 bits [B,H,K,N]
```

完整 64-element tile 使用：

```cpp
const HVX_Vector input_fp16 = vmemu(&input[index]);
vmemu(&output[index]) = hnnx::s16_from_hf_rnd_sat<13>(input_fp16);
```

### Prepacked MatMul

输入：

```text
lhs: FP16
rhs: prepacked Q13 int16 bits
```

核心循环直接加载 RHS：

```cpp
const HVX_Vector rhs_vec = vmemu(&rhs[rhs_row_base]);
```

不再调用 RHS conversion helper。

## 2. 为什么使用 Graph 中间 Tensor

当前 RHS 是 APP_WRITE runtime input，graph load 时没有实际数据，所以不能
直接使用 QHPI：

```text
precomputed_data_size
do_precomputation_function
function_with_precomputed_data
```

中间 Q13 数据大小为：

```text
256 * 256 * sizeof(int16_t) = 131,072 bytes
```

它不适合 DSP stack；全局 buffer 又会破坏多 context 和并发安全。因此通过
额外 PackRhs graph node 让 QNN 管理中间 tensor 的分配和生命周期。

## 3. FP16 Bit-Carrier

最初将中间 tensor 声明为：

```text
QNN_DATATYPE_SFIXED_POINT_16
```

但 HTP prepare 将 graph tensor 显示为 `QUInt16`，而 kernel signature 的
量化类型映射无法稳定匹配，即使尝试 `QHPI_QINT16`、`QHPI_QUINT16` 和
`QHPI_ELEMENT_TYPE_ANY` 仍失败。

最终中间 tensor 声明为 FP16：

```text
rhs_q13_bits: QNN_DATATYPE_FLOAT_16
```

但它只是 Package 内部的 16-bit bit carrier：

- PackRhs 将 raw pointer 解释为 `int16_t*` 并写入 Q13 bits；
- MatMul 将同一 raw pointer 解释为 `const int16_t*`；
- 两个节点之间没有任何 QNN 算子解释其 FP16 数值；
- Execute 内检查 tensor element size 必须为 2 bytes。

这是一种针对 QAIRT 2.47 QHPI 类型匹配限制的 workaround，不应将该中间
tensor 暴露给其他普通 FP16 Op。

## 4. 独立名称

```text
Directory:
  matmul_qhpi_hvx_8row_rhs_prepack

OpPackage:
  MatMulQhpiHvx8RowRhsPrepackOpPackage

Ops:
  MatMulQhpiHvxPackRhs
  MatMulQhpiHvx8RowRhsPrepack

Model:
  libcustom_matmul_qhpi_hvx_8row_rhs_prepack_model.so
```

## 5. 编译

```bash
PKG=qnn_custom_ops/matmul_qhpi_hvx_8row_rhs_prepack/htp/MatMulQhpiHvx8RowRhsPrepackOpPackage
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
  -c qnn_custom_ops/matmul_qhpi_hvx_8row_rhs_prepack/model/custom_matmul_qhpi_hvx_8row_rhs_prepack_model.cpp \
  -t aarch64-android \
  -l custom_matmul_qhpi_hvx_8row_rhs_prepack_model \
  -o qnn_custom_ops/matmul_qhpi_hvx_8row_rhs_prepack/model_libs
```

修改 kernel signature 后必须同时重编并推送 ARM 和 HTP library。ARM package
也包含 OpInfo metadata；只更新 DSP `.so` 会继续使用旧 signature，导致
kernel mismatch。

## 6. 成功日志

Graph finalize 应同时出现：

```text
Matched kernel 'matmulqhpihvxpackrhs_float_16_Execute'
Matched kernel 'matmulqhpihvx8rowrhsprepack_float_16_Execute'
```

执行成功：

```text
QnnGraph_execute done. status 0x0
```

## 7. 正确性

```text
max_abs_error:  0.0009766817092895508
mean_abs_error: 0.00007510452996939421
allclose(1e-3): True
NaN / Inf:      0 / 0
```

结果与 8-row vector-convert 版本一致。

## 8. Profiling

使用：

```text
qnn-sample-app-profile
--profiling_level detailed
--num_inferences 20
```

必须分别提取两个节点，并按同一 inference 相加：

```text
MatMulQhpiHvxPackRhs_0 cycles
MatMulQhpiHvx8RowRhsPrepack_0 cycles
```

丢弃第一轮，剩余 19 次取中位数。

## 9. 性能结果

| 项目 | 中位数 cycles | 最小 cycles | 最大 cycles |
| --- | ---: | ---: | ---: |
| PackRhs | 32,377 | 31,690 | 33,029 |
| Prepacked MatMul | 14,401,232 | 14,301,700 | 14,459,765 |
| Pack + MatMul total | 14,433,290 | 14,334,729 | 14,492,142 |

对比：

| 实现 | 中位数 cycles |
| --- | ---: |
| Baseline | 793,889,648 |
| 4-row vector-convert | 15,042,197 |
| 8-row vector-convert | 14,554,119 |
| 8-row RHS prepack total | 14,433,290 |

相对 8-row vector-convert：

```text
speedup = 14,554,119 / 14,433,290 = 1.00837x
cycles reduction = 0.8302%
```

相对原始 Baseline：

```text
speedup = 793,889,648 / 14,433,290 = 55.0041x
```

## 10. 结论

RHS prepack 正确且略有收益，但在当前 shape 上仅提升约 `0.84%`。

原因：

- RHS vector conversion 已经非常快；
- PackRhs 新增一次完整 RHS 写入；
- MatMul 随后又从中间 tensor 读取 128 KiB；
- 减少的重复转换收益被额外内存流量抵消了大部分。

对于 runtime RHS，这项优化价值有限。对于 static model weight，真正的
graph-load precomputation 可以完全移除每次 inference 的 PackRhs 节点，
更值得后续验证。
