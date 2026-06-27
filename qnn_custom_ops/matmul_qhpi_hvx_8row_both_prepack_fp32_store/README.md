# MatMul QHPI HVX 8-row both-prepack FP32-store

本目录基于 `../matmul_qhpi_hvx_8row_lhs_prepack_fp32_store`，同时预转换
LHS 和 RHS。

## 图结构

```text
lhs FP32 -> Cast FP16 -> PackLhs -> lhs_q13_bits --+
                                                    +-> Q13 MatMul -> FP32
rhs FP32 -> Cast FP16 -> PackRhs -> rhs_q13_bits --+
```

package 包含三个 QHPI op：

```text
MatMulQhpiHvxPackLhs
MatMulQhpiHvxPackRhs
MatMulQhpiHvx8RowBothPrepackFp32Store
```

两个中间 tensor 均使用 `QNN_DATATYPE_FLOAT_16` 作为 package-private
16-bit bit carrier。Pack op 写入 raw Q13 bits，MatMul 按 `int16_t` 读取。

MatMul reduction 内层只保留：

```cpp
const HVX_Vector rhs_vec = vmemu(rhs_q13);
const HVX_Vector lhs_vec = Q6_Vh_vsplat_R(lhs_q13);
acc = Q6_Ww_vmpyacc_WwVhVh(acc, lhs_vec, rhs_vec);
```

## 编译

```bash
PKG=qnn_custom_ops/matmul_qhpi_hvx_8row_both_prepack_fp32_store/htp/MatMulQhpiHvx8RowBothPrepackFp32StoreOpPackage

make -C "$PKG" htp_v75 \
  QNN_INCLUDE=/home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN \
  HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  HEXAGON_SDK_ROOT_V75=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  HEXAGON_TOOLS_VERSION_V75=8.7.06

make -C "$PKG" htp_aarch64 \
  QNN_INCLUDE=/home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN \
  QNN_TARGET_LIB=/home/lingbok/Qualcomm/qairt/2.47.0.260601/lib/aarch64-android \
  HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  X86_LIBNATIVE_RELEASE_DIR=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0/tools/HEXAGON_Tools/8.7.06/Tools \
  ANDROID_NDK_ROOT=/home/lingbok/android/android-ndk-r28

PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qnn_custom_ops/matmul_qhpi_hvx_8row_both_prepack_fp32_store/model/custom_matmul_qhpi_hvx_8row_both_prepack_fp32_store_model.cpp \
  -t aarch64-android \
  -l custom_matmul_qhpi_hvx_8row_both_prepack_fp32_store_model \
  -o qnn_custom_ops/matmul_qhpi_hvx_8row_both_prepack_fp32_store/model_libs
```

## 正确性

```text
elements:       32768
nonzero:        32768
max_abs_error:  0.0007113218307495117
mean_abs_error: 0.00009107097139349207
allclose(1e-3): True
NaN / Inf:      0 / 0
```

结果与 LHS-only prepack 及 FP32-store 版本一致。

## Profiling

20 次 inference，丢弃第一次 warm-up：

| 项目 | Median | Min | Max |
|---|---:|---:|---:|
| PackLhs | 16,969 cycles | 16,403 | 19,782 |
| PackRhs | 32,013 cycles | 31,430 | 32,863 |
| Both-prepack MatMul | 5,589,103 cycles | 5,584,944 | 5,612,041 |
| Accelerator execute | 16,903 us | 16,843 us | 16,968 us |

对比结果：

| 版本 | Accelerator median |
|---|---:|
| FP32-store | 36,023 us |
| LHS-only prepack | 16,191 us |
| Both prepack | 16,903 us |

```text
Both prepack vs FP32-store:
加速比:   2.13x
延迟降低: 53.08%

Both prepack vs LHS-only:
性能回退: 4.40%
```

## 结论

双 prepack 在功能上成立，但对当前矩阵和 kernel 不是更优方案。

`PackRhs` 本身只有约 32K cycles，主要回退发生在 MatMul：

```text
LHS-only MatMul: 5,347,014 cycles
Both MatMul:     5,589,103 cycles
```

当前 RHS 的 FP16-to-Q13 向量转换每个向量会被 8 行复用，并且编译器可以将
转换与 MAC 流水重叠。改成独立 RHS 中间 tensor 后，额外的 tensor 生产、
调度和读取路径没有被省下的转换成本抵消。

因此当前最佳选择仍是：

```text
LHS prepack + RHS inline vector conversion
```

若 RHS 是跨多次 inference 不变的权重，把 RHS Q13 打包移到 graph load 或
离线模型生成阶段，才可能消除每次 inference 的 PackRhs 成本并改变结论。
