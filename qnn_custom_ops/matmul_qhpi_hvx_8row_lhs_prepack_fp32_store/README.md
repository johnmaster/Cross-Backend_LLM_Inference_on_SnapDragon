# MatMul QHPI HVX 8-row LHS prepack FP32-store

本目录基于 `../matmul_qhpi_hvx_8row_fp32_store`，单独验证 LHS 预转换。

## 为什么优化 LHS

原 8-row kernel 对每个 `8x64` 输出 tile 执行：

```text
FP16 LHS scalar -> Q13 -> splat -> MAC
```

当 `N=256` 时有 4 个 64-column tile，同一组 LHS 因此被重复量化 4 次。
而且这些转换位于 reduction 内层，是标量操作。

本版本把图改成：

```text
lhs FP32 -> Cast FP16 -> MatMulQhpiHvxPackLhs -> lhs_q13_bits
rhs FP32 -> Cast FP16 -------------------------> MatMul
```

`PackLhs` 使用 `hnnx::s16_from_hf_rnd_sat<13>`，每条 HVX 指令转换 64 个
FP16 元素。MatMul 直接读取 `int16_t` Q13，只保留 splat 和 MAC。

中间 tensor 使用 `QNN_DATATYPE_FLOAT_16` 作为 package-private 16-bit bit
carrier。QNN 不解释其数值，PackLhs 写入 Q13 bits，MatMul 按 `int16_t`
读取。

## 编译

```bash
PKG=qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_prepack_fp32_store/htp/MatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage

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
  -c qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_prepack_fp32_store/model/custom_matmul_qhpi_hvx_8row_lhs_prepack_fp32_store_model.cpp \
  -t aarch64-android \
  -l custom_matmul_qhpi_hvx_8row_lhs_prepack_fp32_store_model \
  -o qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_prepack_fp32_store/model_libs
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

结果与未 prepack 的 FP32-store 版本完全一致。

## Profiling

运行 20 次并丢弃第一次 warm-up：

| 项目 | Median | Min | Max |
|---|---:|---:|---:|
| PackLhs | 18,692 cycles | 17,375 | 19,537 |
| Prepacked MatMul | 5,347,014 cycles | 5,322,908 | 5,356,290 |
| Accelerator execute | 16,191 us | 16,096 us | 16,243 us |

与单线程 FP32-store 比较：

```text
MatMul:      12,940,799 -> 5,347,014 cycles
核心加速比:  2.42x

Accelerator: 36,023 -> 16,191 us
端到端加速比: 2.22x
延迟降低:     55.05%
```

PackLhs 只占约 18.7K cycles，远小于从 MatMul 内层消除的重复标量转换成本。

## 结论

LHS 重复量化是单线程 8-row FP32-store 的主要瓶颈。下一步最有价值的组合是：

```text
LHS prepack + FP32 store + QHPI multithread
```

单独的多线程版本为 `13,673 us`，单独的 LHS prepack 为 `16,191 us`；
两者优化的是不同层面，组合后仍有继续下降的空间，但实际收益需要测试共享
RHS 转换和多个 worker 的带宽竞争。
