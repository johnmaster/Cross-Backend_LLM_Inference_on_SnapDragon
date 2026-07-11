# MatMul QHPI HVX 8-row LHS-prepack FP32-store multithread

本目录基于
`../matmul_qhpi_hvx_8row_lhs_prepack_fp32_store`，验证把 LHS 预转换和
QHPI self-slicing 多线程合在一起：

```text
FP32 LHS -> Cast FP16 -> PackLhs(Q13 bits) -> multithread MatMul -> FP32 output
```

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

## 多线程划分

`MatMulQhpiHvx8RowLhsPrepackFp32StoreMultithread` 设置：

```cpp
.multithreaded = true
```

kernel 里通过 QHPI handle 获取当前 slice：

```cpp
const uint32_t num_slices = qhpi_num_slices(handle);
const uint32_t slice = qhpi_slice_number(handle);
```

完整 8-row tile 按 slice 分配：

```text
row = slice * 8, slice * 8 + num_slices * 8, ...
```

tail row 按单行分配：

```text
row = full_rows + slice, full_rows + slice + num_slices, ...
```

这样每个 worker 只写自己负责的 output row，不需要额外锁。

## 编译

```bash
PKG=qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_prepack_fp32_store_multithread/htp/MatMulQhpiHvx8RowLhsPrepackFp32StoreMultithreadOpPackage

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
  -c qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_prepack_fp32_store_multithread/model/custom_matmul_qhpi_hvx_8row_lhs_prepack_fp32_store_multithread_model.cpp \
  -t aarch64-android \
  -l custom_matmul_qhpi_hvx_8row_lhs_prepack_fp32_store_multithread_model \
  -o qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_prepack_fp32_store_multithread/model_libs
```

## tiny block full graph 正确性

```text
prefill hidden: max=2.47808099e-02 mean=3.26919090e-03 cosine=0.999762475
prefill key: max=8.54909420e-04 mean=1.22145138e-04 cosine=0.999999940
prefill value: max=5.04963100e-04 mean=1.06622887e-04 cosine=1.000000000
```

这是把 `tiny_llm_block` prefill graph 里的 q_proj 替换成本 custom op 后，
与 builtin QNN HTP 输出对比的结果。

## tiny block full graph profiling

统一使用 `qnn-profile-viewer` 生成 CSV，并由
`tiny_llm_block_custom_matmul/tools/compare_prefill_profiles.py` 跳过第一次
warm-up 后取 median：

```bash
python tiny_llm_block_custom_matmul/tools/compare_prefill_profiles.py
```

当前结果：

```text
builtin             root_cycles=   360575 qnn_us= 2126 netrun_us= 3691 q_proj_cycles=        0 pack_lhs_cycles=     0
custom_multithread  root_cycles=  3933728 qnn_us= 6717 netrun_us= 8241 q_proj_cycles=  3618124 pack_lhs_cycles=     0
custom_lhs_prepack  root_cycles=  1682834 qnn_us= 7960 netrun_us=10762 q_proj_cycles=  1338650 pack_lhs_cycles=  8044
custom_lhs_prepack_mt root_cycles=  2670623 qnn_us= 6074 netrun_us= 7560 q_proj_cycles=  2289029 pack_lhs_cycles=  6446
```

## 结论

组合版本相对第一版 multithread custom op 有收益：

```text
q_proj cycles: 3.62M -> 2.29M
NetRun:        8241 us -> 7560 us
```

但它没有继续超过单线程 LHS-prepack kernel：

```text
q_proj cycles: 1.34M -> 2.29M
```

原因很可能是 tiny block prefill 的 q_proj 只有 `M=32, K=256, N=256`，
每个 worker 分到的 8-row tile 很少，多线程调度、同步和内存带宽竞争会吃掉一部分
收益。这个 package 证明了组合链路可用；下一步更值得做 RHS prepack 或融合
PackLhs + MatMul，减少中间 tensor 和重复转换。
