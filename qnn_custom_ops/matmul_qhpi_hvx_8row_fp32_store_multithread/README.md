# MatMul QHPI HVX 8-row FP32-store multithread

本目录基于 `../matmul_qhpi_hvx_8row_fp32_store`，验证 QHPI self-slicing
多线程策略。矩阵、Q13 计算和 FP32 vector store 均保持不变。

## 优化方法

kernel 注册为：

```cpp
.resources = QHPI_RESOURCE_HVX,
.multithreaded = true,
```

QNN runtime 使用多个 HVX worker 调用同一个 kernel。每次调用通过：

```cpp
qhpi_num_slices(handle)
qhpi_slice_number(handle)
```

取得 slice 总数和当前编号。

`M=128` 被划分为 16 个 8-row tile。slice `s` 处理：

```text
tile s, s + num_slices, s + 2*num_slices, ...
```

不同 slice 写入不同输出行，因此不需要锁、barrier 或 QHPI sync block。
不足 8 行的尾部也按 slice 编号轮转分配。

## 编译

```bash
PKG=qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store_multithread/htp/MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage

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
  -c qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store_multithread/model/custom_matmul_qhpi_hvx_8row_fp32_store_multithread_model.cpp \
  -t aarch64-android \
  -l custom_matmul_qhpi_hvx_8row_fp32_store_multithread_model \
  -o qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store_multithread/model_libs
```

修改 signature 或 `multithreaded` metadata 后，ARM 与 HTP package 必须同时重编。

## 正确性

```text
elements:       32768
nonzero:        32768
max_abs_error:  0.0007113218307495117
mean_abs_error: 0.00009107097139349207
allclose(1e-3): True
NaN / Inf:      0 / 0
```

结果与单线程 FP32-store 版本一致。

## Profiling 方法

使用：

```text
--profiling_level detailed
--num_inferences 20
```

丢弃第一次 warm-up，比较 graph-level：

```text
identifier=Accelerator (execute) time unit=1
```

结果：

| 版本 | Median | Min | Max |
|---|---:|---:|---:|
| 单线程 FP32-store | 36,023 us | 35,517 us | 36,415 us |
| 多线程 FP32-store | 13,673 us | 13,518 us | 13,828 us |

```text
加速比:   2.63x
延迟降低: 62.04%
```

## Profiling 注意事项

多线程版本的 custom-op `cycles` median 约为 14.1M，反而高于单线程约
12.94M。这个事件包含 self-sliced workers 的聚合计算量，不能视作墙钟延迟。

多线程版本必须比较 `Accelerator (execute) time`：

```text
单线程约 36.0 ms
多线程约 13.7 ms
```

日志时间戳中的 inference 间隔也从约 48 ms 降至约 25 ms，与 graph-level
时间的加速方向一致。
