# MatMul QHPI HVX 8-row FP32-store multithread

本目录基于 `../09_matmul_qhpi_hvx_8row_fp32_store`，组合验证 QHPI self-slicing
多线程和算子内部 LHS tile-cache。算子接口仍为 FP16 LHS/RHS、FP32 output，
矩阵计算仍使用 Q13 HVX MAC。

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

这里不需要在 kernel 中使用 `pthread_create`、OpenMP 或手动创建线程。
`.multithreaded = true` 会启用 QHPI self-slicing：QNN/QHPI runtime
负责选择可用的 HVX worker，并让多个 worker 并行进入同一个 `Execute()`
函数。运行 `qnn-net-run` 时也不需要额外添加“线程数”参数。

`num_slices` 不是本算子设置的固定线程数，而是当前执行时由 runtime
通过 `qhpi_num_slices(handle)` 提供；每个 worker 再通过
`qhpi_slice_number(handle)` 获得自己的 `slice` 编号。kernel 使用这两个
值划分工作：

```cpp
const uint32_t num_slices = qhpi_num_slices(handle);
const uint32_t slice = qhpi_slice_number(handle);

for (uint32_t row = slice * 8;
     row < full_rows;
     row += num_slices * 8) {
    // 当前 worker 只计算分配给自己的 8-row tile
}
```

因此，“使用 multithread”包含两部分：

1. 注册阶段设置 `.multithreaded = true`，允许 runtime 多 worker 调度。
2. `Execute()` 阶段按照 `slice` 和 `num_slices` 切分输出行，避免不同
   worker 重复计算或写入同一位置。

如果只有一个 worker，runtime 会提供单 slice，循环仍能正常覆盖所有
输出行；如果存在多个 slice，同一套代码会自动并行执行。实际 slice
数量由 runtime 和设备资源决定，本算子没有把它固定为某个数值。

kernel 先计算能够被 8-row tile 完整覆盖的行数：

```cpp
full_rows = m - (m % 8);
```

这里的 `full_rows` 是将 `m` 向下取整到 8 的倍数，它由 8-row kernel
的 tile 高度决定，与线程数 `num_slices` 无关。例如 `m=130` 时，
`full_rows=128`：第 0～127 行由 16 个 8-row tile 处理，第 128～129
行进入单行 tail 路径。

`M=128` 被划分为 16 个 8-row tile。slice `s` 处理：

```text
tile s, s + num_slices, s + 2*num_slices, ...
```

不同 slice 写入不同输出行，因此不需要锁、barrier 或 QHPI sync block。
不足 8 行的尾部也按 slice 编号轮转分配。

## 编译

```bash
PKG=qnn_custom_ops/13_matmul_qhpi_hvx_8row_fp32_store_multithread/htp/MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage

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
  -c qnn_custom_ops/13_matmul_qhpi_hvx_8row_fp32_store_multithread/model/custom_matmul_qhpi_hvx_8row_fp32_store_multithread_model.cpp \
  -t aarch64-android \
  -l custom_matmul_qhpi_hvx_8row_fp32_store_multithread_model \
  -o qnn_custom_ops/13_matmul_qhpi_hvx_8row_fp32_store_multithread/model_libs
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

## 第二轮优化：multithread + fused LHS tile-cache

第一版 multithread kernel 的 self-slicing 虽然降低了 wall time，但每个
`8x64` output tile 内仍执行：

```text
8 个 FP16 LHS scalar -> Q13
1 个 64-lane FP16 RHS vector -> Q13
8 组 HVX MAC
```

对于当前 `N=256`，同一个 8-row LHS tile 会依次计算 4 个 64-column tile，
因此每个 LHS 元素被标量转换 4 次。多线程只是让多个 worker 并行执行这项重复工作，
没有消除转换本身。

当前版本在每个 worker 内为正在处理的 8-row tile 建立私有 Q13 cache：

```text
FP16 LHS tile [8,K]
  -> HVX FP16-to-Q13 conversion（每个元素一次）
  -> worker-local Q13 cache [8,K]
  -> 复用到所有 64-column output tiles
```

实现要点：

- cache 位于 `Execute()` 的 8-row self-slicing 循环内，不在 worker 之间共享，
  因此不需要锁或 barrier。
- 每行按 64 个 FP16 元素使用 `hnnx::s16_from_hf_rnd_sat<13>` 向量转换。
- MatMul reduction 内层直接从 cache 读取 Q13 LHS，只保留 splat 和 MAC。
- `K` 当前限制为不超过 1024；当前 standalone `K=256` 和 Qwen2.5-0.5B
  projection `K=896` 都在范围内。
- 非 8-row tail 仍使用原有单行路径，保持一般 shape 的正确性。

### 真机正确性

优化前后的输出逐 bit 相同：

```text
optimized vs old multithread:
  max_abs_error = 0
  mean_abs_error = 0
  array_equal    = True

optimized vs FP32 reference:
  elements       = 32768
  nonzero        = 32768
  max_abs_error  = 0.0007113218307495117
  mean_abs_error = 0.00009107097139349207
  allclose(1e-3) = True
  NaN / Inf      = 0 / 0
```

结果保存在：

```text
device_output_tile_cache/Result_0/output.raw
device_output_tile_cache/qnn-profiling-data_0.log
device_output_tile_cache/profile.csv
```

### 与旧 custom 和 QNN builtin 对比

测试条件：

```text
shape:              [1,1,128,256] x [1,1,256,256]
profiling:          detailed
num_inferences:     20
statistics:         丢弃第一次 warm-up 后取中位数
HVX threads:        4
input SHA256:       custom 与 builtin 完全一致
```

| 版本 | NetRun | Accelerator | QNN | root cycles | MatMul/custom-op cycles |
|---|---:|---:|---:|---:|---:|
| QNN builtin | 3,904 us | 2,344 us | 3,876 us | 91,395 | 约 20K–27K builtin subevent |
| 旧 multithread custom | — | 13,673 us | — | — | 约 14.1M |
| multithread + LHS tile-cache | 4,437 us | 2,862 us | 4,350 us | 747,048 | 708,626 |

相对旧 multithread custom：

```text
Accelerator: 13,673 us -> 2,862 us
降低:        79.07%
加速:        4.78x

custom cycles: 约 14.1M -> 708,626
降低:          约 94.97%
```

相对 QNN builtin：

```text
NetRun:      4,437 us vs 3,904 us，custom 慢 13.65%
Accelerator: 2,862 us vs 2,344 us，custom 慢 22.10%
```

builtin 的 MatMul subevent cycles 可能受到内部 fusion、tiling、prepack 和
profiling 归因影响，不能与 external custom-op cycles 做一比一解释；端到端判断仍以
NetRun/Accelerator wall time 为主。当前优化已经把差距从数倍缩小到约 14% 的
NetRun 差距。

### 剩余瓶颈

LHS 重复转换消除后，下一阶段重点是：

1. RHS 仍在每个 8-row tile 中从 FP16 转换为 Q13；静态 projection weight 应优先
   尝试真正的离线/precomputation，而不是每次 inference 重复转换。
2. external OpPackage dispatch 仍有固定开销。
3. 图边界仍存在 FP32 -> FP16 Cast；真实 Qwen graph 还可能额外包含 Reshape 和 bias
   Add。
4. 继续增加线程不是首选。当前已经使用 4 个 HVX worker，下一步应减少每个 worker
   的重复数据准备和图级节点。

## 真实 Qwen seq16 的 tile 粒度实验

该优化算子已经接入 `qwen_block_custom_qnn` 的真实 Qwen2.5-0.5B layer0
`q_proj [16,896] x [896,896]`。8-row self-slicing 只有两个 tiles，因此新增条件分支：

```cpp
const bool use_four_row_tiles = m == 16 && num_slices >= 4;
const uint32_t tile_rows = use_four_row_tiles ? 4 : 8;
```

这让 seq16 q_proj 的四个 HVX workers 各处理一个 4-row tile；standalone
`M=128` 和其他 shape 仍走复用率更高的 8-row kernel。

结果表明 4-row 分支改善 accelerator 路径，但没有改善完整 layer0 NetRun：

| Qwen custom 版本 | Accelerator | excluding wait | QNN accelerator | NetRun |
|---|---:|---:|---:|---:|
| multithread 8-row | 7,356 us | 5,448 us | 13,456 us | 15,861 us |
| multithread 4-row | 6,888 us | 4,987 us | 13,058 us | 15,884 us |

两个版本的三个 graph outputs 均逐 bit 相同。这个实验说明在 `M=16` 上增加有效
worker 数量可以降低 custom kernel wall time，但完整 Transformer block 已受到其他
节点和 runtime wait 限制。详细 graph-level 对比见
`qwen_block_custom_qnn/README.md`。
