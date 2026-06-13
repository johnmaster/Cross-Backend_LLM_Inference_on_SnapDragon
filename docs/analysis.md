# CPU 与 OpenCL GPU 推理结果分析

本文基于当前仓库中的 benchmark 日志：

```text
data/cpu/baseline_threads.log
data/cpu/quant_sweep.log
data/gpu/opencl_ngl_sweep.log
```

测试对象主要是 OnePlus 12 / Snapdragon 8 Gen 3 上的 Qwen2.5-3B-Instruct GGUF 模型。CPU 数据来自 `llama-bench` CPU backend；GPU 数据来自 OpenCL backend，并确认运行时识别到：

```text
ggml_opencl: selected platform: 'QUALCOMM Snapdragon(TM)'
ggml_opencl: device: 'QUALCOMM Adreno(TM) 750 (OpenCL 3.0 Adreno(TM) 750)'
```

## 1. CPU 基线

CPU 基线中，Qwen2.5-3B-Instruct Q4_K_M 在线程数 sweep 下的结果如下：

| 线程数 | pp512 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 1 | 10.80 | 4.12 |
| 2 | 28.76 | 8.37 |
| 4 | 55.31 | 16.90 |
| 6 | 67.05 | 17.34 |
| 8 | 54.85 | 10.83 |

CPU 侧主要结论：

- 4 到 6 线程是更合理的区间。
- 8 线程会明显退化，尤其是 decode 阶段。
- CPU 上 `tg128` 已经能达到可交互水平。

CPU 量化 sweep 中，每个模型的最佳结果如下。这里把 `pp512` 和 `tg128` 的最优线程数分开记录，因为二者不一定出现在同一个 `-t` 配置上：

| 模型文件 | 最佳 pp512 tok/s | pp 最优线程 | 最佳 tg128 tok/s | tg 最优线程 |
| --- | ---: | ---: | ---: | ---: |
| qwen2.5-3b-instruct-q4_0.gguf | 107.03 | 6 | 21.67 | 4 |
| qwen2.5-3b-instruct-q4_k_m.gguf | 61.24 | 6 | 16.70 | 6 |
| qwen2.5-3b-instruct-q5_k_m.gguf | 44.95 | 6 | 12.77 | 6 |
| qwen2.5-3b-instruct-q6_k.gguf | 44.92 | 6 | 12.69 | 6 |
| qwen2.5-3b-instruct-q8_0.gguf | 90.01 | 6 | 12.87 | 6 |

从 CPU 结果看，大多数模型在 `t=6` 下取得最佳 prefill。`Q4_0` 是一个例外：它的 `pp512` 最优在 `t=6`，但 `tg128` 最优在 `t=4`。如果目标是聊天生成速度，`Q4_0` 最快；如果目标是质量和速度平衡，`Q4_K_M` 更稳。

## 2. OpenCL 测试设置

OpenCL sweep 日志位于：

```text
data/gpu/opencl_ngl_sweep.log
```

测试参数：

```text
prompt tokens: 512
generation tokens: 128
threads: 4
ngl list: 0 8 16 24 32 99
models:
- qwen2.5-3b-instruct-q4_0.gguf
- qwen2.5-3b-instruct-q6_k.gguf
- qwen2.5-3b-instruct-q8_0.gguf
```

需要注意：OpenCL 的 `ngl=0` 结果不能直接当作纯 CPU baseline。它仍然运行在 OpenCL build 中，且日志显示 OpenCL platform/device 已初始化；实际吞吐明显低于单独 CPU build。因此 CPU 对照应优先看 `data/cpu/quant_sweep.log`。

## 3. OpenCL ngl 结果

### Q4_0

| ngl | pp512 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 0 | 13.17 | 8.57 |
| 8 | 12.72 | 8.92 |
| 16 | 16.50 | 9.62 |
| 24 | 25.75 | 10.12 |
| 32 | 42.44 | 9.85 |
| 99 | 116.45 | 14.05 |

观察：

- `pp512` 随 `ngl` 增大显著提升，`ngl=99` 达到 116.45 tok/s。
- `tg128` 也随 offload 增加有所提升，但最高只有 14.05 tok/s。
- 相比 CPU Q4_0 最佳值，OpenCL 的 prefill 更快，但 decode 更慢。

与 CPU 最佳结果对比：

| 后端 | pp512 tok/s | pp 配置 | tg128 tok/s | tg 配置 |
| --- | ---: | --- | ---: | --- |
| CPU Q4_0 | 107.03 | `t=6` | 21.67 | `t=4` |
| OpenCL Q4_0 | 116.45 | `ngl=99, t=4` | 14.05 | `ngl=99, t=4` |

Q4_0 的 OpenCL 结论：相比 CPU 最佳 prefill，OpenCL 仍有小幅提升；但 decode 明显低于 CPU 最佳值，不适合追求最快聊天生成速度。

### Q6_K

| ngl | pp512 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 0 | 9.26 | 7.50 |
| 8 | 9.28 | 6.27 |
| 16 | 11.87 | 6.68 |
| 24 | 16.28 | 7.38 |
| 32 | 28.01 | 6.73 |
| 99 | 51.57 | 7.27 |

观察：

- `pp512` 随 `ngl` 增大持续提升，`ngl=99` 达到 51.57 tok/s。
- `tg128` 没有随 `ngl` 明显提升，整体在 6 到 8 tok/s 之间。
- 对 Q6_K 来说，OpenCL offload 主要改善 prefill，不改善 decode。

与 CPU 最佳结果对比：

| 后端 | pp512 tok/s | tg128 tok/s |
| --- | ---: | ---: |
| CPU Q6_K | 44.92 | 12.69 |
| OpenCL Q6_K ngl=99 | 51.57 | 7.27 |

Q6_K 的 OpenCL 结论：prefill 略高于 CPU，但 decode 明显低于 CPU。

### Q8_0

| ngl | pp512 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 0 | 11.18 | 8.59 |
| 8 | 13.59 | 8.61 |
| 16 | 17.93 | 8.15 |
| 24 | 23.60 | 8.76 |
| 32 | 38.99 | 8.68 |
| 99 | 65.06 | 9.06 |

观察：

- `pp512` 同样随 `ngl` 增大提升，`ngl=99` 达到 65.06 tok/s。
- `tg128` 基本稳定在 8 到 9 tok/s，几乎没有从 GPU offload 中受益。
- Q8_0 文件最大，OpenCL 下 decode 表现不占优。

与 CPU 最佳结果对比：

| 后端 | pp512 tok/s | tg128 tok/s |
| --- | ---: | ---: |
| CPU Q8_0 | 90.01 | 12.87 |
| OpenCL Q8_0 ngl=99 | 65.06 | 9.06 |

Q8_0 的 OpenCL 结论：prefill 和 decode 都没有超过 CPU 最佳结果。

## 4. CPU 与 OpenCL 对比

为了更直观看出 CPU 和 OpenCL GPU 的差异，下面把 CPU 最佳结果与 OpenCL `ngl=99` 结果放在同一张表里。

| 模型 | CPU 最佳 pp512 | OpenCL pp512 | pp512 变化 | CPU 最佳 tg128 | OpenCL tg128 | tg128 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_0 | 107.03 | 116.45 | +8.8% | 21.67 | 14.05 | -35.2% |
| Q6_K | 44.92 | 51.57 | +14.8% | 12.69 | 7.27 | -42.7% |
| Q8_0 | 90.01 | 65.06 | -27.7% | 12.87 | 9.06 | -29.6% |

这张表说明：

- `Q4_0` 是 OpenCL 下收益最明显的模型，prefill 相比 CPU 最优 `t=6` 提升约 8.8%。
- `Q6_K` 的 prefill 也有小幅提升，但 decode 损失较大。
- `Q8_0` 在 OpenCL 下没有超过 CPU，prefill 和 decode 都下降。
- 三个模型的 OpenCL decode 都低于 CPU，因此 GPU offload 当前不能提升聊天生成速度。

如果按使用场景排序：

| 场景 | 当前更优配置 | 原因 |
| --- | --- | --- |
| 最快聊天输出 | CPU + Q4_0 | `tg128` 最高，为 21.67 tok/s |
| 长 prompt 输入处理 | OpenCL + Q4_0 + ngl=99 | `pp512` 最高，为 116.45 tok/s |
| 更高量化精度且稳定 decode | CPU + Q6_K / Q8_0 | OpenCL decode 低于 CPU |

## 5. OpenCL 总体结论

OpenCL 后端已经成功调用 Adreno 750，但性能收益集中在 prefill 阶段。

各模型在 `ngl=99` 下的 OpenCL 最佳摘要：

| 模型 | OpenCL pp512 tok/s | OpenCL tg128 tok/s | 相对 CPU 结论 |
| --- | ---: | ---: | --- |
| Q4_0 | 116.45 | 14.05 | prefill 高于 CPU，decode 低于 CPU |
| Q6_K | 51.57 | 7.27 | prefill 略高于 CPU，decode 低于 CPU |
| Q8_0 | 65.06 | 9.06 | prefill 和 decode 均低于 CPU 最佳值 |

最重要的规律：

- `ngl` 增大通常会显著提高 `pp512`。
- `ngl` 增大并不会稳定提高 `tg128`。
- OpenCL 对长输入 prompt 更有价值，对逐 token decode 帮助有限。
- 当前聊天生成速度最快的配置仍然是 CPU 上的 `Q4_0`。

## 6. 为什么 prefill 提升而 decode 不提升

`pp512` 和 `tg128` 对硬件的压力不同：

- `pp512` 一次处理 512 个输入 token，并行度更高，GPU 更容易发挥吞吐优势。
- `tg128` 是逐 token 生成，串行依赖强，每一步都需要 kernel launch、同步和调度。
- 移动 GPU 在小 batch decode 场景下不一定能跑赢 Snapdragon CPU。
- OpenCL backend 对不同量化格式的 kernel 优化程度也不一致。

decode 阶段 OpenCL 低于 CPU，主要不是因为 GPU 没接上，而是因为 decode 的计算形态不适合当前这条 OpenCL 路径：

1. 每次只生成一个 token。decode 不是一次处理很多 token，而是生成一个 token、更新 KV cache、再生成下一个 token。单步任务规模小，GPU 很难像 prefill 那样吃满并行度。
2. GPU 每一步都有固定开销。OpenCL 需要提交 kernel、同步结果、调度下一轮计算；当每轮计算量不大时，这些固定开销会吃掉 GPU 的理论算力优势。
3. CPU 量化 kernel 已经很成熟。`llama.cpp` 在 ARM CPU 上的量化矩阵/向量计算路径比较直接，数据在 CPU cache 和内存层级中流动更短，对 3B Q4_0 这种模型已经能达到较高 `tg128`。
4. GPU offload 可能带来额外的数据流转。即使手机是统一内存架构，CPU/GPU 之间仍然有同步、buffer 管理和命令队列开销，不等于完全免费共享。
5. decode 更接近 memory-bound 和 latency-bound。相比 prefill 的大矩阵并行吞吐，decode 更看重单 token 延迟；移动 GPU 的吞吐优势不一定能转化成低延迟。

所以 OpenCL 的收益集中在 `pp512` 是合理的：prefill 像“大批量并行任务”，decode 更像“很多次小任务串行排队”。当前数据说明 Adreno 750 能提升大块输入处理，但在逐 token 生成阶段，CPU 的低调度开销和成熟 kernel 更占优势。

因此，OpenCL 结果不能简单理解成“GPU 一定比 CPU 快”。更准确的说法是：

```text
Adreno OpenCL 可以提升 Q4_0 的 prefill，但当前没有提升 Qwen2.5-3B 的聊天生成速度。
```

## 7. 当前推荐

如果目标是最快聊天输出：

```text
CPU + Q4_0 + 4 threads
```

如果目标是更长 prompt 的输入处理速度：

```text
OpenCL + Q4_0 + ngl=99
```

如果目标是继续做 GPU 后端评测：

- 重点保留 `Q4_0`，因为它在 OpenCL 下 prefill 收益最明显。
- `Q6_K` 可以作为高量化版本补充，但 decode 不佳。
- `Q8_0` 当前不适合作为 OpenCL 优先配置。
- 后续可以增加不同 `-p` 长度，例如 `-p 1024`、`-p 2048`，观察 OpenCL 的 prefill 优势是否继续扩大。

## 8. 数据注意点

1. OpenCL sweep 使用 `threads=4`，CPU 最佳结果中部分模型使用的是 6 线程，因此对比时要注意线程数不同。
2. OpenCL `ngl=0` 不是纯 CPU baseline，不应用来替代 `data/cpu` 中的 CPU 数据。
3. 移动设备容易受温度和频率影响，长时间 sweep 后半段可能受到热状态影响。
4. 当前结果是单轮 sweep，正式报告中建议重复测试并记录均值、方差、温度和电源状态。
5. OpenCL 后端对不同量化格式支持程度不同，K-quant 模型不应默认纳入 OpenCL 主测试集。
