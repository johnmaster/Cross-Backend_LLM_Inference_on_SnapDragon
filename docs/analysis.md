# data/cpu 分析

本文只分析当前 `data/cpu` 目录下已经归档的 CPU 后端数据。

## 1. 文件概览

`data/cpu` 目录当前包含 3 个文件：

```text
data/cpu/
├── baseline_threads.log
├── quant_sweep.log
└── llama-bench-report.html
```

各文件含义：

- `baseline_threads.log`：固定 Qwen2.5-3B-Instruct Q4_K_M 模型，扫描不同 CPU 线程数的 `llama-bench` 结果。
- `quant_sweep.log`：扫描多个 GGUF 量化版本，并在每个量化版本下测试 2、4、6、8 线程。
- `llama-bench-report.html`：一次 `simpleperf` 采样后生成的 HTML profiling 报告，用于进一步查看函数热点、调用图和火焰图。

日志中的 benchmark 都使用 CPU 后端，主要指标是：

- `pp512`：prompt processing，输入 prompt 处理速度，单位为 tok/s。
- `tg128`：text generation，输出 token 生成速度，单位为 tok/s。

其中 `tg128` 更接近日常聊天时用户感受到的生成速度。

## 2. baseline_threads.log

`baseline_threads.log` 使用的模型是：

```text
qwen2.5-3b-instruct-q4_k_m.gguf
```

测试参数：

```text
-p 512
-n 128
backend: CPU
```

结果摘要：

| 线程数 | pp512 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 1 | 10.80 | 4.12 |
| 2 | 28.76 | 8.37 |
| 4 | 55.31 | 16.90 |
| 6 | 67.05 | 17.34 |
| 8 | 54.85 | 10.83 |

结论：

- 1 到 4 线程提升很明显，`tg128` 从 4.12 tok/s 提升到 16.90 tok/s。
- 6 线程的 `pp512` 最高，为 67.05 tok/s；`tg128` 也最高，为 17.34 tok/s。
- 8 线程性能明显下降，`tg128` 只有 10.83 tok/s，低于 4 线程和 6 线程。
- 对 Q4_K_M 来说，当前数据下推荐线程数是 4 或 6，不建议默认使用 8 线程。

这个结果说明手机 CPU 推理不是线程越多越好。8 线程可能引入调度开销、内存带宽压力、温控降频，或者让低效核心参与过多，导致生成速度反而下降。

## 3. quant_sweep.log

`quant_sweep.log` 覆盖了 5 个模型文件和 4 组线程数：

```text
models:
- qwen2.5-3b-instruct-q4_0.gguf
- qwen2.5-3b-instruct-q6_k.gguf
- qwen2.5-3b-instruct-q4_k_m.gguf
- qwen2.5-3b-instruct-q8_0.gguf
- qwen2.5-3b-instruct-q5_k_m.gguf

threads:
- 2
- 4
- 6
- 8
```

每个模型的最佳结果如下：

| 模型文件 | 日志显示大小 | 最佳线程数 | 最佳 pp512 tok/s | 最佳 tg128 tok/s |
| --- | ---: | ---: | ---: | ---: |
| qwen2.5-3b-instruct-q4_0.gguf | 1.86 GiB | 4 | 87.94 | 21.67 |
| qwen2.5-3b-instruct-q4_k_m.gguf | 1.95 GiB | 6 | 61.24 | 16.70 |
| qwen2.5-3b-instruct-q5_k_m.gguf | 2.27 GiB | 6 | 44.95 | 12.77 |
| qwen2.5-3b-instruct-q6_k.gguf | 2.27 GiB | 6 | 44.92 | 12.69 |
| qwen2.5-3b-instruct-q8_0.gguf | 3.36 GiB | 6 | 90.01 | 12.87 |

按 `tg128` 排序：

| 排名 | 模型文件 | 最佳 tg128 tok/s | 对应线程数 |
| ---: | --- | ---: | ---: |
| 1 | qwen2.5-3b-instruct-q4_0.gguf | 21.67 | 4 |
| 2 | qwen2.5-3b-instruct-q4_k_m.gguf | 16.70 | 6 |
| 3 | qwen2.5-3b-instruct-q8_0.gguf | 12.87 | 6 |
| 4 | qwen2.5-3b-instruct-q5_k_m.gguf | 12.77 | 6 |
| 5 | qwen2.5-3b-instruct-q6_k.gguf | 12.69 | 6 |

主要观察：

- `Q4_0` 的生成速度最高，最佳 `tg128` 为 21.67 tok/s。
- `Q4_K_M` 的生成速度低于 `Q4_0`，但仍达到 16.70 tok/s，是当前更均衡的选择。
- `Q8_0` 的 `pp512` 很高，6 线程达到 90.01 tok/s，但 `tg128` 只有 12.87 tok/s；它并不适合作为追求聊天生成速度时的首选。
- `Q5_K_M` 与日志里的 `q6_k` 结果非常接近，而且日志中二者大小都显示为 2.27 GiB，模型显示名也都类似 `qwen2 3B Q5_K - Medium`。这里建议后续核对 `qwen2.5-3b-instruct-q6_k.gguf` 文件是否确实是 Q6_K。
- 8 线程在所有模型上都不是最佳选择，并且经常出现明显退化。

如果只根据当前 CPU 数据选默认配置：

| 目标 | 推荐配置 |
| --- | --- |
| 最高生成速度 | `qwen2.5-3b-instruct-q4_0.gguf` + `-t 4` |
| 速度与质量平衡 | `qwen2.5-3b-instruct-q4_k_m.gguf` + `-t 4` 或 `-t 6` |
| 更高量化精度但接受较慢速度 | `qwen2.5-3b-instruct-q5_k_m.gguf` 或 `qwen2.5-3b-instruct-q8_0.gguf` + `-t 6` |

## 4. 线程数规律

从两个日志合起来看，线程数规律比较一致：

- 2 线程通常已经能明显快于单线程。
- 4 线程是一个稳定高效的点。
- 6 线程通常能提高 prompt processing，也经常给出最高或接近最高的生成速度。
- 8 线程普遍不理想，尤其对 `tg128` 影响明显。

因此，当前 CPU 数据支持这样的经验规则：

```text
默认优先测试 -t 4 和 -t 6，不把 -t 8 作为默认最佳配置。
```

## 5. llama-bench-report.html

`llama-bench-report.html` 是 `simpleperf` 生成的 HTML 报告。它不是吞吐量日志，而是性能剖析报告，用来分析 CPU 时间花在了哪些函数、调用路径或共享库上。

这个文件后续适合重点查看：

- Flamegraph
- Functions
- Call graph
- Shared libraries
- `ggml_*`
- `llama_*`
- `matmul`
- `vec_dot`
- `quantize`
- `dequantize`

当前 Markdown 没有展开 HTML 里的具体函数占比，因为这些信息需要从报告页面中读取并摘录。建议后续把报告中的 Top functions、Top libraries 和主要火焰图结论整理成一个独立的 Markdown 摘要，避免关键分析只保存在 HTML 页面里。

## 6. 当前数据的注意点

当前 `data/cpu` 数据已经足够支撑 CPU 后端的初步结论，但还有几点会影响严谨性：

1. 日志中没有记录温度、电量、是否插电、手机壳、环境温度等信息。
2. 移动设备容易受温控影响，连续跑不同配置时，后跑的配置可能处在更热的状态。
3. 目前日志更像单轮 sweep，正式对比时最好每个配置重复多次。
4. `q6_k` 的日志显示结果与 Q5_K_M 过于接近，需要核对模型文件。
5. HTML profiling 报告已经存在，但其中的热点结论还没有被转写成可读的文本分析。

## 7. 总结

只看 `data/cpu`，当前最清楚的结论是：

- CPU 后端已经可以在 OnePlus 12 / Snapdragon 8 Gen 3 上流畅运行 Qwen2.5-3B-Instruct 3B 量级模型。
- `Q4_0` 速度最快，最佳生成速度约 21.67 tok/s。
- `Q4_K_M` 更均衡，最佳生成速度约 16.70 到 17.34 tok/s。
- 线程数推荐优先使用 4 或 6。
- 8 线程普遍退化，不建议作为默认配置。
- `llama-bench-report.html` 为后续函数级热点分析提供了基础，但还需要把 HTML 中的关键结论摘录出来。
