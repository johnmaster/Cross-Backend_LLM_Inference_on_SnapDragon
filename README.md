# Cross-Backend LLM Inference on Snapdragon

本仓库记录在 OnePlus 12（Snapdragon 8 Gen 3）上进行本地 LLM 推理、QNN
量化和 HTP Custom Op 优化的完整实验过程。项目不只比较不同 backend 的速度，
还覆盖从 NumPy/PyTorch reference、ONNX 导出、QNN 转换，到 Hexagon/HVX kernel、
Android 真机运行、正确性校验和 profiling 的工程闭环。

当前仓库有两条相互关联的主线：

1. `llama.cpp`：比较 Qwen2.5 GGUF 模型在 CPU 和 Adreno OpenCL GPU 上的
   prefill/decode 性能。
2. QNN/HTP：学习量化，开发 custom MatMul OpPackage，在 tiny Transformer block
   中验证替换方法，最终迁移到真实 Qwen2.5-0.5B decoder layer。

```text
llama.cpp CPU/OpenCL benchmark
                |
                v
QNN quantization basics and converter experiments
                |
                v
standalone QNN HTP custom MatMul
                |
                v
tiny Qwen-style block: builtin -> custom q_proj
                |
                v
real Qwen2.5-0.5B layer0: builtin -> custom q_proj
                |
                v
device-Q13 prefill projection -> faster than builtin
                |
                v
KV-cache decode -> grouped GQA -> delta KV
                |
                v
persistent runner -> rpcmem/memRegister shared cache
```

## 当前完成状态

| 阶段 | 状态 | 主要结果 |
|---|---|---|
| CPU/OpenCL benchmark | 已完成 | CPU 更适合当前 decode；OpenCL 可改善部分模型的 prefill |
| QNN 量化实验 01–11 | 已完成 | 覆盖 INT8、per-axis、QDQ、误差拆分、W8A16、校准和 W4FP16a |
| 独立 HTP Custom MatMul | 已完成 | 从标量 reference 逐步发展到 QHPI/HVX 8-row、FP32 store、prepack 和 tile-cache |
| Tiny Qwen-style block | 已完成 | prefill/decode ONNX、QNN builtin、设备正确性和 profiling 均已跑通 |
| Tiny block custom q_proj | 已完成 | 成功 patch converter 生成的 QNN C++ 并加载外部 HTP OpPackage |
| 真实 Qwen2.5 layer0 prefill | 已完成 | device-Q13 4x128 custom q_proj 保持逐 bit 输出，NetRun `13105 us`，比 builtin 快 `11.61%` |
| 真实 Qwen2.5 layer0 decode | 已完成 | past16/32/64/128、grouped-GQA、host-managed delta KV 和正确性/profile 均已完成 |
| Persistent KV-cache runner | 已完成 | 一次初始化/分配后连续 100 步；shared MEMHANDLE 模式 step-total 中位数均值 `4941.67 us` |

## 测试平台

| 项目 | 环境 |
|---|---|
| 手机 | OnePlus 12 |
| SoC | Snapdragon 8 Gen 3 |
| CPU | Qualcomm Kryo |
| GPU | Adreno 750 |
| HTP | Hexagon v75 |
| 系统与连接 | Android、Termux、ADB |
| QNN/QAIRT | `2.47.0.260601` |
| Hexagon SDK | `5.5.5.0` |
| Hexagon Tools | `8.7.06` |
| Android NDK | r28 |
| CPU profiling | `simpleperf` |
| QNN profiling | `qnn-net-run` + `qnn-profile-viewer` |

这些版本是本仓库已经验证过的组合。不同 QAIRT、Hexagon SDK 或 NDK 版本可能在
converter 输出、OpPackage API、编译参数和 HTP graph optimization 上表现不同。

## 仓库结构

```text
.
├── data/                         # llama.cpp CPU/GPU benchmark 原始日志与报告
├── docs/
│   ├── analysis.md               # CPU 与 OpenCL benchmark 分析
│   ├── oneplus12-llm-startup.md  # OnePlus 12 环境搭建和 profiling 记录
│   ├── QNN_learning/             # QNN SDK、HTP 架构、SampleApp、Custom Op 笔记
│   └── ai-hub/                   # Qualcomm AI Hub 使用笔记
├── qnn_quantization/             # 11 个递进的 QNN 量化实验
├── qnn_custom_ops/               # standalone QNN/QHPI/HVX MatMul OpPackage 实验
│   └── qwen_flash_attention_hvx/ # Qwen GQA blockwise online-softmax 实验
├── tiny_llm_block/               # 手写 Qwen-style block 和 QNN builtin baseline
├── tiny_llm_block_custom_matmul/ # tiny block projection custom-op 替换
├── qwen_block_custom_qnn/        # 真实 Qwen2.5-0.5B layer0 custom QNN 案例
└── scripts/                      # llama.cpp benchmark 与 QNN throughput 脚本
```

各目录中的 README 是具体实验的可复现文档；根 README 只提供全局入口和关键结论。

## 实验导航与复现入口

### 1. Snapdragon 上的 `llama.cpp`

- [设备环境与启动记录](docs/oneplus12-llm-startup.md)
- [CPU 与 OpenCL 结果分析](docs/analysis.md)
- `data/cpu/`、`data/gpu/` 中的原始 benchmark 日志
- `scripts/run_baseline.sh`、`run_quant_sweep.sh` 和
  `run_opencl_ngl_sweep.sh`

### 2. QNN 量化

从 [qnn_quantization/README.md](qnn_quantization/README.md) 开始，按编号阅读：

| 目录 | 主题 |
|---|---|
| `01_int8_per_tensor` | symmetric/asymmetric INT8、scale、offset、zero point |
| `02_qnn_quantize_dequantize` | 在真实 QNN HTP 图中验证 Quantize/Dequantize |
| `03_int8_per_axis` | per-tensor 与 per-axis weight 量化 |
| `04_qnn_int8_per_axis` | QNN per-axis encoding 和 HTP 支持边界 |
| `05_quantized_matmul` | FP32、INT8 per-tensor、INT8 per-axis MatMul 对比 |
| `06_onnx_qdq_matmul` | ONNX/QDQ/converter 到 HTP 的完整路径 |
| `07_quantization_error_decomposition` | weight、activation、output requantization 误差拆分 |
| `08_w8a16_matmul` | W8A8 与 W8A16 mixed precision |
| `09_calibration_sensitivity` | calibration 范围和离群值敏感性 |
| `10_percentile_calibration` | min-max 与 percentile calibration |
| `11_w4fp16_blockwise_matmul` | W4FP16a block quantization 和 converter 支持边界 |

### 3. HTP Custom MatMul

[qnn_custom_ops/matmul/README.md](qnn_custom_ops/matmul/README.md) 从零记录普通 QNN
HTP Custom OpPackage 的生成、编译、部署和验证。之后的 `matmul_qhpi_*` 目录按单一
变量逐步演进：

```text
QHPI scalar reference
-> FP16 scalar
-> HVX Q13 MatMul
-> 4-row / 8-row RHS reuse
-> vector conversion
-> direct FP32 vector store
-> QHPI self-slicing
-> LHS/RHS prepack
-> fused LHS tile-cache
```

其中：

- `matmul_qhpi_scalar` 和 `matmul_qhpi_fp16_scalar` 是正确性/API baseline。
- `matmul_qhpi_hvx` 开始使用真实 HVX SIMD 指令。
- `matmul_qhpi_hvx_multi_row*` 和 `matmul_qhpi_hvx_8row*` 分别验证 tile、转换、
  store、线程和 prepack 策略。
- `matmul_qhpi_hvx_8row_fp32_store_multithread` 当前同时承载 prefill
  device-Q13 projection 实验和 decode attention 负实验所需的 QHPI op。

这里必须区分“被调度到 HVX worker”和“真正使用 HVX SIMD”；同样，外部开发环境
缺少 HMX tile/layout 控制细节，本仓库没有把普通标量或单条探测指令误称为 HMX
MatMul。

### 4. Tiny Qwen-style Transformer Block

[tiny_llm_block/README.md](tiny_llm_block/README.md) 使用固定随机权重实现缩小版
Qwen-style decoder block，保留：

```text
RMSNorm -> GQA attention -> RoPE -> residual
        -> RMSNorm -> SwiGLU MLP -> residual
```

它包含 `hidden_size=256`、8 个 Query heads、2 个 KV heads，并分别导出：

- prefill：`[1, 32, 256]`
- decode：单 token 输入加长度为 32 的 past KV cache

NumPy、PyTorch、ONNX Runtime 和 QNN HTP 使用同一套权重与输入，使它成为
custom-op 图替换前的稳定 reference。

[tiny_llm_block_custom_matmul/README.md](tiny_llm_block_custom_matmul/README.md)
进一步记录：

```text
converter-generated QNN C++
-> 定位 q_proj FullyConnected/MatMul
-> 插入 Cast/Reshape
-> 替换为外部 HTP Custom Op
-> 编译 model library
-> Android qnn-net-run
-> correctness/profile 对比
```

当前 tiny block 中效果最好的 custom q_proj 是 LHS tile-cache 版本：

```text
custom_lhs_tile_cache:
root_cycles=480235
qnn_us=4821
netrun_us=6238
q_proj_cycles=146244
```

它将早期独立 LHS-prepack 版本约 `1.34M` 的 custom q_proj cycles 降到约
`146K`。端到端仍未超过 QNN builtin，瓶颈已经从 kernel 内部扩展到 Cast/Reshape、
外部 OpPackage dispatch、layout 转换和静态 weight 预处理。

### 5. 真实 Qwen2.5-0.5B Decoder Layer

[qwen_block_custom_qnn/README.md](qwen_block_custom_qnn/README.md) 是当前完整度最高的
案例。它直接读取本地 `Qwen/Qwen2.5-0.5B-Instruct` 的 `config.json` 和
`model.safetensors`，手写真实 layer 0 的：

```text
RMSNorm
-> q_proj/k_proj/v_proj
-> RoPE + 14-query-head / 2-KV-head GQA
-> causal attention + o_proj + residual
-> RMSNorm
-> SwiGLU(gate_proj/up_proj/down_proj)
-> residual
```

prefill 固定输入为 `[1,16,896]`，decode 使用 token=1 和真实 KV cache。
当前已经完成：

1. 真实权重下载与文件说明。
2. PyTorch layer0 prefill 和固定 shape ONNX 导出。
3. ONNX Runtime 与 PyTorch 对齐。
4. ONNX -> QNN C++/bin -> Android model library。
5. OnePlus 12 HTP builtin baseline。
6. patch converter 生成的 `_MatMul`，替换真实 `q_proj`。
7. multithread + LHS tile-cache，并针对 `M=16` 验证 4-row 分工。
8. 用独立 HTP conversion probe 导出设备实际的 FP16-to-Q13 INT16 权重。
9. 离线嵌入 device-Q13 RHS，删除 q_proj runtime Cast。
10. 将 device-Q13 kernel 从 4x64 扩展到 4x128。
11. 删除最终 BIN 中不再引用的 FP32 q_proj payload。
12. 导出真实 KV-cache decode graph，并建立 past16/32/64/128 sweep。
13. 用 grouped-GQA 消除显式 K/V Tile。
14. 改为 graph 只输出 current K/V delta，宿主维护 persistent cache。
15. 实现一次初始化、一次 tensor 分配的连续 decode runner。
16. 为 runner 加入 rpcmem + `memRegister` shared MEMHANDLE。
17. 验证并记录 FP16 KV boundary、fused attention、I/O cache 和 NCHW V boundary
    等未采用方案。

#### Prefill：device-Q13 custom q_proj 已超过 builtin

10 次 inference、去 warmup 中位数：

| 版本 | q_proj cycles | NetRun |
|---|---:|---:|
| QNN builtin | graph optimizer 内部实现 | `14826 us` |
| 初始 custom q_proj | 较高 | `16310 us` |
| multithread 8-row | 改善 | `15861 us` |
| device-Q13 4x64 | `826368` | `13152 us` |
| **device-Q13 4x128** | **`582392`** | **`13105 us`** |

最终 4x128 版本相对 builtin 快 `1721 us / 11.61%`，且输出与先前 4-row custom
逐 bit 相同。移除未使用 FP32 q_proj payload 后，model library 另减少
`3,211,536 bytes`。

#### Decode：采用 builtin projection + grouped-GQA delta KV

decode 中 `M=1`，复用面向 prefill 的 device-Q13 kernel 没有收益：

```text
past16 builtin decode:          11007 us
past16 device-Q13 q_proj:       11041 us（未采用）
```

past128 的 adopted graph 依次完成：

```text
普通 GQA
-> grouped-GQA 删除显式 K/V Tile
-> graph 只输出 current K/V delta
-> host 维护固定窗口 cache
```

| past128 版本 | NetRun | 结论 |
|---|---:|---|
| 普通 decode | `12032 us` | 初始基线 |
| grouped-GQA | `11862 us` | 输出逐 bit 相同 |
| **grouped-GQA + delta KV** | **`10907 us`** | 正式 decode graph |

delta-KV 相对 full present K/V 改善 `955 us / 8.05%`，并将 K/V 输出 payload
缩小 129 倍。

#### Persistent runner 与共享 cache

专用 QNN C++ runner 让 backend/context/graph 和 tensor 只创建一次，连续执行
固定 past128 sliding-window decode；每步把 current K/V 写回同一 cache buffer。
step 0 的 hidden/current-K/current-V 与正式 baseline 全部逐 bit相同。

100 步去首步的普通 persistent runner：

```text
graphExecute median:       4988 us
CPU cache update median:     34 us
step total median:         5037 us
```

关闭 profiling 的相同 100 次进程 wall time 从 qnn-net-run 的 `1.573442 s`
降至 persistent runner 的 `1.316746 s`，改善 `16.32%`。加入 rpcmem +
`memRegister` 后，两组正反夹心共 6 次测试的 step-total 中位数均值从
`4977.67` 降至 `4941.67 us`，再改善约 `0.72%`。

这里必须区分不同计时口径：`5037 us` 是应用内 `graphExecute + cache update`，
不含初始化和逐步文件 I/O；与 qnn-net-run 的公平 runner 级证据是相同 100 次的
进程 wall-time A/B。

#### 已验证但未采用

| 实验 | 结果 | 原因 |
|---|---|---|
| fused-bias q_proj | 更慢且误差增大 | 外部 Add 删除没有转化为端到端收益 |
| FP16 KV boundary | `10907 -> 11767 us` | Cast/native-I/O 成本超过带宽收益 |
| qnn-net-run shared/input cache | 慢 `2.50%/3.55%` | 注册、同步或 bookkeeping |
| custom fused decode attention | head-contiguous HVX QK+AV `11403 us` | accelerator 快于 builtin `6.14%`，但端到端仍慢 `4.55%` 且最大误差为 `0.22265625` |
| NCHW value-cache boundary | persistent step 慢 `0.99%` | 下游 V Concat/layout rewrite 变差 |

当前结论不是“所有 custom op 都更快”，而是：

- prefill 的固定 `M=16` q_proj 采用 device-Q13 4x128 custom kernel；
- decode 的 `M=1` projection 和 attention 保留 QNN builtin；
- decode 系统层收益主要来自 grouped-GQA、delta KV、persistent tensor 和 shared
  registered memory；
- 所有正优化和负优化都保留了生成文件、真机输出、CSV profile 与复现说明。

这项结果的价值不仅是 kernel 快慢，还在于完整验证了：

```text
真实 checkpoint -> Transformer block -> ONNX -> QNN graph
-> converter C++ patch -> external HTP OpPackage
-> 真机执行 -> 数值误差 -> graph/op profiling -> 瓶颈解释
```

## 关键性能结论

### `llama.cpp` CPU 与 OpenCL

| Model | Backend | Config | pp512 tok/s | tg128 tok/s |
|---|---|---|---:|---:|
| Q4_0 | CPU | 6 threads | 107.03 | 21.18 |
| Q4_0 | CPU | 4 threads | 87.94 | 21.67 |
| Q4_K_M | CPU | 6 threads | 61.24 | 16.70 |
| Q4_0 | OpenCL | `ngl=99`, 4 threads | 116.45 | 14.05 |
| Q6_K | OpenCL | `ngl=99`, 4 threads | 51.57 | 7.27 |
| Q8_0 | OpenCL | `ngl=99`, 4 threads | 65.06 | 9.06 |

- 当前最快 decode 是 CPU Q4_0、4 threads，约 `21.67 tok/s`。
- OpenCL Q4_0 将最佳 prefill 从 `107.03` 提高到 `116.45 tok/s`。
- 当前所有已测模型的 OpenCL decode 都慢于 CPU，说明短序列、逐 token kernel
  launch、同步和数据移动开销不能被并行计算收益抵消。

### Custom Op profiling 的解释原则

QNN builtin 的原始 projection 可能被 graph optimizer 融合、改写或预打包，CSV 中
某些 builtin 节点甚至显示为 `0 cycles`。这不表示计算免费，也不能与外部 custom op
节点的 cycles 做一比一比较。本仓库统一使用以下优先级：

1. graph-level `root_cycles`、`qnn_us`、`netrun_us` 判断端到端结果。
2. custom-op cycles 定位自定义 kernel 内部瓶颈。
3. builtin subevent cycles 只用于辅助定位，不作为绝对耗时。

## 常用脚本

CPU thread sweep：

```bash
bash scripts/run_baseline.sh
```

CPU quantization sweep：

```bash
bash scripts/run_quant_sweep.sh
```

OpenCL `ngl` sweep：

```bash
adb push scripts/run_opencl_ngl_sweep.sh /data/local/tmp/llama-opencl/
adb shell chmod +x /data/local/tmp/llama-opencl/run_opencl_ngl_sweep.sh
adb shell /data/local/tmp/llama-opencl/run_opencl_ngl_sweep.sh
adb pull /data/local/tmp/llama-opencl/opencl_ngl_sweep.log data/gpu/opencl_ngl_sweep.log
```

QNN throughput sweep 入口：

```bash
bash scripts/run_qnn_throughput_all.sh
```

QNN/HTP 的具体转换、编译、`adb push`、`qnn-net-run` 和 profile 命令较长，并且
与实验目录和 OpPackage 名称绑定，因此保留在各子目录 README 中。

## 产物与可复现性

仓库保留：

- 源代码、patch/分析脚本和实验 README。
- 小型 raw 输出、`qnn-profiling-data_0.log` 和 `profile.csv`。
- `llama.cpp` benchmark 原始日志。

以下大文件通常由 `.gitignore` 排除，需要按子目录 README 重新生成或自行准备：

- Qwen checkpoint 和 Hugging Face cache。
- ONNX 模型。
- converter 生成的大型 QNN C++/bin。
- Android model library。
- HTP/ARM64 OpPackage build 目录。

真实 Qwen 模型默认放置在：

```text
qwen_block_custom_qnn/model/data/models/Qwen2.5-0.5B-Instruct/
```

模型权重受其自身许可证约束，不随本仓库提交。运行 QNN/HTP 实验还需要单独安装
QAIRT SDK、Hexagon SDK/Tools、Android NDK，并在设备上准备相应 QNN runtime 库。

## 文档索引

- [OnePlus 12 本地 LLM 环境搭建](docs/oneplus12-llm-startup.md)
- [CPU 与 OpenCL benchmark 分析](docs/analysis.md)
- [QNN 学习笔记](docs/QNN_learning/qnn-learning.md)
- [Qualcomm HTP 架构笔记](docs/QNN_learning/architecture/qualcomm_hexagon_htp.md)
- [QNN 量化实验](qnn_quantization/README.md)
- [Tiny Qwen-style block](tiny_llm_block/README.md)
- [Tiny block custom MatMul](tiny_llm_block_custom_matmul/README.md)
- [真实 Qwen2.5 block custom QNN](qwen_block_custom_qnn/README.md)
- [真实 Qwen2.5 block 优化实验日志](qwen_block_custom_qnn/OPTIMIZATION_LOG.md)
- [Qwen GQA FlashAttention QHPI/HTP](qnn_custom_ops/qwen_flash_attention_hvx/README.md)
