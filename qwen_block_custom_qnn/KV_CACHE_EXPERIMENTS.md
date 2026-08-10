# Qwen Block KV Cache Decode 实验
本文档记录 `qwen_block_custom_qnn` 中独立的 KV cache decode 实验，
包括 fixed-shape decode、grouped-GQA、宿主持久 cache、shared buffer、
fused attention 以及不同 graph boundary 的采用和未采用结论。

prefill q_proj custom-op 的主线说明见 [README.md](README.md)。


## 实验目标

Prefill 会一次处理多个 token，而 decode 每次只处理一个新 token。为了避免每一步
重复计算历史 token 的 K/V，本实验把上一时刻的 KV cache 作为 graph 输入，并输出
追加当前 token 后的新 cache：

```text
past_key/value [1, 2, 16, 64]
        │
        ├── concat current_key/value [1, 2, 1, 64]
        ▼
present_key/value [1, 2, 17, 64]
```

这里 `2` 是 Qwen2.5-0.5B 的 KV head 数，`64` 是每个 attention head 的维度。
当前 query token 使用绝对位置 `16` 的 RoPE，并且只能关注 past 16 个 token 和
当前 token。随后通过 GQA 把两个 KV heads 扩展给 14 个 query heads。

这次实验是 layer0 的单步 fixed-shape decode，用于验证 KV cache 数据流和比较
q_proj 性能；它还不是包含 24 层 cache、tokenizer、采样循环的完整文本生成程序。

## 输入输出和产物

```text
inputs:
  hidden_states [1, 1, 896]       float32
  past_key      [1, 2, 16, 64]    float32
  past_value    [1, 2, 16, 64]    float32

outputs:
  hidden_out    [1, 1, 896]       float32
  present_key   [1, 2, 17, 64]    float32
  present_value [1, 2, 17, 64]    float32
```

主要文件：

| 文件或目录 | 内容 |
|---|---|
| `model/qwen2_0_5b_layer0_decode_past16.onnx` | fixed past16、单 token decode ONNX |
| `test_data/layer0_decode_past16/` | hidden、past KV 和 PyTorch reference |
| `generated/qwen2_0_5b_layer0_decode_past16.cpp/.bin` | QNN builtin 转换结果 |
| `tools/patch_qwen_decode_q_proj_offline_q13.py` | 将 decode q_proj 替换为 custom Q13 op |
| `generated/qwen2_0_5b_layer0_decode_past16_q_proj_offline_q13.cpp` | custom decode QNN graph |
| `device_output/builtin_layer0_decode_past16/` | builtin 输出和 profiling |
| `device_output/custom_q13_layer0_decode_past16/` | custom 输出和 profiling |

导出命令：

```bash
PYTHONPATH=/home/lingbok/anaconda3/envs/qairt-2.47/lib/python3.12/site-packages \
python3 qwen_block_custom_qnn/tools/export_qwen_block_onnx.py \
  --seq-len 1 \
  --past-len 16 \
  --output qwen_block_custom_qnn/model/qwen2_0_5b_layer0_decode_past16.onnx \
  --test-data-dir qwen_block_custom_qnn/test_data/layer0_decode_past16
```

## 正确性

PyTorch 与 ONNX Runtime：

| 输出 | max abs error | mean abs error |
|---|---:|---:|
| `hidden_out` | 9.5367e-7 | 1.3708e-7 |
| `present_key` | 9.5367e-7 | 3.0824e-9 |
| `present_value` | 3.7253e-8 | 4.0596e-10 |

QNN custom 相对 FP32 reference：

```text
hidden_out cosine = 0.999997854
present_key/value 相对 builtin 逐 bit 相同
```

只替换 q_proj 不会改变 k_proj/v_proj，因此 custom 与 builtin 的 present KV 完全
一致；hidden 输出的差异来自 q_proj 的 device-Q13 数值路径，并继续经过 attention
和 MLP 传播。

## Decode 专用 M=1 内核

Prefill 的 q_proj 是 `M=16, K=N=896`，采用 4x128 micro-kernel。Decode 只有一行，
直接复用 4-row kernel 会浪费另外三行计算，因此增加了 `1x128` 分支：

```text
M=1:
  1 个 LHS Q13 row
  2 个 HVX_VectorPair accumulators
  每次同时计算 128 个输出列
  896 个输出列共 7 个 column tiles
```

decode q_proj 的 FP32 权重与 prefill q_proj payload 的 SHA256 完全一致，所以能够
复用 conversion probe 在目标设备导出的精确 Q13 权重。QHPI 的其他 worker slice
在 `M=1` 时不重复计算，由 slice 0 独占该行。

## Builtin 与 custom 性能

测试环境保持一致：

```text
Device: OnePlus 12 / Snapdragon 8 Gen 3
Backend: QAIRT/QNN 2.47, HTP v75
Mode: burst + detailed profiling
Runs: 10，去除首次运行后取中位数
```

| decode past16 | root cycles | QNN accelerator | QNN | NetRun |
|---|---:|---:|---:|---:|
| **builtin** | **1,712,987** | **8,738 us** | **10,954 us** | **11,007 us** |
| custom Q13 M=1 | 1,867,003 | 9,547 us | 11,010 us | 11,041 us |

q_proj 节点拆解：

| 路径 | cycles |
|---|---:|
| builtin fused q_proj | 26,065 |
| custom `1x128` q_proj | 72,112 |
| custom 独立 bias Add | 49,859 |
| custom q_proj + bias | 121,971 |

custom NetRun 比 builtin 慢 `34 us`，约 `0.31%`。虽然端到端差距不大，但 kernel
证据非常明确：M=1 是小 GEMV，builtin 的融合和调度开销更低；custom 路径还需要
单独执行 bias Add。

因此当前采用策略是：

```text
prefill seq16 q_proj -> custom device-Q13 4x128
decode token1 q_proj -> QNN builtin
```

KV cache decode 本身已经实现成功；未采用的是当前 custom decode q_proj 性能方案。
后续若继续优化，应优先研究 bias 融合、QHPI 调度成本和 decode 专用预打包 GEMV
布局，而不是继续扩大多行 tile。

## KV cache 优化 1：context sweep 与 QNN 边界布局修正

在优化 cache 前先增加 `past_len=16/32/64/128` 四组 builtin graph，建立上下文长度
基线。导出后均先通过 PyTorch 与 ONNX Runtime；设备结果由
`tools/summarize_decode_context_sweep.py` 统一取 10 次运行去首轮中位数。

过程中发现一个重要的 QNN converter 边界布局问题：

```text
past16 generated graph:
  past_key/value API shape = [1, 2, 16, 64]       NCHW

past32/64/128 generated graph:
  past_key/value API shape = [1, S, 64, 2]        NHWC
  present_key API shape    = [1, S+1, 64, 2]      NHWC
  present_value API shape  = [1, 2, S+1, 64]      NCHW
```

因此不能仅根据原 ONNX 的语义 shape 把同一种 NCHW raw 送给所有 QNN graph。
错误布局仍能成功执行，但会静默地产生错误 cache；最初 past32/64/128 的
`present_value` cosine 甚至只有 `0.0437/0.0160/0.0051`。

修正内容：

```text
tools/prepare_decode_qnn_io.py
  NCHW past K/V -> converter 实际要求的 NHWC raw

tools/compare_qnn_output.py --qnn-decode-layout
  converter 暴露的 present_key NHWC -> ONNX NCHW 后再比较
```

错误实验没有删除，保存在 `device_output/*layout_wrong/`；正式目录保存修正结果。
修正后的 past32/64/128：

| past length | hidden cosine | present-key cosine | present-value cosine |
|---:|---:|---:|---:|
| 32 | 0.999999583 | 0.999999762 | 1.000000000 |
| 64 | 0.999999762 | 1.000000119 | 1.000000119 |
| 128 | 0.999999821 | 1.000000000 | 1.000000000 |

性能基线：

| past length | root cycles | QNN accelerator | QNN | NetRun |
|---:|---:|---:|---:|---:|
| 16 | 1,712,987 | 8,738 us | 10,954 us | 11,007 us |
| 32 | 2,070,466 | 9,077 us | 11,828 us | 11,852 us |
| 64 | 2,140,213 | 9,193 us | 11,816 us | 11,844 us |
| 128 | 2,180,782 | 9,433 us | 12,007 us | 12,032 us |

完整机器可读汇总在
`device_output/decode_context_sweep_summary.csv`。不同 shape 会触发不同 QNN graph
rewrite，所以个别节点 cycles 不保证单调；判断优化必须在同一 past length 做 A/B。

## KV cache 优化 2：grouped-GQA 消除 K/V Tile

原始实现用 `repeat_interleave(7)` 把 2 个 KV heads 显式展开为 14 个 heads：

```text
K/V [1, 2, S, 64]
  -> Tile
K/V [1, 14, S, 64]
```

新 `--grouped-gqa` 路径把 Q reshape 成：

```text
Q [1, 2 KV groups, 7 query heads/group, 1, 64]
```

然后让 MatMul 在 KV-group 维广播 K/V，不物化七倍大小的 cache view。转换后的
past128 QNN C++ 中 `"Tile"` 节点数量从 `2` 变为 `0`。

导出命令：

```bash
PYTHONPATH=/home/lingbok/anaconda3/envs/qairt-2.47/lib/python3.12/site-packages \
python3 qwen_block_custom_qnn/tools/export_qwen_block_onnx.py \
  --seq-len 1 \
  --past-len 128 \
  --grouped-gqa \
  --output qwen_block_custom_qnn/model/qwen2_0_5b_layer0_decode_past128_grouped_gqa.onnx \
  --test-data-dir qwen_block_custom_qnn/test_data/layer0_decode_past128_grouped_gqa
```

past128 同 shape A/B：

| past128 | root cycles | QNN accelerator | QNN | NetRun |
|---|---:|---:|---:|---:|
| explicit K/V Tile | 2,180,782 | **9,433 us** | 12,007 us | 12,032 us |
| **grouped GQA** | **2,054,078** | 9,540 us | **11,806 us** | **11,862 us** |

attention 节点：

| past128 | QK | Softmax | AV | 合计 |
|---|---:|---:|---:|---:|
| explicit Tile | 124,608 | 76,797 | 82,025 | 283,430 cycles |
| grouped GQA | 35,069 | 26,360 | 48,304 | 109,733 cycles |

结果：

```text
三个 graph outputs：逐 bit 相同
attention cycles：下降 61.28%
root cycles：下降 5.81%
NetRun：12032 -> 11862 us，改善 170 us / 1.41%
```

QNN accelerator wall time有 `107 us` 反向波动，但 QNN、NetRun 和 accelerator
root cycles 均支持采用 grouped-GQA，且输出逐 bit 相同。采用结果保存在：

```text
device_output/builtin_layer0_decode_past128_grouped_gqa/
```

这是当前 KV-cache decode 的第一项正式采用优化。

## KV cache 优化 3：宿主维护 cache，只输出当前 K/V delta

完整 present-cache graph 每步输出：

```text
present_key/value [1, 2, 129, 64]
```

但真实生成循环只需要把当前 token 的 K/V 写入宿主持久 cache。新增
`--delta-kv-output` 后，attention 内部仍通过 past/current concat 读取完整上下文，
graph 边界只返回：

```text
current_key/value [1, 2, 1, 64]
```

宿主伪代码：

```text
hidden, delta_k, delta_v = decode(hidden, key_cache[:position], value_cache[:position])
key_cache[position] = delta_k
value_cache[position] = delta_v
position += 1
```

导出命令：

```bash
PYTHONPATH=/home/lingbok/anaconda3/envs/qairt-2.47/lib/python3.12/site-packages \
python3 qwen_block_custom_qnn/tools/export_qwen_block_onnx.py \
  --seq-len 1 \
  --past-len 128 \
  --grouped-gqa \
  --delta-kv-output \
  --output qwen_block_custom_qnn/model/qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv.onnx \
  --test-data-dir qwen_block_custom_qnn/test_data/layer0_decode_past128_grouped_gqa_delta_kv
```

边界 K/V payload 从：

```text
full present K/V: 132,096 bytes
current K/V only:   1,024 bytes
reduction:             129x
```

past128 grouped-GQA A/B：

| 输出策略 | QNN accelerator | QNN | NetRun | Output cycles |
|---|---:|---:|---:|---:|
| 完整 present K/V | 9,540 us | 11,806 us | 11,862 us | 41,007 |
| **current K/V delta** | **9,493 us** | **10,880 us** | **10,907 us** | **8,798** |

```text
QNN improvement:    926 us / 7.84%
NetRun improvement: 955 us / 8.05%
Output cycles:      下降 78.54%
```

交叉正确性不是只和 FP32 reference 比较：将完整-cache graph 的最后一个 K/V
位置取出，与 delta graph 输出比较，`hidden/current_key/current_value` 三者均
逐 bit 相同。

需要明确它优化了 graph 输出和 RPC/宿主搬运，并没有消除 attention 内部的
past/current Concat，也没有消除下一步输入 past cache 的传输。profiling 中
accelerator root cycles 从 `2,054,078` 波动到 `2,231,170`，与 QNN/NetRun wall
time方向相反；采用依据是逐 bit 正确、输出 payload 缩小 129 倍、Output cycles
和两级 wall time一致下降。

采用结果：

```text
device_output/builtin_layer0_decode_past128_grouped_gqa_delta_kv/
```

当前 decode 推荐组合：

```text
builtin projections
+ grouped-GQA attention
+ host-managed persistent KV cache
+ current-token KV delta output
```

下一步才是固定容量或共享 buffer 方案，目标是同时消除完整 past-cache 输入搬运和
内部 Concat；这需要验证 QNN persistent tensor、共享内存或 custom cache-update op，
不能把本轮 delta-output 误称为原地 cache 更新。

## KV cache 优化 4（未采用）：FP16 cache 边界

目标是保持 attention 计算为 FP32，只把 graph 边界的 past/current K/V 改成 FP16：

```text
past K/V input:     131,072 -> 65,536 bytes
current K/V output:   1,024 ->    512 bytes
```

导出器增加 `--fp16-kv-cache`。PyTorch 与 ONNX Runtime 校验通过；与 FP32
reference 相比，hidden 最大误差为 `3.75e-4`，cosine 约为 1。

QAIRT 2.47 converter 会把 ONNX FP16 Cast 在转换期解释，并将生成 QNN C++ 的
past/current KV API tensor 恢复成 `FLOAT_32`。因此只生成 FP16 raw 并不能得到
FP16 QNN graph，直接运行还会把一个输入文件误判成两个 batch。

为完成真实 FP16 边界实验，增加：

```text
tools/patch_qwen_decode_fp16_kv_boundary.py
```

它把 past K/V APP_WRITE 和 current K/V APP_READ 恢复成 `FLOAT_16`，并在
attention 侧插入 FP16→FP32 Cast。运行时必须指定：

```bash
--native_input_tensor_names \
qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv_fp16:past_key,past_value \
--output_data_type native_only
```

past128 grouped-GQA delta-KV A/B：

| KV 边界 | root cycles | QNN accelerator | QNN | NetRun | Output cycles |
|---|---:|---:|---:|---:|---:|
| **FP32** | 2,231,170 | 9,493 us | **10,880 us** | **10,907 us** | 8,798 |
| FP16 | **1,890,236** | **9,184 us** | 11,741 us | 11,767 us | **3,214** |

FP16 的 accelerator 路径有收益：

```text
root cycles:    下降 15.28%
accelerator us: 下降 3.26%
output cycles:  下降 63.47%
```

但端到端退化：

```text
QNN:    慢 861 us / 7.91%
NetRun: 慢 860 us / 7.88%
```

FP16 QNN 与 FP32 QNN 交叉比较：

```text
hidden_out:    逐 bit 相同
current_key:   max_abs=1.53e-5, cosine=1
current_value: max_abs=7.45e-9, cosine≈1
```

结论：数值和容量满足要求，但 QAIRT 2.47 当前的边界 Cast/native I/O 路径没有
形成端到端收益，因此不采用，正式基线继续使用 FP32 KV。负实验保存在：

```text
device_output/builtin_layer0_decode_past128_grouped_gqa_delta_kv_fp16/
```

FP16 仍可能在更长 context、24 层完整模型或共享内存消除 RPC copy 后变得有利，
届时应重新测试，不能直接套用本次 past128 单层结论。

## KV cache 优化 5（未采用）：QNN shared buffer 与 input cache

本地 QAIRT 2.47 能力检查确认：

```text
qnn-net-run --shared_buffer
QNN SampleAppSharedBuffer + memRegister
QNN HTP RPC shared-buffer memory
qnn-net-run --max_input_cache_tensor_sets
```

但 shared buffer 只改变 graph I/O 的内存注册方式，不会自动把本次
`current_key/value` 输出连接为下一次的 past-cache 输入，也不会修改 cache
position。因此先对完全相同的 past128 FP32 grouped-GQA delta graph做纯 I/O A/B。

第一次 10 次实验显示 shared buffer 更慢。为排除不同时间窗口的温度和电源波动，
随后在同一设备连续执行普通模式和 shared-buffer 模式各 30 次，再单独测试一个
input tensor-set cache。均采用去首轮中位数：

| I/O 模式 | root cycles | QNN accelerator | QNN | NetRun |
|---|---:|---:|---:|---:|
| **普通模式** | 2,350,585 | 9,654 us | **11,736 us** | **11,758 us** |
| `--shared_buffer` | 2,331,778 | 9,713 us | 12,028 us | 12,052 us |
| `--max_input_cache_tensor_sets 1` | **2,295,054** | **9,550 us** | 12,145 us | 12,175 us |

相对同窗口普通模式：

```text
shared buffer NetRun:     慢 294 us / 2.50%
input tensor cache NetRun:慢 417 us / 3.55%
```

两种模式的 accelerator cycles 没有恶化，甚至略低，但 QNN/NetRun wall time
增加，说明注册、同步或宿主 bookkeeping 成本超过了 past128 单层 cache 的复制收益。
结果不支持采用。

完整汇总：

```text
device_output/decode_io_mode_summary.csv
device_output/decode_delta_baseline_repeat30/
device_output/decode_delta_shared_repeat30/
device_output/decode_delta_input_cache_repeat30/
```

这项负实验也明确了 persistent cache 的边界：`--shared_buffer` 不是跨 execute 的
自动 KV feedback。真正的原地 cache 仍需要自定义 host runner 持有注册内存、更新
position，并让 graph 读取有效 cache；或者设计 custom attention/cache-update op。
考虑到 Concat 在当前 profile 中只有几千 cycles，而 MLP/attention 更大，不应为了
删除 Concat 立即引入高复杂度 runner。

## KV cache 优化 6（未采用）：固定 past128 的 decode attention 融合

本轮真正导出了并运行了一个 KV-cache decode fused graph，而不是只在文档中讨论。
融合边界为：

```text
输入：
  Q = [1, 14, 1, 64] FP32
  K = [1, 129, 64, 2] FP32（QNN NHWC）
  V = [1, 2, 129, 64] FP32

被融合：
  K Transpose/Unsqueeze
  QK MatMul
  / sqrt(64)
  causal mask Add
  Softmax
  V Unsqueeze
  attention-probability × V MatMul
  最后的 [2,7,1,64] -> [14,1,64] Reshape

输出：
  attention = [1, 14, 1, 64] FP32
```

`query_len=1` 且输入已包含 128 个 past token 和当前 token，因此 129 个位置全部
有效，不需要在内核中读取全零 causal mask。内核按 query head 分给 4 个 QHPI
worker，并用 online softmax 避免物化 `[14,129]` score/probability 中间张量。
KV delta 输出和宿主持久 cache 契约完全不变。

保存的实现与生成文件：

```text
tools/patch_qwen_decode_fused_attention.py
generated/qwen2_0_5b_layer0_decode_past128_fused_attention.cpp
model_libs/aarch64-android/libqwen2_0_5b_layer0_decode_past128_fused_attention.so
qnn_custom_ops/.../MatMulQhpiHvx8RowFp32StoreMultithread.cpp
```

图补丁会显式删除原 attention 节点的 graph construction 调用，而不是依赖 HTP
dead-code elimination；融合节点名为 `_FusedDecodeAttentionPast128`，op type 为
`Qwen2DecodeAttentionPast128Fp32`。

### 实验 6.1：标量 FP32 online softmax（当前保留的可读参考实现）

使用 `std::exp`，数值通过：

```text
hidden_out max_abs = 1.953125e-3
hidden_out mean_abs = 2.866132e-4
hidden_out cosine = 0.999999881
current_key/current_value = 逐 bit 相同
```

但融合节点约 `14.3M cycles`，平均 NetRun `22423 us`，明显慢于正式 builtin
delta-KV 基线的 `10907 us`。结果保存在：

```text
device_output/fused_attention_layer0_decode_past128/
```

### 实验 6.2：快速指数近似（未保留到主源码）

将 `exp(x), x<=0` 改成 `ln2` range reduction + 五阶多项式，`x<=-16` flush
为零；并根据 online-softmax 的 max 分支把每 token 两次 exp 降成一次。输出精度
与 6.1 基本相同，但融合节点反而约 `17.9M cycles`，NetRun `24158 us`。这说明
瓶颈不只是 libm 调用，大量标量 FP32 score/value 循环才是核心问题。

```text
device_output/fused_attention_fast_exp_layer0_decode_past128/
```

### 实验 6.3：HVX FP32 value 累加（未保留到主源码）

把 64 维 value accumulator 改为两个 128-byte HVX vector，融合节点降到约
`11.25M cycles`，NetRun 降到 `19432 us`，但仍远慢于 builtin；同时误差放大：

```text
hidden_out max_abs = 0.22265625
hidden_out mean_abs = 0.00644305
hidden_out cosine = 0.999941289
```

因此该版本既不满足性能目标，也不满足当前数值门槛，已回退：

```text
device_output/fused_attention_hvx_value_layer0_decode_past128/
```

正式结论：本次融合验证了 graph boundary 和 custom-op 工程链路，但不能采用。
builtin QK/Softmax/AV 在正式基线中合计只有约 `101030 cycles`，而第一版融合内核
中的标量 FP32 工作高出两个数量级。后续只有在完成“连续 K layout + HVX score
dot/reduction + 经验证的 HVX softmax + FP32 value FMA”后才值得复测；不能仅靠
减少 QNN 节点数期待加速。正式 decode 仍采用 past128 grouped-GQA delta-KV
builtin attention。

后续已将该实验抽离为独立的动态 KV、GQA-aware、32-token blockwise
online-softmax OpPackage，见
[`qnn_custom_ops/qwen_flash_attention_hvx`](../qnn_custom_ops/qwen_flash_attention_hvx/README.md)。
独立版本成功接回真实 graph，但当前标量 FP32 实现的 NetRun 仍为 `23338 us`，
hidden cosine 为 `0.999947424`，因此同样未采用。

独立版本随后复测了 HVX IEEE-FP AV：通过 `-mhvx-ieee-fp` 启用 v75 FP32
vector 指令，将 64 维 value accumulator 改成两个 128-byte HVX vector。
fused cycles 从 `17,618,383` 降至 `11,371,942`，root cycles 从
`19,508,660` 降至 `13,165,785`，NetRun 从 `23338 us` 降至 `19234 us`。
但 hidden 最大误差增至 `0.22265625`、cosine 为 `0.999941244`，而且仍比
正式 builtin `10907 us` 慢 `76.34%`，因此只保留为可复现实验，不采用。
详细构建错误、编译开关和 profile 见独立目录 README。

继续把 K boundary 改为 head-contiguous `[1,2,129,64]` 并实现 HVX QK 后，
fused attention 降至 `891176 cycles`，NetRun 降至 `11403 us`。其中 QNN
accelerator 为 `8910 us`，已经比 builtin 的 `9493 us` 快 `6.14%`；但端到端
仍比 builtin `10907 us` 慢 `496 us / 4.55%`，且 AV 最大误差仍为
`0.22265625`。因此该布局和内核作为有效实验保留，尚不替换正式 decode graph。

## KV cache 优化 7（采用为 runner 基线）：持久化 tensor 与宿主滑动 KV cache

相比继续调整失败的 fused attention，更有意义的下一步是模拟真实 autoregressive
decode：model/backend/context/graph 只初始化一次，输入输出 tensor 只分配一次，
每步生成的 K/V 写回同一块 cache buffer。

本项目在已有 QNN SampleApp 上新增：

```text
--persistent_decode_past128
```

实现位于：

```text
qnn_custom_ops/tools/qnn_sample_app_profile/src/QnnSampleApp.cpp
qnn_custom_ops/tools/qnn_sample_app_profile/src/QnnSampleApp.hpp
qnn_custom_ops/tools/qnn_sample_app_profile/src/main.cpp
```

执行流程：

```text
初始化 backend/context/graph（一次）
          |
分配并读取 hidden/past-K/past-V（一次）
          |
    graphExecute
          |
current K/V: [1,2,1,64] NCHW
          |
转成每 token [64,2] head-interleaved NHWC
          |
past cache 左移一个 token，并追加 current K/V
          |
下一次 graphExecute 复用相同 tensor 地址
```

当前 cache 是固定 128-token sliding window：

```text
past_key/past_value QNN buffer: [1,128,64,2] FP32 NHWC
每步删除最旧 token:             128 floats
每步追加 current K/V:           128 floats
```

runner 会输出 `persistent_decode_timings.csv`，分别记录：

```text
graph_execute_us
cache_update_us
step_total_us
```

100 步真机测试，去掉第 1 步：

| 指标 | 中位数 | 均值 | P90 | P95 |
|---|---:|---:|---:|---:|
| `graphExecute` | 4,988 us | 4,982.68 us | 5,609 us | 6,155 us |
| CPU cache update | 34 us | 64.65 us | 40 us | 51 us |
| execute + cache update | **5,037 us** | 5,047.47 us | 5,692 us | 6,404 us |

cache 更新中位数只占 step total 约 `0.68%`。第一步 step total 为 `6229 us`。
同一个输入的第一步输出与正式 delta-KV baseline 比较：

```text
hidden_out:    逐 bit 相同
current_key:   逐 bit 相同
current_value: 逐 bit 相同
```

另做相同设备、关闭 profiling、各执行 100 次的进程 wall-time 对比：

| runner | 100 次总 wall time |
|---|---:|
| qnn-net-run | 1.573442 s |
| persistent runner | **1.316746 s** |

persistent runner 总时间减少 `0.256696 s / 16.32%`。这里不能把 `5037 us`
直接与详细 profiling 下 qnn-net-run 的 `10907 us` 当作纯 HTP 加速：
`5037 us` 是应用内 `graphExecute + cache update`，不含初始化和逐步文件 I/O；
`10907 us` 是 NetRun 端到端 profile。100 次同配置 wall-time A/B 才是更公平的
runner 级证据。

一键复现：

```bash
cd qwen_block_custom_qnn
STEPS=100 bash tools/run_persistent_kv_runner_repro.sh
```

结果：

```text
device_output/persistent_runner_layer0_decode_past128_final100/
device_output/persistent_runner_layer0_decode_past128_repeat100/
```

限制必须明确：

1. 这是单个 layer0、固定 past128 的 sliding-window runner，不是完整 24 层生成。
2. 测试中 hidden input 保持不变；后续 cache 状态真实更新，但不代表一段真实文本的
   token-by-token hidden state。
3. cache 更新仍在 CPU 普通内存中；尚未使用 QNN `memRegister`/shared buffer，
   HTP 仍把 past K/V 当普通 graph input。
4. graph 内部仍存在 K/V `Concat`，还不是设备侧原地 cache update。

因此本版本采用为“persistent runner 基线”，不是最终 persistent KV 实现。下一步
应在这个 runner 中加入注册共享内存，再评估 HTP 可直接访问的 cache buffer；只有
共享内存正确性和 wall-time 都通过后，才考虑修改 graph/cache-update op。

## KV cache 优化 8（采用为可选模式）：persistent rpcmem + memRegister

在优化 7 的 runner 上继续加入 QAIRT SampleAppSharedBuffer 使用的标准路径：

```text
libcdsprpc.so
  ├── rpcmem_alloc
  ├── rpcmem_to_fd
  └── rpcmem_free
          |
Qnn_MemDescriptor_t(QNN_MEM_TYPE_ION)
          |
context memRegister
          |
QNN_TENSORMEMTYPE_MEMHANDLE
```

新增参数：

```text
--persistent_shared_buffer
```

shared 模式下所有 graph input/output 都由 rpcmem 分配并在 context 上注册一次，
100 步期间不反复注册；SampleApp 额外保存 tensor ID 到 host pointer 的映射，因此
CPU 仍可直接更新 past K/V。退出时按顺序执行 `memDeRegister` 和 `rpcmem_free`。

首先执行 3 步 smoke test，step 0 与普通 persistent/builtin baseline 比较：

```text
hidden_out:    逐 bit 相同
current_key:   逐 bit 相同
current_value: 逐 bit相同
```

由于预期收益小，测试没有只跑一次，而是执行两组夹心顺序，每组 100 步并去首步：

| 顺序 | 模式 | graphExecute 中位数 | cache update 中位数 | step total 中位数 |
|---:|---|---:|---:|---:|
| 1 | normal | 4938 us | 36 us | 4974 us |
| 2 | shared | 4891 us | 39 us | 4934 us |
| 3 | normal | 4920 us | 37 us | 4968 us |
| 4 | shared | 4896 us | 40 us | 4936 us |
| 5 | normal | 4954 us | 36 us | 4991 us |
| 6 | shared | 4919 us | 39 us | 4955 us |

三组中位数再取均值：

| 模式 | graphExecute | cache update | step total |
|---|---:|---:|---:|
| normal | 4937.33 us | **36.33 us** | 4977.67 us |
| shared | **4902.00 us** | 39.33 us | **4941.67 us** |

shared 的 graphExecute 改善约 `0.72%`，step total 改善约 `0.72%`；CPU 对 rpcmem
cache 的更新慢约 `3 us`，但 graphExecute 降低约 `35 us`，最终仍有小幅净收益。
两种正反顺序的 6 组结果方向一致，因此不是单次偶然波动，但收益很小，不能扩大解释。

机器可读结果：

```text
device_output/persistent_shared_ab100/normal_a.csv
device_output/persistent_shared_ab100/shared.csv
device_output/persistent_shared_ab100/normal_b.csv
device_output/persistent_shared_ab100/shared_a_reverse.csv
device_output/persistent_shared_ab100/normal_reverse.csv
device_output/persistent_shared_ab100/shared_b_reverse.csv
device_output/persistent_shared_ab100/summary.csv
```

复现脚本默认打开 shared：

```bash
STEPS=100 SHARED_BUFFER=1 bash tools/run_persistent_kv_runner_repro.sh
```

普通模式 A/B：

```bash
STEPS=100 SHARED_BUFFER=0 bash tools/run_persistent_kv_runner_repro.sh
```

结论：shared persistent 模式采用为长序列 runner 的推荐选项，但保留普通模式作为
基线和兼容 fallback。它仍没有消除 graph 内部 Concat，也没有让 custom op 在 HTP
内部原地更新 cache；下一阶段若继续，应修改 graph contract，而不是继续优化这
`0.7%` 的 host memory 路径。

## KV cache 优化 9（未采用）：NCHW value-cache graph boundary

在考虑 destructive cache-update op 前，先用正式 profile 计算收益上限：

```text
K Concat 中位数: 3677 cycles
V Concat 中位数: 1795 cycles
合计:             5472 cycles，约占 root cycles 0.25%
```

因此去 Concat 的潜在收益过小，不足以承担 input/output alias 和 destructive
kernel 的正确性风险。profile 中更大的边界节点是：

```text
past_value_nchw Transpose: 25203 cycles
```

本轮改为验证更安全的 graph contract：

```text
旧 value cache input: [1,128,64,2] QNN NHWC
  -> graph 内 Transpose
  -> [1,2,128,64] NCHW

实验 value cache input: [1,2,128,64] NCHW
  -> 直接送入 V Concat
```

实现和输入：

```text
tools/patch_qwen_decode_nchw_value_cache.py
tools/device_input_list_layer0_decode_past128_nchw_value_cache.txt
generated/qwen2_0_5b_layer0_decode_past128_nchw_value_cache.cpp
model_libs/aarch64-android/libqwen2_0_5b_layer0_decode_past128_nchw_value_cache.so
```

persistent runner 同时增加 shape 检测。NCHW V cache 按两个 head 分别左移
`127×64` floats 并追加 current value；旧 NHWC 路径保持不变。

正确性 smoke test：

```text
hidden_out/current_key/current_value 全部逐 bit 相同
```

但 shared persistent 条件下交替执行旧图/NCHW 图各两次、每次 100 步后：

| 模式 | repeat | graphExecute | cache update | step total |
|---|---|---:|---:|---:|
| 旧 NHWC V boundary | A | 4918 us | 39 us | 4960 us |
| NCHW V boundary | A | 4926 us | 41 us | 4978 us |
| 旧 NHWC V boundary | B | 4882 us | 39 us | 4919 us |
| NCHW V boundary | B | 4938 us | 40 us | 4999 us |
| 旧图中位数均值 |  | **4900 us** | **39 us** | **4939.5 us** |
| NCHW 中位数均值 |  | 4932 us | 40.5 us | 4988.5 us |

NCHW step total 反而慢 `49 us / 0.99%`。

同窗口 detailed profile 确认 `past_value_nchw` 已完全消失，但 V Concat 从
`3592` 增至 `16323 cycles`，说明 HTP 为新的 boundary/layout 在下游 Concat
选择了更不利的转换或图优化。端到端结果：

| 图 | root cycles | QNN accelerator | QNN | NetRun |
|---|---:|---:|---:|---:|
| 旧 NHWC V | 2,246,194 | 9,535 us | **11,112 us** | **11,133 us** |
| NCHW V | 2,343,297 | **9,469 us** | 11,648 us | 11,685 us |

虽然 accelerator wall time略降，root/QNN/NetRun 均恶化，最终不采用。正式
persistent shared runner 继续使用旧 NHWC value-cache boundary。

结果：

```text
device_output/nchw_value_cache_ab100/
device_output/nchw_value_cache_ab100/summary.csv
device_output/nchw_value_cache_profile/
device_output/old_value_cache_profile_same_window/
```
