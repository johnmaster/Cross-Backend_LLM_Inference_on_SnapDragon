# Tiny LLM Block

本目录用于把仓库中的 MatMul、量化和 HTP profiling 实验连接到真实的
Transformer 数据流。

## 第一阶段：FP32 Reference

当前实现是一个纯 NumPy、Qwen 风格的单层 decoder block：

```text
input
  -> RMSNorm
  -> Q/K/V projection
  -> RoPE
  -> grouped-query causal attention
  -> output projection
  -> residual
  -> RMSNorm
  -> SwiGLU MLP
  -> residual
```

默认配置：

| 参数 | 值 |
|---|---:|
| hidden size | 256 |
| intermediate size | 768 |
| attention heads | 8 |
| KV heads | 2 |
| head dimension | 32 |
| RoPE theta | 1,000,000 |

张量约定：

```text
hidden state: [batch, sequence, hidden]
linear weight: [input, output]
KV cache:      [batch, kv_heads, sequence, head_dim]
```

实现同时支持整段 prefill 和带 KV cache 的增量 decode。随机数种子固定，因此
后续 ONNX、QNN builtin 和自定义 OpPackage 可以使用它作为稳定的正确性基线。

运行示例：

```bash
python -m tiny_llm_block.reference
```

运行测试：

```bash
python -m unittest tiny_llm_block.test_reference
```

测试覆盖：

- 输出和 KV cache shape
- prefill 与逐 token decode 数值一致性
- causal mask 保证未来 token 不影响过去输出
- 非法 hidden dimension 检查

## 后续阶段

## 第二阶段：固定 Shape ONNX

导出两张互相衔接的 FP32 ONNX 图：

```text
tiny_block_prefill_seq32.onnx
  input:  hidden_states [1, 32, 256]
  output: hidden_out, present_key, present_value

tiny_block_decode_past32.onnx
  input:  hidden_states [1, 1, 256]
          past_key/past_value [1, 2, 32, 32]
  output: hidden_out, present_key, present_value
```

导出需要默认环境中的 PyTorch，以及 `qairt-2.47` 环境中的 ONNX 和 ONNX
Runtime。当前机器可直接执行：

```bash
PYTHONPATH=/home/lingbok/anaconda3/envs/qairt-2.47/lib/python3.12/site-packages \
python -m tiny_llm_block.export_onnx
```

脚本会：

- 从 NumPy reference 复制完全相同的权重。
- 生成 `model/` 中的 prefill 和 decode ONNX。
- 生成 `test_data/` 中的 FP32 raw 输入、期望输出和 QNN input-list。
- 使用 ONNX checker 检查模型。
- 使用 ONNX Runtime 对齐 NumPy reference。

## 后续阶段

## 第三阶段：QNN FP32 HTP

QAIRT 2.47 converter 默认把 rank-3 hidden state 的外部接口从 NFC
`[batch, sequence, hidden]` 改成 NCF `[batch, hidden, sequence]`。raw 文件的
元素数量不变，因此错误输入不会触发 shape error，只会产生错误结果。转换时必须
保留 ONNX I/O layout：

```bash
qnn-onnx-converter \
  -i tiny_llm_block/model/tiny_block_prefill_seq32.onnx \
  -o tiny_llm_block/generated/tiny_block_prefill.cpp \
  --preserve_io layout

qnn-onnx-converter \
  -i tiny_llm_block/model/tiny_block_decode_past32.onnx \
  -o tiny_llm_block/generated/tiny_block_decode.cpp \
  --preserve_io layout
```

生成 Android model library：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
qnn-model-lib-generator \
  -c tiny_llm_block/generated/tiny_block_prefill.cpp \
  -b tiny_llm_block/generated/tiny_block_prefill.bin \
  -t aarch64-android \
  -l tiny_block_prefill \
  -o tiny_llm_block/model_libs
```

decode 使用相同命令，将 `prefill` 替换为 `decode`。

两张图均已在 OnePlus 12 / Snapdragon 8 Gen 3 的 HTP backend 上成功 finalize
并执行。FP32 HTP 输出与 NumPy reference 的误差：

| 模式 | 输出 | max error | mean error | cosine |
|---|---|---:|---:|---:|
| prefill | hidden | 0.0010000 | 0.0000666 | 0.999999881 |
| prefill | key | 0.0008549 | 0.0001221 | 0.999999940 |
| prefill | value | 0.0005050 | 0.0001066 | 1.000000000 |
| decode | hidden | 0.0004747 | 0.0001259 | 0.999999702 |
| decode | key | 0.0005240 | 0.0000486 | 1.000000000 |
| decode | value | 0.0003884 | 0.0000464 | 1.000000119 |

检查本地保存的三路设备输出：

```bash
python tiny_llm_block/compare_qnn_output.py \
  prefill tiny_llm_block/device_output/prefill/Result_0
python tiny_llm_block/compare_qnn_output.py \
  decode tiny_llm_block/device_output/decode/Result_0
```

profiling 使用 `burst`、detailed level、每张图 10 次 inference。排除第一次
warm-up 后取 median：

| 模式 | accelerator cycles | QNN accelerator | NetRun |
|---|---:|---:|---:|
| prefill seq=32 | 360,575 | 2,126 us | 3,691 us |
| decode token=1, past=32 | 149,241 | 1,850 us | 3,301 us |

```bash
python tiny_llm_block/compare_profile.py
```

decode 的 accelerator cycles 明显少于 prefill，但 QNN accelerator 和 NetRun
延迟下降有限，说明单 token 阶段已明显受到 KV I/O、layout conversion、调度和
小算子固定开销影响。

## 后续阶段

1. 比较 W8A8 与 W8A16 的逐层误差和 cycles。
2. 扩展 prefill sequence length 与 KV cache length sweep。
3. 将 Linear 逐步替换为仓库中的自定义 MatMul OpPackage。
