# Qwen Block Custom QNN/HTP Case Study

本目录用于把 `tiny_llm_block_custom_matmul` 里验证过的 QNN/HTP custom op
链路迁移到真实 Qwen-family transformer block 上，形成一个可复现的工程验证案例。

一句话目标：

```text
从真实 Qwen block 中定位 attention/MLP projection，
把其中一个或多个 FullyConnected/MatMul 替换为自定义 QNN HTP OpPackage，
在 Snapdragon 设备上完成正确性、性能和瓶颈分析闭环。
```

## 复现顺序总览

从当前仓库状态复现这条链路，顺序是：

```text
1. 准备真实 Qwen2.5-0.5B-Instruct 权重
2. 导出 layer0 prefill 固定 shape ONNX
3. 用 ONNX Runtime 对齐 PyTorch 手写 reference
4. 用 qnn-onnx-converter 生成 QNN C++/bin
5. 用 qnn-model-lib-generator 生成 Android model lib
6. adb 推送 model lib 和 input raw 到设备
7. qnn-net-run 在 HTP 上跑 builtin baseline
8. qnn-profile-viewer 生成 profile.csv
9. compare_qnn_output.py 检查数值
10. summarize_profile.py 汇总性能指标
11. patch q_proj 为 custom HTP MatMul
12. 重新编译 model lib，加载 custom OpPackage 运行
13. 对比 builtin/custom 的正确性和 profiling
14. 用独立 HTP probe 导出设备实际的 FP16-to-Q13 INT16 权重
15. 把 device-Q13 权重嵌回 QNN model BIN，并删除 q_proj RHS runtime Cast
16. 验证 device-Q13 graph outputs 与原 custom 逐 bit 一致
17. 重新执行 10 次 detailed profiling
18. 用 run_device_q13_repro.sh 固化完整复现流程
19. 将 device-Q13 kernel 扩展为 4x128，并验证逐 bit 输出
20. 从最终 BIN 删除不再使用的 FP32 q_proj payload
```

当前已经完成第 1 到第 20 步。最终 device-Q13 `4x128` custom 版本不仅保持了原
custom 的逐 bit 输出，还在同一设备上实现了端到端快于 QNN builtin。

## M0. 先拿到真实 Qwen 模型

模型快照固定在：

```text
qwen_block_custom_qnn/model/data/models/Qwen2.5-0.5B-Instruct/
```

如果需要重新下载，可以使用：

```bash
mkdir -p qwen_block_custom_qnn/model/data/models

huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir qwen_block_custom_qnn/model/data/models/Qwen2.5-0.5B-Instruct \
  --local-dir-use-symlinks False
```

如果当前网络不能直接访问 Hugging Face，可以用 ModelScope 或离线拷贝，但最终目录最好
保持同样结构，让后续脚本只依赖一个本地路径：

```text
qwen_block_custom_qnn/model/data/models/Qwen2.5-0.5B-Instruct/config.json
qwen_block_custom_qnn/model/data/models/Qwen2.5-0.5B-Instruct/model.safetensors
qwen_block_custom_qnn/model/data/models/Qwen2.5-0.5B-Instruct/tokenizer.json
```

当前模型 config：

```text
model_type: qwen2
hidden_size: 896
intermediate_size: 4864
num_hidden_layers: 24
num_attention_heads: 14
num_key_value_heads: 2
head_dim: 64
rope_theta: 1000000.0
torch_dtype: bfloat16
```

### 下载目录中的文件说明

`model/data/` 下的内容可以分成模型快照和下载缓存两部分：

```text
model/data/
├── models/Qwen2.5-0.5B-Instruct/  # 后续导出和推理实际使用的模型快照
├── .hf_cache/version.txt          # huggingface_hub 本地缓存格式版本
└── .hf_home/hub/version.txt       # Hugging Face Hub 缓存格式版本
```

两个 `version.txt` 都只是下载工具产生的缓存元数据，不包含模型参数，也不是导出
ONNX 的输入；离线拷贝模型时只复制 `models/Qwen2.5-0.5B-Instruct/` 即可。

模型快照目录中的文件用途如下：

| 文件 | 用途 | 本案例是否直接使用 |
|---|---|---|
| `model.safetensors` | 模型参数文件，包含 embedding、24 个 decoder layer 和最终 RMSNorm 等权重；该模型通过 `tie_word_embeddings: true` 让输出层复用 embedding 权重，因此没有单独的 `lm_head` 张量 | 是，导出单层 ONNX 时从中读取 layer 0 权重 |
| `config.json` | 模型结构配置，包括 hidden size、MLP intermediate size、层数、attention/KV head 数、RoPE 参数、词表大小和数据类型等 | 是，用于按正确形状和超参数构造 Qwen2 decoder block |
| `tokenizer.json` | Hugging Face fast tokenizer 的完整序列化文件，包含词表、BPE merge 规则、normalizer/pre-tokenizer、特殊 token 等 | 当前固定张量的单层导出不用；做文本输入或完整生成时使用 |
| `tokenizer_config.json` | tokenizer 的行为配置，包括特殊 token、chat template、是否添加 BOS、最大长度等 | 当前不用；对话 prompt 格式化和文本 tokenization 时使用 |
| `vocab.json` | Qwen byte-level BPE 的 token 到 token ID 映射 | `tokenizer.json` 的拆分版组成部分，慢速/兼容 tokenizer 可能使用 |
| `merges.txt` | BPE 子词合并规则及优先级 | 与 `vocab.json` 配套；决定文本如何逐步合并为 token |
| `generation_config.json` | 默认文本生成参数，例如 EOS/PAD token、temperature、top-p、top-k 和 repetition penalty | 单层 ONNX 导出不用；完整自回归生成时可作为默认参数 |
| `README.md` | Hugging Face 模型卡，说明模型简介、用法、评测和注意事项 | 文档参考，不参与加载或导出 |
| `LICENSE` | 模型许可证，说明权重的使用和分发条件 | 合规文件，不参与加载或导出 |
| `.gitattributes` | Hugging Face 仓库的 Git/LFS 属性配置，标记大文件的存储方式 | 本地运行不需要，仅在 Git/LFS 管理模型仓库时有用 |

本案例的 `export_qwen_block_onnx.py` 绕过 Transformers 的完整模型加载，直接读取的
最小文件集是：

```text
config.json
model.safetensors
```

因此 tokenizer 相关文件虽然属于完整 Qwen 模型快照，但不会影响当前以随机
`hidden_states` 为输入的 layer 0 prefill 数值验证。后续如果从真实 prompt 生成输入、
扩展到完整模型文本生成，才需要保留 `tokenizer.json`、`tokenizer_config.json`，或兼容
路径所需的 `vocab.json + merges.txt`；生成配置则可以按实验需要覆盖。

## Qwen2.5-0.5B-Instruct 基本模型结构

Qwen2.5-0.5B-Instruct 是一个约 5 亿参数的 decoder-only Transformer。完整推理过程是
先把 token ID 映射为 hidden states，依次经过 24 个 Qwen2 decoder layer，再经过最终
RMSNorm；输出层与 token embedding 共享权重，将 hidden states 投影回 151936 维词表，
得到下一个 token 的 logits。

```text
token IDs
  -> token embedding [vocab_size=151936, hidden_size=896]
  -> 24 x Qwen2 decoder layer
  -> final RMSNorm
  -> shared embedding / LM output projection
  -> logits [vocab_size=151936]
```

本地 `config.json` 给出的主要结构参数如下：

| 参数 | 数值 | 含义 |
|---|---:|---|
| `vocab_size` | 151936 | tokenizer 词表大小，也是最终 logits 的通道数 |
| `hidden_size` | 896 | 每个 token 的隐藏向量宽度 |
| `num_hidden_layers` | 24 | decoder layer 数量 |
| `intermediate_size` | 4864 | SwiGLU MLP 的中间维度 |
| `num_attention_heads` | 14 | Query attention head 数 |
| `num_key_value_heads` | 2 | Key/Value head 数，使用 GQA |
| `head_dim` | 64 | 每个 attention head 的维度，即 `896 / 14` |
| `max_position_embeddings` | 32768 | 配置支持的最大位置长度 |
| `rope_theta` | 1000000.0 | RoPE 旋转位置编码的基频参数 |
| `rms_norm_eps` | `1e-6` | RMSNorm 的数值稳定项 |
| `hidden_act` | `silu` | MLP gate 分支使用的激活函数 |
| `torch_dtype` | `bfloat16` | 原始权重声明的数据类型 |

### Transformer Block 的基本结构

Qwen2 的一个 Transformer Block（代码中称为 `Qwen2DecoderLayer`）由两个串联的
子层组成：第一层是 causal self-attention，用来混合不同 token 位置的信息；第二层是
MLP，用来独立变换每个 token 的特征。两层都采用 **pre-norm**：先做 RMSNorm，再做
子层计算，最后与进入该子层之前的输入进行残差相加。

设一个 block 的输入为 `x`，其基本计算为：

```text
h = x + SelfAttention(RMSNorm(x))
y = h + MLP(RMSNorm(h))
```

这里 `x` 和 `y` 的形状都为 `[batch, sequence, hidden_size]`，因此可以连续堆叠 24 个
block。残差连接保留原始特征并改善深层网络的梯度传播；RMSNorm 负责控制进入
attention 和 MLP 的数值尺度。

展开到 Qwen2.5-0.5B 的具体维度后，一个 block 的数据流如下：

```text
hidden_states [B, S, 896]
  |
  +-> RMSNorm
  |     -> q_proj: 896 -> 896  -> 14 Query heads
  |     -> k_proj: 896 -> 128  ->  2 Key heads
  |     -> v_proj: 896 -> 128  ->  2 Value heads
  |     -> RoPE(Q, K) + causal attention
  |     -> o_proj: 896 -> 896
  +-> residual add
  |
  +-> RMSNorm
  |     -> gate_proj: 896 -> 4864 -> SiLU
  |     -> up_proj:   896 -> 4864
  |     -> element-wise multiply
  |     -> down_proj: 4864 -> 896
  +-> residual add
  |
hidden_out [B, S, 896]
```

Self-attention 先由同一个归一化输入分别生成 Q、K、V。Qwen2.5-0.5B 使用 GQA
（Grouped-Query Attention）：14 个 Query heads 共享 2 个 Key/Value heads，因此每个
KV head 服务 7 个 Query heads。Q 和 K 应用 RoPE 以编码位置信息，attention score
除以 `sqrt(head_dim)` 后施加 causal mask，使当前位置只能关注自己及之前的 token：

```text
Q = q_proj(x_norm)
K = k_proj(x_norm)
V = v_proj(x_norm)

Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(64) + causal_mask) @ V
```

各个 head 的结果重新拼接为 896 维，再通过 `o_proj` 映射回 block 的 hidden size，
以便与 attention 子层的输入进行第一次残差相加。

### MLP 的基本结构

Qwen2.5 的 MLP 不是传统的“Linear -> 激活 -> Linear”，而是带门控的 SwiGLU
结构。归一化后的输入同时进入 `gate_proj` 和 `up_proj` 两条并行分支：

```text
                         +-> gate_proj -> SiLU --+
x [B, S, 896] -> RMSNorm |                        | element-wise multiply
                         +-> up_proj -------------+
                                                   -> down_proj
                                                   -> [B, S, 896]
                                                   -> residual add
```

三个线性投影的作用和维度为：

| 投影 | 输入/输出维度 | 作用 |
|---|---|---|
| `gate_proj` | `896 -> 4864` | 生成门控值，并经过 SiLU 激活 |
| `up_proj` | `896 -> 4864` | 将输入扩展到中间维度，提供被门控的特征 |
| `down_proj` | `4864 -> 896` | 将门控后的中间特征压回 hidden size |

MLP 的完整计算为：

```text
gate = SiLU(gate_proj(x_norm))
up   = up_proj(x_norm)
MLP(x_norm) = down_proj(gate * up)
```

其中 `*` 是逐元素乘法，不是矩阵乘法。`gate_proj` 和 `up_proj` 输出相同的
`[B, S, 4864]` 形状，逐元素相乘后再由 `down_proj` 恢复为 `[B, S, 896]`，最后与
MLP 子层的输入 `h` 做第二次残差相加。对当前 custom-op 工作而言，这三个 projection
都是大规模矩阵乘法，也是除 attention projection 外最值得分析和替换的计算热点。

当前 case study 并未导出上述完整 24 层模型，而是只读取 `model.layers.0.*` 权重，导出
一个固定 `batch=1, sequence=16` 的真实 layer 0。其输入为 `[1, 16, 896]` 的
`hidden_states`，输出为同形状的 `hidden_out`，并额外输出形状为 `[1, 2, 16, 64]`
的 `present_key` 和 `present_value`。这样既保留了真实 Qwen block 的 attention、RoPE、
GQA、SwiGLU 和 projection 结构，又避免一开始承担完整模型的转换及设备内存开销。

注意：当前默认 Python 环境里 `torchvision` 和 `torch` 存在版本冲突，直接
`AutoModelForCausalLM.from_pretrained(...)` 会在导入 `torchvision::nms` 时失败。
为了不被这个环境问题阻塞，本目录的导出脚本不依赖 Transformers model import，而是
直接读取：

```text
config.json + model.safetensors
```

然后手写单层 Qwen2 decoder block 的 RMSNorm、RoPE、GQA attention 和 SwiGLU MLP。

模型权重目录被 `.gitignore` 忽略，不提交到仓库：

```text
qwen_block_custom_qnn/model/data/
```

## Qwen Block 中要替换的算子

Qwen decoder layer 的核心 projection 通常包括：

| 模块 | 典型 op | 权重形状方向 | 替换优先级 |
|---|---|---|---|
| attention `q_proj` | Linear / MatMul / FullyConnected | hidden -> hidden | 高 |
| attention `k_proj` | Linear / MatMul / FullyConnected | hidden -> kv_hidden | 中 |
| attention `v_proj` | Linear / MatMul / FullyConnected | hidden -> kv_hidden | 中 |
| attention `o_proj` | Linear / MatMul / FullyConnected | hidden -> hidden | 中 |
| MLP `gate_proj` | Linear / MatMul / FullyConnected | hidden -> intermediate | 高 |
| MLP `up_proj` | Linear / MatMul / FullyConnected | hidden -> intermediate | 高 |
| MLP `down_proj` | Linear / MatMul / FullyConnected | intermediate -> hidden | 中 |

第一轮建议只替换 `q_proj`：

- 它和 `tiny_llm_block_custom_matmul` 中已经做过的路径最接近。
- 输出 shape 通常是 `[batch, seq, hidden]`，便于比对。
- 如果 `q_proj` 跑通，后续迁移到 `gate_proj/up_proj/down_proj` 是同一类工程问题。

## 里程碑

### M1. 抽取真实 Qwen 单层

目标产物：

```text
qwen_block_custom_qnn/model/qwen2_0_5b_layer0_prefill_seq16.onnx
qwen_block_custom_qnn/test_data/layer0_prefill_seq16/*.raw
```

要求：

- 固定 shape，先不做 dynamic axes。
- prefill 先用较短 sequence，例如 `seq=16` 或 `seq=32`。
- 导出时保留 Qwen 原始权重，不重新随机初始化。
- ONNX Runtime 输出和 PyTorch 输出对齐。

当前已经完成 prefill 单层导出。脚本：

```text
qwen_block_custom_qnn/tools/export_qwen_block_onnx.py
```

运行命令：

```bash
PYTHONPATH=/home/lingbok/anaconda3/envs/qairt-2.47/lib/python3.12/site-packages \
python3 qwen_block_custom_qnn/tools/export_qwen_block_onnx.py
```

这里使用默认 Python 的 `torch/safetensors`，并通过 `PYTHONPATH` 加入
`qairt-2.47` 环境中的 `onnx/onnxruntime`。

导出的 graph：

```text
input:
  hidden_states [1, 16, 896] float32

outputs:
  hidden_out    [1, 16, 896] float32
  present_key   [1, 2, 16, 64] float32
  present_value [1, 2, 16, 64] float32
```

ONNX Runtime 校验结果：

```text
hidden_out    max_abs=1.40070915e-06 mean_abs=9.91071829e-08 allclose=True
present_key   max_abs=1.52587891e-05 mean_abs=5.54184453e-07 allclose=True
present_value max_abs=8.19563866e-08 mean_abs=8.39844461e-09 allclose=True
```

除 prefill 外，现已增加 fixed-past KV-cache decode 导出：

```text
input:
  hidden_states [1, 1, 896]       float32
  past_key      [1, 2, 16, 64]    float32
  past_value    [1, 2, 16, 64]    float32

outputs:
  hidden_out    [1, 1, 896]       float32
  present_key   [1, 2, 17, 64]    float32
  present_value [1, 2, 17, 64]    float32
```

当前 token 使用 position 16 做 RoPE，attention 的 K/V 是 past cache 与当前
token K/V 的 concat，因此这是真实 fixed-shape KV-cache decode，而不是把
prefill 的 sequence 简单改成 1。

PyTorch 与 ONNX Runtime 校验：

```text
hidden_out    max_abs=9.53674316e-07
present_key   max_abs=9.53674316e-07
present_value max_abs=3.72529030e-08
```

导出产物中，ONNX 和 raw reference 都可以由脚本重新生成。为了避免仓库过大，
`model/*.onnx` 和 `test_data/layer*_prefill_seq*/` 已经被 `.gitignore` 忽略。

### M2. QNN builtin baseline

目标产物：

```text
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16.cpp
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16.bin
qwen_block_custom_qnn/model_libs/aarch64-android/libqwen2_0_5b_layer0_prefill_seq16.so
qwen_block_custom_qnn/device_output/builtin_layer0_prefill_seq16/profile.csv
```

需要记录：

- `qnn-onnx-converter` 命令。
- `qnn-model-lib-generator` 命令。
- `qnn-net-run` 命令。
- builtin graph 的 root cycles、QNN us、NetRun us。
- 每个 projection 在 CSV 中的 node 名称。

当前 builtin baseline 已经完成。

ONNX -> QNN C++：

```bash
source /home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/envsetup.sh
export LD_LIBRARY_PATH="/home/lingbok/anaconda3/envs/qairt-2.47/lib:$LD_LIBRARY_PATH"

/home/lingbok/anaconda3/envs/qairt-2.47/bin/python \
  /home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-onnx-converter \
  -i qwen_block_custom_qnn/model/qwen2_0_5b_layer0_prefill_seq16.onnx \
  -o qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16.cpp \
  --preserve_io layout
```

converter 在当前 sandbox 中会打印 multiprocessing socket 权限 warning，但最终成功：

```text
Total MACs: 15243584
Total Params Count: 14910592
Model CPP saved at: qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16.cpp
Model BIN saved at: qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16.bin
Conversion complete
```

QNN model lib 编译：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16.cpp \
  -b qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16.bin \
  -t aarch64-android \
  -l qwen2_0_5b_layer0_prefill_seq16 \
  -o qwen_block_custom_qnn/model_libs
```

设备运行：

```bash
adb shell mkdir -p \
  /data/local/tmp/qnn/qwen_block_custom_qnn/lib \
  /data/local/tmp/qnn/qwen_block_custom_qnn/input/layer0_prefill_seq16

adb push \
  qwen_block_custom_qnn/model_libs/aarch64-android/libqwen2_0_5b_layer0_prefill_seq16.so \
  /data/local/tmp/qnn/qwen_block_custom_qnn/lib/libqwen2_0_5b_layer0_prefill_seq16.so

adb push \
  qwen_block_custom_qnn/test_data/layer0_prefill_seq16/hidden_states.raw \
  /data/local/tmp/qnn/qwen_block_custom_qnn/input/layer0_prefill_seq16/hidden_states.raw

adb push \
  qwen_block_custom_qnn/test_data/device_layer0_prefill_seq16_input_list.txt \
  /data/local/tmp/qnn/qwen_block_custom_qnn/input/layer0_prefill_seq16/input_list.txt

adb shell '
cd /data/local/tmp/qnn
rm -rf qwen_block_custom_qnn/output_builtin_layer0_prefill_seq16
export LD_LIBRARY_PATH="$PWD/qwen_block_custom_qnn/lib:$PWD/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
./bin/qnn-net-run \
  --backend lib/libQnnHtp.so \
  --model qwen_block_custom_qnn/lib/libqwen2_0_5b_layer0_prefill_seq16.so \
  --input_list qwen_block_custom_qnn/input/layer0_prefill_seq16/input_list.txt \
  --output_dir qwen_block_custom_qnn/output_builtin_layer0_prefill_seq16 \
  --input_data_type float \
  --output_data_type float_only \
  --perf_profile burst \
  --profiling_level detailed \
  --num_inferences 10 \
  --log_level info
'
```

拉回并分析：

```bash
mkdir -p qwen_block_custom_qnn/device_output/builtin_layer0_prefill_seq16

adb pull \
  /data/local/tmp/qnn/qwen_block_custom_qnn/output_builtin_layer0_prefill_seq16/Result_0 \
  qwen_block_custom_qnn/device_output/builtin_layer0_prefill_seq16/

adb pull \
  /data/local/tmp/qnn/qwen_block_custom_qnn/output_builtin_layer0_prefill_seq16/qnn-profiling-data_0.log \
  qwen_block_custom_qnn/device_output/builtin_layer0_prefill_seq16/qnn-profiling-data_0.log

/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-profile-viewer \
  --input_log qwen_block_custom_qnn/device_output/builtin_layer0_prefill_seq16/qnn-profiling-data_0.log \
  --output qwen_block_custom_qnn/device_output/builtin_layer0_prefill_seq16/profile.csv

python3 qwen_block_custom_qnn/tools/compare_qnn_output.py
python3 qwen_block_custom_qnn/tools/summarize_profile.py
```

正确性结果：

```text
hidden_out.raw    max_abs=3.63290310e-03 mean_abs=3.51709779e-04 cosine=0.999999881
present_key.raw   max_abs=6.74743652e-02 mean_abs=3.93668935e-03 cosine=0.999999881
present_value.raw max_abs=8.48025084e-05 mean_abs=7.87456520e-06 cosine=1.000000000
```

Profiling 摘要，median-after-warmup：

```text
root_cycles        2276136
qnn_accel_us       12141
qnn_us             14777
netrun_us          14826
q_proj_cycles      0
gate_proj_cycles   178255
up_proj_cycles     350868
down_proj_cycles   375479
```

注意：和 tiny block 一样，QNN builtin 的 projection subevent cycles 不能简单理解成
原始 MatMul/FullyConnected 的完整真实耗时。`q_proj_cycles=0` 不代表没有计算，而是
说明 builtin 可能经过了 graph rewrite、fusion 或 profiling 归因变化。后续 custom
op 对比时，要优先看 graph-level `root_cycles/qnn_us/netrun_us`。

QNN converter/model-lib 产物也可以重新生成，并且体积较大，因此已忽略：

```text
qwen_block_custom_qnn/generated/*.cpp
qwen_block_custom_qnn/generated/*.bin
qwen_block_custom_qnn/generated/*.json
qwen_block_custom_qnn/model_libs/
```

保留下来的小文件主要是：

```text
qwen_block_custom_qnn/tools/export_qwen_block_onnx.py
qwen_block_custom_qnn/tools/compare_qnn_output.py
qwen_block_custom_qnn/tools/summarize_profile.py
qwen_block_custom_qnn/test_data/device_layer0_prefill_seq16_input_list.txt
qwen_block_custom_qnn/device_output/builtin_layer0_prefill_seq16/profile.csv
qwen_block_custom_qnn/device_output/builtin_layer0_prefill_seq16/qnn-profiling-data_0.log
```

### M3. 识别并 patch `q_proj`

目标产物：

```text
qwen_block_custom_qnn/tools/patch_qwen_prefill_q_proj_custom.py
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_custom.cpp
```

当前已经完成 `q_proj` patch。QNN converter 生成的 q_proj 节点是 `_MatMul`：

```text
_MatMul_pre_reshape [16, 896]
onnx__MatMul_227   [896, 896]
self_attn_q_proj_bias [896]
-> FullyConnected
-> _Add_1_output_0_fc [16, 896]
```

patch 后的数据流是：

```text
_MatMul_pre_reshape [16, 896]
-> Reshape [1, 1, 16, 896]
-> Cast fp16

onnx__MatMul_227 [1, 1, 896, 896]
-> Cast fp16

lhs_fp16, rhs_fp16
-> MatMulQhpiHvx8RowLhsTileCacheFp32Store
-> fp32 output [1, 1, 16, 896]
-> Reshape [16, 896]
-> ElementWiseBinary Add self_attn_q_proj_bias
-> _Add_1_output_0_fc [16, 896]
```

这里保留 bias add 是必要的，因为原始 QNN `FullyConnected` 把 bias 作为第三个输入；
custom MatMul 只做矩阵乘，所以 bias 必须单独补回。

生成 patched QNN C++：

```bash
python3 qwen_block_custom_qnn/tools/patch_qwen_prefill_q_proj_custom.py
```

输出：

```text
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_custom.cpp
```

复用的 custom op package：

```text
qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_tile_cache_fp32_store
```

需要的两个库：

```text
qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_tile_cache_fp32_store/htp/MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage/build/aarch64-android/libQnnMatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage.so
qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_tile_cache_fp32_store/htp/MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage/build/hexagon-v75/libQnnMatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage.so
```

编译 patched model lib：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_custom.cpp \
  -b qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16.bin \
  -t aarch64-android \
  -l qwen2_0_5b_layer0_prefill_seq16_q_proj_custom \
  -o qwen_block_custom_qnn/model_libs
```

### M4. 设备运行与正确性

目标产物：

```text
qwen_block_custom_qnn/device_output/q_proj_custom_layer0_prefill_seq16/Result_0/
  qnn-profiling-data_0.log
  profile.csv
```

推送 patched model lib 和 custom OpPackage：

```bash
adb push \
  qwen_block_custom_qnn/model_libs/aarch64-android/libqwen2_0_5b_layer0_prefill_seq16_q_proj_custom.so \
  /data/local/tmp/qnn/qwen_block_custom_qnn/lib/libqwen2_0_5b_layer0_prefill_seq16_q_proj_custom.so

adb push \
  qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_tile_cache_fp32_store/htp/MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage/build/aarch64-android/libQnnMatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage.so \
  /data/local/tmp/qnn/qwen_block_custom_qnn/lib/libQnnMatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage.so

adb push \
  qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_tile_cache_fp32_store/htp/MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage/build/hexagon-v75/libQnnMatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage.so \
  /data/local/tmp/qnn/dsp/libQnnMatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage.so
```

运行 custom q_proj：

```bash
adb shell '
cd /data/local/tmp/qnn
rm -rf qwen_block_custom_qnn/output_q_proj_custom_layer0_prefill_seq16
export LD_LIBRARY_PATH="$PWD/qwen_block_custom_qnn/lib:$PWD/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
./bin/qnn-net-run \
  --backend lib/libQnnHtp.so \
  --model qwen_block_custom_qnn/lib/libqwen2_0_5b_layer0_prefill_seq16_q_proj_custom.so \
  --input_list qwen_block_custom_qnn/input/layer0_prefill_seq16/input_list.txt \
  --output_dir qwen_block_custom_qnn/output_q_proj_custom_layer0_prefill_seq16 \
  --input_data_type float \
  --output_data_type float_only \
  --perf_profile burst \
  --profiling_level detailed \
  --num_inferences 10 \
  --log_level info \
  --op_packages qwen_block_custom_qnn/lib/libQnnMatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage.so:MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackageInterfaceProvider:CPU,libQnnMatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage.so:MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackageInterfaceProvider:HTP
'
```

拉回输出并生成 CSV：

```bash
mkdir -p qwen_block_custom_qnn/device_output/q_proj_custom_layer0_prefill_seq16

adb pull \
  /data/local/tmp/qnn/qwen_block_custom_qnn/output_q_proj_custom_layer0_prefill_seq16/Result_0 \
  qwen_block_custom_qnn/device_output/q_proj_custom_layer0_prefill_seq16/

adb pull \
  /data/local/tmp/qnn/qwen_block_custom_qnn/output_q_proj_custom_layer0_prefill_seq16/qnn-profiling-data_0.log \
  qwen_block_custom_qnn/device_output/q_proj_custom_layer0_prefill_seq16/qnn-profiling-data_0.log

/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-profile-viewer \
  --input_log qwen_block_custom_qnn/device_output/q_proj_custom_layer0_prefill_seq16/qnn-profiling-data_0.log \
  --output qwen_block_custom_qnn/device_output/q_proj_custom_layer0_prefill_seq16/profile.csv
```

正确性需要同时做两种误差对比：

1. custom QNN output vs PyTorch/ONNX reference。
2. custom QNN output vs QNN builtin output。

这样能区分：

- QNN builtin 本身和 PyTorch 的误差。
- custom op 替换额外引入的误差。

custom QNN output vs reference：

```text
hidden_out.raw    max_abs=1.83071047e-02 mean_abs=1.19500572e-03 cosine=0.999998629
present_key.raw   max_abs=6.74743652e-02 mean_abs=3.93668935e-03 cosine=0.999999881
present_value.raw max_abs=8.48025084e-05 mean_abs=7.87456520e-06 cosine=1.000000000
```

custom QNN output vs builtin QNN：

```text
hidden_out.raw    max_abs=1.87988281e-02 mean_abs=1.15328748e-03 cosine=0.999998569
present_key.raw   max_abs=0.00000000e+00 mean_abs=0.00000000e+00 cosine=1.000000119
present_value.raw max_abs=0.00000000e+00 mean_abs=0.00000000e+00 cosine=1.000000000
```

这里 `present_key/present_value` 与 builtin 完全一致，是合理的：这次只替换了
`q_proj`，不会影响 `k_proj/v_proj` 产生的 KV cache。`hidden_out` 有额外误差，
主要来自 q_proj custom path 中的 fp16 cast 和 custom MatMul 计算路径。

Profiling 摘要，median-after-warmup：

```text
builtin:
root_cycles             2276136
qnn_accel_us            12141
qnn_us                  14777
netrun_us               14826
q_proj_rhs_cast_cycles  0
q_proj_cycles           0
q_proj_bias_add_cycles  0
gate_proj_cycles        178255
up_proj_cycles          350868
down_proj_cycles        375479

q_proj_custom:
root_cycles             3171078
qnn_accel_us            14486
qnn_us                  16254
netrun_us               16310
q_proj_rhs_cast_cycles  2961
q_proj_cycles           798959
q_proj_bias_add_cycles  159020
gate_proj_cycles        231841
up_proj_cycles          325152
down_proj_cycles        386739
```

结论：

- custom q_proj 已经成功接入真实 Qwen layer0 prefill graph，并在 HTP 上运行。
- 数值上可用：`hidden_out` cosine 仍然约为 `0.9999986`。
- 在 M4 的初始版本中，custom 慢于 QNN builtin：`netrun_us` 从 `14826` 增加到
  `16310`；后续 M7 已通过 device-Q13 把 NetRun 降到 `13152 us`。
- custom q_proj 自身约 `798959` cycles，额外 bias add 约 `159020` cycles。
- builtin 的 `q_proj_cycles=0` 不能理解成 q_proj 免费，而是 profiling 归因/fusion 后没有以
  `_MatMul:` subevent 形式暴露。
- 这个结果说明下一步优化重点不是盲目替换更多 projection，而是减少 custom path 的
  cast、reshape、bias add 开销，并尽量接近 QNN builtin 的 layout/prepack/fusion 行为。

### M5. 尝试融合 q_proj bias

上一版 custom q_proj 的 profile 中，`_MatMul_bias_add` 约为 `159020` cycles。
因此下一步尝试把 q_proj bias 作为 custom op 的第三个输入，在 MatMul 写回 FP32
输出时直接加上 bias，去掉外部 QNN `ElementWiseBinary Add`。

改动包括：

- `MatMulQhpiHvx8RowLhsTileCacheFp32Store` 支持 optional 第三输入 `bias`。
- QNN OpPackage validate 允许 2 输入或 3 输入。
- HVX `8x64/1x64` store 前加 FP32 bias vector。
- 新增 fused-bias patch 脚本：

```text
qwen_block_custom_qnn/tools/patch_qwen_prefill_q_proj_custom_fused_bias.py
```

生成 fused-bias QNN C++：

```bash
python3 qwen_block_custom_qnn/tools/patch_qwen_prefill_q_proj_custom_fused_bias.py
```

输出：

```text
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_custom_fused_bias.cpp
```

这个版本在 QNN graph 中先把 `self_attn_q_proj_bias` reshape 成 `[1,1,1,896]`，
再作为 custom op 第三个输入：

```text
self_attn_q_proj_bias [896]
-> _MatMul_bias_reshape [1,1,1,896]

_MatMul_lhs_fp16, _MatMul_rhs_fp16, _MatMul_q_proj_bias_4d
-> MatMulQhpiHvx8RowLhsTileCacheFp32Store
-> _MatMul_custom_output_reshape
-> _Add_1_output_0_fc
```

重新编译 OpPackage：

```bash
PKG=qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_tile_cache_fp32_store/htp/MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage

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
```

编译 fused-bias model lib：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_custom_fused_bias.cpp \
  -b qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16.bin \
  -t aarch64-android \
  -l qwen2_0_5b_layer0_prefill_seq16_q_proj_custom_fused_bias \
  -o qwen_block_custom_qnn/model_libs
```

运行时只需要把 `--model` 换成 fused-bias model lib，其他参数和 M4 一样：

```text
qwen_block_custom_qnn/lib/libqwen2_0_5b_layer0_prefill_seq16_q_proj_custom_fused_bias.so
```

fused-bias 版本可以成功在设备上运行，并且 profile 中外部 bias add 已消失：

```text
q_proj_bias_add_cycles  0
```

但数值和性能结果都没有变好。

fused-bias vs reference：

```text
hidden_out.raw    max_abs=7.81732798e-02 mean_abs=4.95759305e-03 cosine=0.999977410
present_key.raw   max_abs=6.74743652e-02 mean_abs=3.93668935e-03 cosine=0.999999881
present_value.raw max_abs=8.48025084e-05 mean_abs=7.87456520e-06 cosine=1.000000000
```

fused-bias vs old custom：

```text
hidden_out.raw    max_abs=7.61718750e-02 mean_abs=4.77580959e-03 cosine=0.999978423
present_key.raw   max_abs=0.00000000e+00 mean_abs=0.00000000e+00 cosine=1.000000119
present_value.raw max_abs=0.00000000e+00 mean_abs=0.00000000e+00 cosine=1.000000000
```

fused-bias profile，median-after-warmup：

```text
root_cycles             3113048
qnn_accel_us            14340
qnn_us                  16717
netrun_us               16764
q_proj_rhs_cast_cycles  3388
q_proj_cycles           799981
q_proj_bias_add_cycles  0
gate_proj_cycles        221750
up_proj_cycles          367354
down_proj_cycles        374743
```

结论：

- fused-bias 的工程链路是通的：3-input custom op、QNN validate、HTP runtime 都能跑。
- 它确实去掉了外部 `_MatMul_bias_add`。
- 但 `hidden_out` 误差明显大于 old custom，cosine 从约 `0.9999986` 降到约 `0.9999774`。
- 端到端也没有收益：`netrun_us` 从 old custom 的 `16310` 变成 `16764`。
- 因此当前不应把 fused-bias 作为默认优化版本。后续如果继续研究，需要先单独导出
  q_proj 输出，比较 custom op 内部 bias add 与 QNN 外部 Add 的逐元素差异。

### M6. multithread + LHS tile-cache q_proj

为了把 standalone MatMul 中验证过的 self-slicing 和 fused LHS tile-cache 接入真实
Qwen graph，新增 patch 脚本：

```text
qwen_block_custom_qnn/tools/patch_qwen_prefill_q_proj_multithread_tile_cache.py
```

它以已经验证过的 old custom QNN C++ 为输入，只把 package/type 改为：

```text
MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage
MatMulQhpiHvx8RowFp32StoreMultithread
```

生成和编译：

```bash
python3 \
  qwen_block_custom_qnn/tools/patch_qwen_prefill_q_proj_multithread_tile_cache.py

PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_custom_multithread_tile_cache.cpp \
  -b qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16.bin \
  -t aarch64-android \
  -l qwen2_0_5b_layer0_prefill_seq16_q_proj_custom_multithread_tile_cache \
  -o qwen_block_custom_qnn/model_libs
```

为了避免手工命令遗漏，当前版本还提供完整复现脚本：

```bash
bash qwen_block_custom_qnn/tools/run_multithread_tile_cache_repro.sh
```

脚本依次完成 QNN C++ patch、HTP/ARM OpPackage 编译、Android model lib 编译、
真机部署、10 次 detailed profiling、结果拉取、精度检查和 profile 汇总。以下路径均可
通过同名环境变量覆盖：

```text
QAIRT_ROOT
HEXAGON_SDK_ROOT
HEXAGON_TOOLS_VERSION
ANDROID_NDK_ROOT
DEVICE_QNN_ROOT
DEVICE_CASE_DIR
NUM_INFERENCES
RUN_TAG
```

每次运行的材料保存在
`qwen_block_custom_qnn/device_output/${RUN_TAG}/`，其中：

| 文件 | 含义 |
|---|---|
| `Result_0/*.raw` | 三个 Qwen block 输出 |
| `qnn-profiling-data_0.log` | QNN 原始 profiling 日志 |
| `profile.csv` | `qnn-profile-viewer` 解码结果 |
| `correctness.txt` | 相对导出 reference 的误差与 cosine |
| `summary.txt` | 丢弃第一次 warm-up 后的各项中位数 |

默认脚本复现当前源码中的 `M=16` 4-row 分支；前一版 8-row 数据作为历史对照保留在
`device_output/q_proj_custom_multithread_tile_cache_layer0_prefill_seq16/`。

首先验证 8-row self-slicing。结果与 old custom 三个输出逐 bit 相同：

```text
hidden_out.raw    max_abs_vs_old=0 array_equal=True
present_key.raw   max_abs_vs_old=0 array_equal=True
present_value.raw max_abs_vs_old=0 array_equal=True
```

10 次 inference、丢弃第一次 warm-up 后的中位数：

| 版本 | root cycles | QNN accelerator | QNN | NetRun | Accelerator |
|---|---:|---:|---:|---:|---:|
| builtin | 2,276,136 | 12,141 us | 14,777 us | 14,826 us | 6,032 us |
| old custom 8-row single-thread | 3,171,078 | 14,486 us | 16,254 us | 16,310 us | 8,209 us |
| multithread + 8-row LHS cache | 3,222,940 | 13,456 us | 15,816 us | 15,861 us | 7,356 us |

相对 old custom：

```text
QNN accelerator: 14,486 -> 13,456 us，降低 7.11%
Accelerator:       8,209 ->  7,356 us，降低 10.39%
NetRun:           16,310 -> 15,861 us，降低 2.75%
```

因此真实 Qwen layer0 的 custom-vs-builtin NetRun 差距从约 `10.0%` 缩小到约
`7.0%`。custom subevent 聚合 cycles 没有同步下降，原因仍是 self-sliced worker
的 cycles 会聚合记录；判断多线程收益应看 wall time。

#### `M=16` 的 4-row self-slicing 实验

Qwen prefill seq16 只有两个 8-row tiles，4 个 HVX workers 中只有两个 worker
获得实际 q_proj 工作。为验证更细的并行粒度，kernel 在 `M=16 && num_slices>=4`
时使用 4-row tiles，其他 shape 继续使用 8-row 路径。

4-row 输出仍与 8-row multithread 逐 bit 相同。结果：

```text
Accelerator:                7,356 -> 6,888 us，降低 6.36%
Accelerator excluding wait: 5,448 -> 4,987 us，降低 8.46%
QNN accelerator:           13,456 -> 13,058 us，降低 2.96%
NetRun:                    15,861 -> 15,884 us，基本持平
```

4-row 分支改善了 custom kernel 所在的 accelerator 路径，但完整 layer0 的 NetRun
没有进一步收益。这说明当前端到端瓶颈已经扩散到 graph 其他节点、RPC/runtime wait
和静态 weight 数据准备，继续缩小 row tile 不是主要方向。

结果目录：

```text
qwen_block_custom_qnn/device_output/q_proj_custom_multithread_tile_cache_layer0_prefill_seq16/
qwen_block_custom_qnn/device_output/q_proj_custom_multithread_4row_layer0_prefill_seq16/
```

#### 未采用实验：QHPI graph-load static RHS precompute

为消除每次 inference 内重复的 RHS FP16-to-Q13 转换，还验证过
`QHPI_Kernel_v1::do_precomputation_function` 路径。实验模型 patch 保存在：

```text
qwen_block_custom_qnn/tools/patch_qwen_prefill_q_proj_static_rhs_precomputed.py
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_static_rhs_precomputed.cpp
```

验证过两种输入方式：

1. precompute 直接读取 `_MatMul_rhs_fp16`，但该 tensor 是上游 `Cast` 的运行时输出；
2. 增加原始 static FP32 权重 `onnx__MatMul_227` 作为第三输入，在 graph load 时读取。

后续用独立的 16-byte 最小 probe 复查后确认：QAIRT 2.47 / HTP v75 的
`do_precomputation_function` 和 `function_with_precomputed_data` 可以正常完成
graph finalize 与 inference。之前的 `Context Creation failure` 实际来自错误的
`qnn-net-run --op_packages` 参数：

```text
错误:
  libPackage.so:interfaceProvider

正确:
  /device/arm/libPackage.so:完整PackageInterfaceProvider符号:CPU,
  libPackage.so:完整PackageInterfaceProvider符号:HTP
```

独立验证包保存在 `qnn_custom_ops/qhpi_precompute_probe/`。因此 static RHS
precompute API 本身可用，但继续放大实验得到以下结果：

| 实验 | 结果 |
|---|---|
| 128 KiB header-only | inference 成功 |
| 1 MiB header-only | inference 成功 |
| 约 1.53 MiB header-only | inference 成功 |
| 约 1.53 MiB，遍历完整 static RHS 并写 Q13 | graph finalize 成功，execution failure |
| 约 896 KiB，只缓存 RHS 前 512 列 | graph finalize 成功，execution failure |

所以失败不是 `precomputed_data_size` 上限，而是当前 runtime 在 graph-load precompute
阶段不保证输入 tensor 完整 payload 可供遍历。QHPI 文档明确举例的是 shape 和
quantization 等 runtime metadata，并没有保证上游 tensor 数据已经执行或 static
payload 已按 kernel layout materialize。

因此完整 static RHS kernel 仍未加入默认注册表。后续实验改为在模型生成阶段准备
static INT16 tensor，不再依赖 QHPI graph-load 回调读取完整权重 payload。

### M7. 设备导出 Q13 RHS：当前采用版本

#### 为什么不能直接在主机生成普通 Q13

旧 custom kernel 在每次 inference 中对 FP16 RHS 调用：

```cpp
hnnx::s16_from_hf_rnd_sat<13>(fp16_values)
```

QAIRT 2.47 的 `HTP/core/hvx_mathops.h` 明确说明：

- half-way case 使用 away-from-zero；
- 该模板只保证 `FBITS=-2..9`；
- `FBITS>=10` 会产生内部舍入异常。

第一次主机离线方案按标准数学 Q13 生成 INT16，模型能够执行，但没有复现旧 kernel
的数值行为：

| 版本 | `hidden_out` max abs vs reference | mean abs | cosine | NetRun |
|---|---:|---:|---:|---:|
| 4-row custom | 0.0183071 | 0.00119501 | 0.999998629 | 15,884 us |
| 主机数学 Q13 | 0.0990594 | 0.00797660 | 0.999942243 | 16,745 us |

ties-to-even 和 away-from-zero 都测试过，均不能匹配，所以该版本作为负实验保留，
不作为默认模型。

相关文件：

```text
qwen_block_custom_qnn/tools/generate_q_proj_q13_model_bin.py
qwen_block_custom_qnn/tools/patch_qwen_prefill_q_proj_offline_q13.py
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_q13.bin
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_q13.json
qwen_block_custom_qnn/device_output/q_proj_offline_q13_layer0_prefill_seq16/
```

#### 用 HTP conversion probe 获取真实转换结果

为了逐 bit 复现旧 kernel，新增最小 probe graph：

```text
FP32 q_proj weight
  -> QNN builtin Cast(FP16)
  -> HTP hnnx::s16_from_hf_rnd_sat<13>
  -> APP_READ QNN_DATATYPE_INT_16
  -> qnn-net-run --output_data_type native_only
```

probe 文件：

```text
qwen_block_custom_qnn/tools/device_q13_probe_model.cpp
qwen_block_custom_qnn/tools/device_q13_probe_input_list.txt
qwen_block_custom_qnn/device_output/device_q13_conversion_probe/Result_0/weight_q13_native.raw
```

probe 输出大小为 `802816 * 2 = 1605632` 字节，SHA256 为：

```text
b838c624526c30de48b0a8a3fe3c853b8a570e6d3a47e167302498f3868fc38e
```

对所有 `802816` 个 q_proj 权重逐元素比较后，当前 QAIRT/HTP 组合上的实际关系是：

```text
device_int16 == mathematical_q13 XOR 0x8000
delta ∈ {-32768, +32768}
different elements = 802816 / 802816
```

这不是标准 Q13 语义，而是当前未受支持 `<13>` intrinsic 的真实设备行为。因此采用
版本必须使用设备 probe 导出的 native INT16，不能用 NumPy 普通量化结果替代。

#### 嵌回 QNN graph

生成器把 device-Q13 payload 添加到原 QNN tar BIN：

```text
onnx__MatMul_227_q13.raw
```

graph patch 随后完成：

1. 将 q_proj RHS static tensor 改为 `QNN_DATATYPE_INT_16`。
2. 删除 `_MatMul_rhs_cast_fp16`。
3. 让 q_proj custom op 直接读取 INT16 RHS。
4. 把 op type 改为
   `MatMulQhpiHvxOfflineQ13RhsFp32StoreMultithread`。
5. kernel 保留运行时 LHS FP16-to-Q13 tile cache，但不再转换静态 RHS。

最终产物：

```text
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_device_q13.bin
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_device_q13.json
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_offline_q13.cpp
qwen_block_custom_qnn/model_libs/aarch64-android/libqwen2_0_5b_layer0_prefill_seq16_q_proj_device_q13.so
```

#### 正确性

device-Q13 与原 4-row custom 的三个 graph outputs 全部逐 bit 相同：

```text
hidden_out.raw    max_abs=0 mean_abs=0 array_equal=True
present_key.raw   max_abs=0 mean_abs=0 array_equal=True
present_value.raw max_abs=0 mean_abs=0 array_equal=True
```

相对导出 FP32 reference 的误差也与原 custom 保持一致：

```text
hidden_out.raw    max_abs=1.83071047e-02 mean_abs=1.19500572e-03 cosine=0.999998629
present_key.raw   max_abs=6.74743652e-02 mean_abs=3.93668935e-03 cosine=0.999999881
present_value.raw max_abs=8.48025084e-05 mean_abs=7.87456520e-06 cosine=1.000000000
```

#### 真机性能

测试条件保持一致：

```text
device:         OnePlus 12 / Snapdragon 8 Gen 3
runtime:        QAIRT/QNN 2.47, HTP v75
shape:          Qwen2.5-0.5B layer0 prefill, B=1, S=16
perf profile:   burst
profiling:      detailed
inferences:     10
statistics:     丢弃第一次 warm-up 后取中位数
HVX threads:    4
```

结果：

| 版本 | root cycles | QNN accelerator | QNN | NetRun |
|---|---:|---:|---:|---:|
| QNN builtin | 2,276,136 | 12,141 us | 14,777 us | 14,826 us |
| old custom | 3,171,078 | 14,486 us | 16,254 us | 16,310 us |
| multithread 4-row | — | 13,058 us | — | 15,884 us |
| device-Q13 4x64 | 3,131,207 | 10,347 us | 13,125 us | 13,152 us |
| **device-Q13 4x128** | **3,114,247** | **10,325 us** | **13,080 us** | **13,105 us** |

端到端提升：

```text
4x64 vs multithread 4-row: (15884 - 13152) / 15884 = 17.20%
4x64 vs QNN builtin:       (14826 - 13152) / 14826 = 11.29%
4x128 vs QNN builtin:      (14826 - 13105) / 14826 = 11.61%
```

device-Q13 版本的主要 profile 中位数：

```text
root_cycles               3131207
qnn_accel_us              10347
qnn_us                    13125
netrun_us                 13152
q_proj_rhs_cast_cycles        0
q_proj_cycles            826368
q_proj_bias_add_cycles   152671
gate_proj_cycles         231876
up_proj_cycles           359163
down_proj_cycles         389421
```

当前源码进一步采用 `4x128` micro-kernel：每次 reduction 同时读取两个相邻的
64-column RHS vectors，并把四个 LHS splat 复用到 128 个输出列。每个输出的累加
顺序保持不变，因此三个 graph outputs 与 `4x64` 版本逐 bit 相同。

```text
4x128 root_cycles                3114247
4x128 qnn_accel_us                 10325
4x128 qnn_us                       13080
4x128 netrun_us                    13105
4x128 q_proj_cycles              582392
```

q_proj cycles 相对 `4x64` 降低 `29.52%`，但 NetRun 只再降低 `47 us`，说明端到端
瓶颈已经扩散到其他节点和 runtime 路径。详细结果保存在：

```text
qwen_block_custom_qnn/device_output/q_proj_device_q13_4x128_layer0_prefill_seq16/
```

暂不直接采用 `4x256`：它需要 16 个 `HVX_VectorPair` accumulator，仅 accumulator
就占满 32 个 HVX vector registers，极易造成 register spill。

还测试过 `4x128` reduction 2-way unroll。输出仍逐 bit 相同，但 q_proj cycles 从
`582392` 增加到 `672836`，退化 `15.53%`。虽然该次 NetRun 为 `13091 us`，相对
`13105 us` 仅差 `14 us`，其方向与 kernel cycles 相反且小于整图波动，因此不采用。
源码已恢复非展开版本，负实验保存在：

```text
qwen_block_custom_qnn/device_output/q_proj_device_q13_4x128_unroll2_layer0_prefill_seq16/
```

#### 删除不再使用的 FP32 q_proj 权重

device-Q13 graph 已只引用 `onnx__MatMul_227_q13.raw`，因此最终 BIN 使用
`--drop-source-member` 删除原 `onnx__MatMul_227.raw`：

```text
original QNN BIN:               59,678,720 bytes
slim device-Q13 BIN:            58,071,040 bytes
old non-slim device-Q13 model:  61,713,616 bytes
slim device-Q13 model lib:      58,502,080 bytes
model lib reduction:             3,211,536 bytes (5.20%)
```

删除量等于 `896 * 896 * sizeof(float)`。瘦身模型正常 compose、finalize 和 execute，
三个输出与非瘦身 `4x128` 逐 bit 相同。10 次 profiling 的 NetRun 中位数为
`13077 us`；与 `13105 us` 的小差异视为运行波动，采用理由是减少包体、传输和加载
材料，而不是 execute kernel 加速。

最终一键复现结果目录为：

```text
qwen_block_custom_qnn/device_output/q_proj_device_q13_4x128_slim_layer0_prefill_seq16/
```

custom op cycles 与 builtin 内部 fused subevent cycles 的归因方式不同，不能直接一一
比较；是否采用仍以相同输入和 profiling 策略下的 NetRun/QNN wall time 为准。

#### 一键复现

完整的 probe、构建、设备导出、BIN 重打包、模型编译、逐 bit 校验和 profiling 已固化：

```bash
bash qwen_block_custom_qnn/tools/run_device_q13_repro.sh
```

可覆盖环境变量：

```text
QAIRT_ROOT
HEXAGON_SDK_ROOT
HEXAGON_TOOLS_VERSION
ANDROID_NDK_ROOT
DEVICE_QNN_ROOT
NUM_INFERENCES
```

结果保存在：

```text
qwen_block_custom_qnn/device_output/device_q13_conversion_probe/
qwen_block_custom_qnn/device_output/q_proj_device_q13_layer0_prefill_seq16/
```

其中包括 native INT16 权重、三个 raw 输出、原始 profiling log、CSV、正确性摘要和
性能摘要。

注意：device-Q13 固定的是当前 QAIRT 2.47、Hexagon compiler、HTP v75 和目标 SoC
的实际 intrinsic 行为。升级 SDK、编译器或更换 SoC 后，必须重新执行 conversion
probe，并重新检查权重 SHA256 和三个 graph outputs，不能直接复用旧 payload。

### M10. 形成技术总结

最后要在根 README 中总结成可快速理解的工程结果：

```text
Case study: replacing Qwen q_proj with custom QNN HTP MatMul
Device: Snapdragon 8 Gen 3 / OnePlus 12
Runtime: QAIRT/QNN 2.47, HTP v75
Flow: ONNX -> QNN C++ -> custom OpPackage -> Android qnn-net-run -> profile
Result: correctness error, graph-level latency, op cycles, bottleneck analysis
```

本实验最终已经超过 builtin；同时，中间失败实验仍有价值，因为真实工程里重要的不只是
最终延迟，还包括：

- 能接入真实模型。
- 能定位 converter 生成的 graph。
- 能替换并跑通 HTP custom op。
- 能解释为什么 builtin 更快，例如内部 fusion、weight prepack、layout rewrite。

## 和已有实验的关系

已有目录提供的复用能力：

| 目录 | 可复用内容 |
|---|---|
| `tiny_llm_block` | Qwen-style block 数据流、prefill/decode、QNN builtin 流程 |
| `tiny_llm_block_custom_matmul` | QNN C++ patch、custom op 替换、profile 对比脚本 |
| `qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store_multithread` | 当前 device-Q13 采用版本和 conversion probe |
| `qnn_quantization` | QNN quantization 公式和 QDQ/量化实验背景 |

当前真实 Qwen case 最值得复用的 custom op 是：

```text
MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage
MatMulQhpiHvxOfflineQ13RhsFp32StoreMultithread
```

它结合 4-row self-slicing、worker-private LHS cache 和设备导出的静态 INT16 RHS，
在真实 layer0 prefill seq16 上将 NetRun 降到 `13,152 us`。

## KV cache decode 实验

### 实验目标

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

### 输入输出和产物

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

### 正确性

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

### Decode 专用 M=1 内核

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

### Builtin 与 custom 性能

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

### KV cache 优化 1：context sweep 与 QNN 边界布局修正

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

### KV cache 优化 2：grouped-GQA 消除 K/V Tile

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

### KV cache 优化 3：宿主维护 cache，只输出当前 K/V delta

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

### KV cache 优化 4（未采用）：FP16 cache 边界

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

### KV cache 优化 5（未采用）：QNN shared buffer 与 input cache

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

### KV cache 优化 6（未采用）：固定 past128 的 decode attention 融合

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

#### 实验 6.1：标量 FP32 online softmax（当前保留的可读参考实现）

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

#### 实验 6.2：快速指数近似（未保留到主源码）

将 `exp(x), x<=0` 改成 `ln2` range reduction + 五阶多项式，`x<=-16` flush
为零；并根据 online-softmax 的 max 分支把每 token 两次 exp 降成一次。输出精度
与 6.1 基本相同，但融合节点反而约 `17.9M cycles`，NetRun `24158 us`。这说明
瓶颈不只是 libm 调用，大量标量 FP32 score/value 循环才是核心问题。

```text
device_output/fused_attention_fast_exp_layer0_decode_past128/
```

#### 实验 6.3：HVX FP32 value 累加（未保留到主源码）

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

### KV cache 优化 7（采用为 runner 基线）：持久化 tensor 与宿主滑动 KV cache

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

### KV cache 优化 8（采用为可选模式）：persistent rpcmem + memRegister

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

### KV cache 优化 9（未采用）：NCHW value-cache graph boundary

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

## 当前状态

当前状态：

- M0 已完成：真实 `Qwen2.5-0.5B-Instruct` 权重已下载到本地。
- M1 已完成：真实 Qwen layer0 prefill 固定 shape ONNX 已导出并通过 ONNX Runtime 校验。
- M2 已完成：QNN builtin baseline 已在 OnePlus 12 / Snapdragon 8 Gen 3 HTP 上跑通。
- M3 已完成：定位 QNN C++ 中的 `_MatMul` q_proj 节点，并 patch 成 custom HTP MatMul。
- M4 已完成：custom q_proj model lib 已在设备上跑通，并生成 correctness/profile 结果。
- M5 已完成：尝试 fused-bias custom q_proj，工程上跑通，但数值和端到端性能都不如 old custom。
- M6 已完成：接入 multithread + LHS tile-cache；8-row 版本将 NetRun 从
  `16310` 降至 `15861 us`，4-row 分支只改善 accelerator 路径，NetRun 基本持平。
- M7 已完成：用 HTP conversion probe 导出设备实际 Q13 权重，离线嵌入 q_proj 并
  删除 RHS runtime Cast；输出与 4-row custom 逐 bit 相同，4x64 NetRun 为
  `13152 us`。
- M8 已完成：device-Q13 kernel 从 4x64 扩展到 4x128，q_proj cycles 从
  `826368` 降至 `582392`，NetRun 降至 `13105 us`，相对 builtin 的 `14826 us`
  快 `11.61%`。
- M9 已完成：移除最终 device-Q13 模型中不再引用的 FP32 q_proj payload，model lib
  减少 `3,211,536 bytes`，三个输出逐 bit 相同。
- M10 已完成：实现 `past_len=16, token=1` 的真实 KV-cache decode，并在同一设备
  对比 builtin 与 M=1 device-Q13 custom q_proj。功能和精度通过，但 custom decode
  NetRun 为 `11041 us`，略慢于 builtin 的 `11007 us`，因此不采用 custom decode。
- M11 已完成：建立 past16/32/64/128 decode sweep，修正 QNN KV 边界 layout；
  grouped-GQA 消除 K/V Tile，在 past128 输出逐 bit 相同的前提下将 NetRun 从
  `12032 us` 降到 `11862 us`。
- M12 已完成：改成宿主维护持久 cache、graph 只输出当前 K/V delta；past128
  NetRun 从 `11862 us` 降到 `10907 us`，三路对应输出逐 bit 相同。
- M13 已完成：验证 FP16 KV 边界；容量减半且 accelerator cycles 下降，但 NetRun
  从 `10907 us` 退化到 `11767 us`，因此记录但不采用。
- M14 已完成：验证 QNN shared buffer 和 input tensor-set cache；30 次同窗口复测
  分别比普通模式慢 `2.50%/3.55%`，因此记录但不采用。
- M15 已完成：导出固定 past128 fused decode-attention graph，连续完成标量
  online-softmax、快速 exp 和 HVX value 累加三轮真机实验；工程和参考精度跑通，
  但最快版本仍为 `19432 us` 且误差放大，因此不采用，回到 builtin attention。
- M16 已完成：实现一次初始化、一次 tensor 分配、连续 100 步执行的 persistent
  past128 runner；第一步三路输出逐 bit 相同，`execute + CPU cache update`
  中位数 `5037 us`，相同无 profiling 100 次进程 wall time 比 qnn-net-run
  改善 `16.32%`，采用为后续 shared-memory cache 的 runner 基线。
- M17 已完成：为 persistent runner 加入 rpcmem + `memRegister` 的 MEMHANDLE
  模式；两组正反夹心共 6 次 100 步测试均保持逐 bit正确，step-total 中位数均值
  从 `4977.67` 降至 `4941.67 us`（约 `0.72%`），采用为可选推荐模式。
- M18 已完成：验证 NCHW value-cache boundary，成功删除约 25k-cycle 的输入
  Transpose且输出逐 bit相同，但下游 V Concat/layout 成本增加，persistent
  step total 反而慢 `0.99%`，因此保留实验文件但不采用。

下一步仍不应无验证地一次替换所有 projection。应以当前 device-Q13 q_proj 为基线：

```text
1. 把同一 device conversion 流程逐个扩展到 gate_proj/up_proj/down_proj。
2. 每替换一个 projection 都单独保存逐 bit/误差结果和 10 次 profiling。
3. 评估模型 BIN 增加 INT16 副本的空间代价，并在最终版本移除不再使用的 FP32 q_proj payload。
4. 继续研究 custom output 与 bias Add 的安全融合，但不能采用已失败的 fused-bias 实现。
5. decode 保留 builtin q_proj；若继续研究，应优先消除独立 bias Add/调度开销，
   而不是复用面向 prefill 的多行策略。
```

到这里，这个目录已经形成了一个完整的真实 Qwen custom op 工程案例：真实模型权重、
单层 prefill/decode 导出、KV cache、QNN builtin、QNN C++ patch、custom HTP
OpPackage、设备运行、CSV profile 和性能结论都有了。当前 prefill 采用版本已经
端到端超过 QNN builtin；decode 则根据实测保留 builtin q_proj。
