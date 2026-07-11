# Qwen Block Custom QNN/HTP Case Study

本目录用于把 `tiny_llm_block_custom_matmul` 里验证过的 QNN/HTP custom op
链路迁移到真实 Qwen-family transformer block 上，形成一个可复现的工程验证案例。

一句话目标：

```text
从真实 Qwen block 中定位 attention/MLP projection，
把其中一个或多个 FullyConnected/MatMul 替换为自定义 QNN HTP OpPackage，
在 Snapdragon 设备上完成正确性、性能和瓶颈分析闭环。
```

## 为什么这一步最重要

`tiny_llm_block` 证明的是技术链路可控：

```text
ONNX -> QNN converter -> QNN C++ patch -> custom HTP OpPackage
-> Android qnn-net-run -> qnn-profile-viewer -> correctness/perf analysis
```

但更关键的问题是：

```text
这套能力能不能迁移到真实 LLM？
```

所以本目录要回答的是：

- 是否能从真实 Qwen 模型中抽取一个 decoder block。
- 是否能识别真实 graph 中的 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`。
- 是否能替换 QNN converter 产物中的 projection op。
- 替换后是否能在 HTP 上运行，并给出数值误差和 profiling 结论。
- 如果 custom op 没有超过 QNN builtin，能否解释瓶颈在哪里。

这比单纯继续优化 tiny block 更接近真实端侧 LLM 推理优化工作流。

## 建议选型

第一阶段建议选一个小 Qwen 模型或 Qwen-style 模型，先抽取单层，而不是直接全模型：

| 候选 | 用途 | 原因 |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 首选真实 Qwen-family case | hidden/intermediate 相对小，结构真实，和当前 tiny block 更接近 |
| `Qwen/Qwen3-0.6B` | 后续可选 | 更新一代 Qwen，但先用 Qwen2.5 降低变量 |
| `Qwen2.5-1.5B-Instruct` | 第二阶段 | 更接近真实端侧负载，但导出/运行压力更大 |
| 当前 `tiny_llm_block` | 对照组 | 已跑通完整 QNN/HTP/custom op 流程 |

不要一开始就追求完整 Qwen 文本生成。更合理的第一阶段目标是：

```text
真实 Qwen decoder layer 单层 prefill/decode
```

因为它已经覆盖 attention、RoPE、KV cache、MLP、projection 和 QNN profiling。

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
```

当前已经完成第 1 到第 13 步，形成了一个真实 Qwen block 的 custom QNN/HTP
替换案例。

## M0. 先拿到真实 Qwen 模型

第一步是下载并固定一个真实模型快照。不要直接在脚本里每次
`from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")` 自动联网下载；那样不利于复现。
建议把模型放到一个固定的本地目录。当前本机已经下载到：

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

当前 M1 只完成 prefill。decode 会增加 fixed past KV 输入和 cache concat，建议在
prefill 完成 QNN builtin baseline 后再补。

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
- 性能上 custom 版本目前慢于 QNN builtin：`netrun_us` 从 `14826` 增加到 `16310`。
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

### M6. 形成技术总结

最后要在根 README 中总结成可快速理解的工程结果：

```text
Case study: replacing Qwen q_proj with custom QNN HTP MatMul
Device: Snapdragon 8 Gen 3 / OnePlus 12
Runtime: QAIRT/QNN 2.47, HTP v75
Flow: ONNX -> QNN C++ -> custom OpPackage -> Android qnn-net-run -> profile
Result: correctness error, graph-level latency, op cycles, bottleneck analysis
```

即使 custom op 没有超过 builtin，也有价值。因为真实工程里重要的不只是“赢”，还包括：

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
| `qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_tile_cache_fp32_store` | 当前最有效的 custom q_proj baseline |
| `qnn_quantization` | QNN quantization 公式和 QDQ/量化实验背景 |

当前最值得复用的 custom op 是：

```text
MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage
```

因为在 tiny block 中它已经把 q_proj custom cycles 从百万级降低到约 `146k`。

## 当前状态

当前状态：

- M0 已完成：真实 `Qwen2.5-0.5B-Instruct` 权重已下载到本地。
- M1 已完成：真实 Qwen layer0 prefill 固定 shape ONNX 已导出并通过 ONNX Runtime 校验。
- M2 已完成：QNN builtin baseline 已在 OnePlus 12 / Snapdragon 8 Gen 3 HTP 上跑通。
- M3 已完成：定位 QNN C++ 中的 `_MatMul` q_proj 节点，并 patch 成 custom HTP MatMul。
- M4 已完成：custom q_proj model lib 已在设备上跑通，并生成 correctness/profile 结果。
- M5 已完成：尝试 fused-bias custom q_proj，工程上跑通，但数值和端到端性能都不如 old custom。

下一步不要直接改所有 projection。当前应该先围绕已经跑通的 `q_proj` 做 targeted
优化，重点看：

```text
1. 先不要继续使用 fused-bias 作为默认版本；它去掉了外部 Add，但误差更大且端到端更慢。
2. 单独导出 q_proj 中间输出，定位 fused-bias 误差来自 bias 读取、FP32 vector add，还是 QNN Add 语义差异。
3. 能不能预处理 q_proj weight，减少 runtime Cast/layout 转换。
4. 能不能让 custom op 直接接受 QNN converter 产物的实际 layout，减少 reshape/cast。
5. 如果要继续替换 gate_proj/up_proj/down_proj，必须先证明 q_proj custom path 的 overhead 可控。
```

到这里，这个目录已经形成了一个完整的真实 Qwen custom op 工程案例：真实模型权重、
单层导出、QNN builtin、QNN C++ patch、custom HTP OpPackage、设备运行、CSV profile
和性能结论都有了。
