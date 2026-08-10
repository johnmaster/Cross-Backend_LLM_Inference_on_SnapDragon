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

除 prefill 外，fixed-shape KV-cache decode 导出和后续优化见
[KV_CACHE_EXPERIMENTS.md](KV_CACHE_EXPERIMENTS.md)。

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

#### Profiling 字段说明

本文的性能汇总由 `tools/summarize_profile.py` 从 `profile.csv` 提取。每项测试运行
10 次，丢弃第一次 warm-up，再对其余记录取中位数。几个 graph-level 字段的原始
事件和计时边界如下：

| 字段 | `profile.csv` 原始事件 | 含义 |
|---|---|---|
| `root_cycles` | `Accelerator (execute) time (cycles)` | HTP accelerator 执行整张 graph 的 ROOT cycle 统计，用于分析设备侧工作量 |
| `qnn_accel_us` | `QNN accelerator (execute) time` | QNN HTP backend 观察到的 accelerator 阶段 wall-time，包括整张图的算子执行、设备调度及相关等待 |
| `qnn_us` | `QNN (execute) time` | 一次 QNN graph execute 的总时间，除 accelerator 阶段外还包含 backend、RPC、同步和输入输出管理等开销 |
| `netrun_us` | `NETRUN / ROOT / EXECUTE` | `qnn-net-run` 应用层观察到的一次 graph execute 时间，是本文比较单次 inference 端到端延迟的主要指标 |

计时范围通常可近似理解为：

```text
accelerator 工作
  ⊂ QNN accelerator
  ⊂ QNN graph execute
  ⊂ qnn-net-run graph execute
```

因此一般会看到：

```text
qnn_accel_us <= qnn_us <= netrun_us
```

这里的 `netrun_us` 是每次 `EXECUTE` 事件，不是进程从启动到退出的总时间；它不包含
backend/model library 加载和 graph finalize。若要衡量真实连续 decode，还应使用
persistent runner 的多步 wall-time A/B。

`root_cycles` 也不能直接按固定频率换算为上述微秒值。HTP 可能动态调频，而且 QHPI
self-slicing 下多个 worker 的算子 cycles 可能被聚合：聚合 cycles 增加时，wall-time
仍可能因并行而下降。因此单线程热点分析可以参考 `root_cycles` 和各 op subevent
cycles；判断多线程及端到端优化是否有效，应优先看 `qnn_accel_us`、`qnn_us` 和
`netrun_us`。

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

### M7. 离线预转换静态 RHS：device-Q13

M7 要解决的不是 MatMul 主循环本身，而是静态 RHS 在每次推理中被重复
转换的问题。q_proj 权重在原图中是 `[896, 896]` FP32 tensor，原运行时
路径为：

```text
FP32 static RHS
  -> QNN Cast(FP16)
  -> hnnx::s16_from_hf_rnd_sat<13>
  -> QHPI/HVX MatMul
```

权重不随 inference 变化，因此理想方案是只转换一次，把 INT16 RHS 直接存入
model BIN，运行时跳过 RHS Cast 和 FP16-to-INT16 转换。难点在于：这里必须
复现旧 kernel 的实际输入位模式，不能只生成数学意义上的普通 Q13。

#### 1. 为什么普通主机 Q13 不等价

普通 Q13 是一种有 13 个小数位的 `int16_t` 定点表示。对实数 `x` 进行编码时，
通常在主机上执行：

```text
q = saturate_int16(round(x * 2^13))
x_approx = q * 2^-13
```

例如 `x=0.5` 会被编码为 `4096`。因此最初的直觉是：先在主机上把 FP32
权重转成 FP16，再按上式舍入和饱和为 INT16，就可以离线复现运行时转换。

但旧 custom kernel 并不是用一段受保证的标准 Q13 代码，而是对 FP16 RHS
调用：

```cpp
hnnx::s16_from_hf_rnd_sat<13>(fp16_values)
```

其中模板参数 `13` 表示按 `2^13` 缩放。然而 QAIRT 2.47 的
`HTP/core/hvx_mathops.h` 只保证该 intrinsic 在 `FBITS=-2..9` 范围内的行为，
并注明 `FBITS>=10` 可能存在内部舍入异常。`FBITS=13` 虽然可以编译和执行，
但已超出有保证的参数范围，所以不能假定其结果等于上述数学 Q13。

实验也验证了这一点。首次主机离线方案按普通 Q13 生成 INT16，模型能够
执行，但没有复现旧 kernel 的数值行为：

| 版本 | `hidden_out` max abs vs reference | mean abs | cosine | NetRun |
|---|---:|---:|---:|---:|
| 4-row custom | 0.0183071 | 0.00119501 | 0.999998629 | 15,884 us |
| 主机数学 Q13 | 0.0990594 | 0.00797660 | 0.999942243 | 16,745 us |

`ties-to-even` 和 `away-from-zero` 两种舍入规则都测试过，均不能匹配。
这说明差异不只是主机舍入规则选错，而是超范围 intrinsic 的实际设备行为
与普通 Q13 根本不同。因此主机数学 Q13 只作为负实验，不作为采用版本。

相关文件：

```text
qwen_block_custom_qnn/tools/generate_q_proj_q13_model_bin.py
qwen_block_custom_qnn/tools/patch_qwen_prefill_q_proj_offline_q13.py
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_q13.bin
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_q13.json
qwen_block_custom_qnn/device_output/q_proj_offline_q13_layer0_prefill_seq16/
```

#### 2. 用 HTP conversion probe 获取旧 kernel 的实际结果

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

即所有元素的最高位都与普通 Q13 相反。这不是标准 Q13 语义，而是当前
QAIRT/Hexagon/HTP 组合对超范围 `<13>` intrinsic 的实际执行结果。

因此 M7 的目标不是纠正旧 kernel 的 Q13 语义，而是在保持现有 graph 输出不变的
前提下消除重复转换。所以采用版本必须使用设备 probe 导出的 native INT16，
不能用 NumPy 生成的普通 Q13 替代。

#### 3. 将 device-Q13 原样嵌回 QNN graph

生成脚本从原 QNN tar BIN 中读取 FP32 成员 `onnx__MatMul_227.raw`，并将
probe 产生的 `weight_q13_native.raw` 字节原样写回新 BIN：

```text
onnx__MatMul_227_q13.raw
```

采用路径使用 `--device-q13`，生成器只校验 payload 是否为
`896 * 896 * sizeof(int16_t)` 字节，不会再量化、转置或 reshape。因此：

```text
weight_q13_native.raw
  -> byte-for-byte copy
  -> onnx__MatMul_227_q13.raw in the new model BIN
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

#### 4. 正确性：先保证与原 custom graph 逐 bit 一致

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

#### 5. 真机性能

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

端到端提升：

```text
4x64 vs multithread 4-row: (15884 - 13152) / 15884 = 17.20%
4x64 vs QNN builtin:       (14826 - 13152) / 14826 = 11.29%
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

M7 的核心收益来自消除静态 RHS 的运行时 Cast 和重复 FP16-to-INT16 转换；
它与后续 M8 的 micro-kernel 列宽优化是两个独立步骤。

### M8. device-Q13 micro-kernel 从 4x64 扩展到 4x128

在 M7 的 device-Q13 `4x64` 路径上，当前源码进一步采用 `4x128`
micro-kernel：每次 reduction 同时读取两个相邻的
64-column RHS vectors，并把四个 LHS splat 复用到 128 个输出列。每个输出的累加
顺序保持不变，因此三个 graph outputs 与 `4x64` 版本逐 bit 相同。

```text
4x128 root_cycles                3114247
4x128 qnn_accel_us                 10325
4x128 qnn_us                       13080
4x128 netrun_us                    13105
4x128 q_proj_cycles              582392
```

`4x128` 相对 `4x64` 的 q_proj cycles 降低 `29.52%`，NetRun 从
`13152 us` 降至 `13105 us`，只再降低 `47 us`。相对 QNN builtin 的 NetRun
改善为 `(14826 - 13105) / 14826 = 11.61%`。这说明端到端
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

### M9. 删除不再使用的 FP32 q_proj 权重

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

#### M7–M9 一键复现

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

KV cache decode、grouped-GQA、宿主持久 cache、shared buffer、fused attention
及 graph boundary 实验已迁移到独立文档：

- [KV_CACHE_EXPERIMENTS.md](KV_CACHE_EXPERIMENTS.md)

当前结论是：decode 保留 builtin q_proj；persistent runner 采用宿主滑动 cache，
并可选使用 `rpcmem + memRegister`。详细的正确性、性能数据和未采用实验见上述文档。

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
单层 prefill 导出、QNN builtin、QNN C++ patch、custom HTP
OpPackage、设备运行、CSV profile 和性能结论都有了。当前 prefill 采用版本已经
端到端超过 QNN builtin。KV cache decode 工程另见独立实验文档。
