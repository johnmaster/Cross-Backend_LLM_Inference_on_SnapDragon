# Tiny LLM Block

本目录用于把仓库中的 MatMul、量化和 HTP profiling 实验连接到真实的
Transformer 数据流。

## 模型来源与整体流程

这里的模型不是从 Hugging Face 或真实 Qwen checkpoint 中裁剪出来的一层，而是
手工实现的缩小版 Qwen-style decoder block。它保留了 RMSNorm、RoPE、GQA、
causal attention 和 SwiGLU 等主要数据流，但不包含 token embedding、多层堆叠、
最终 RMSNorm 和 LM head，因此不是一个可以独立生成文本的完整 LLM。

完整生成流程如下：

```text
NumPy reference（定义结构并生成确定性随机权重）
  -> PyTorch 等价实现（复制相同权重）
  -> torch.onnx.export（分别 trace prefill 和 decode）
  -> 固定 shape FP32 ONNX
  -> qnn-onnx-converter
  -> QNN model C++ + 权重 bin
  -> qnn-model-lib-generator
  -> Android model library
```

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

权重不是预训练参数，而是在 `TinyDecoderBlock` 中使用 `seed=42` 和正态分布
`N(0, 0.02)` 生成。两个 RMSNorm weight 初始化为 1，Q/K/V bias 初始化为 0。
固定随机种子使每次重新导出都能得到相同的 reference 参数和数值结果。

从仓库根目录运行 reference 示例：

```bash
python -m tiny_llm_block.reference
```

从仓库根目录运行测试：

```bash
python -m unittest tiny_llm_block.test_reference
```

这两个命令依赖 `tiny_llm_block/__init__.py` 让该目录作为 Python package 被导入。
如果已经 `cd tiny_llm_block`，则不要再使用上面的 module 路径；建议回到仓库根目录
执行，避免导入路径不一致。

测试覆盖：

- 输出和 KV cache shape
- prefill 与逐 token decode 数值一致性
- causal mask 保证未来 token 不影响过去输出
- 非法 hidden dimension 检查

## 第二阶段：固定 Shape ONNX

导出两张互相衔接的 FP32 ONNX 图：

```text
tiny_block_prefill_seq32.onnx
  input:  hidden_states [1, 32, 256]
  output: hidden_out    [1, 32, 256]
          present_key   [1, 2, 32, 32]
          present_value [1, 2, 32, 32]

tiny_block_decode_past32.onnx
  input:  hidden_states [1, 1, 256]
          past_key/past_value [1, 2, 32, 32]
  output: hidden_out    [1, 1, 256]
          present_key   [1, 2, 33, 32]
          present_value [1, 2, 33, 32]
```

### ONNX 是怎样导出的

`export_onnx.py` 中的 `TorchTinyDecoderBlock` 用 PyTorch 重新表达了 NumPy
reference 的同一套计算。构造它时会遍历 `TinyDecoderBlock`，把其中所有 NumPy
数组复制为 PyTorch buffer：

```python
for name, value in vars(reference).items():
    if isinstance(value, np.ndarray):
        self.register_buffer(name, torch.from_numpy(value.copy()))
```

因此 ONNX 使用的不是另一组新权重，而是 NumPy reference 的完全相同权重。这里
使用 buffer 而不是 `nn.Linear` parameter；调用 `torch.onnx.export` 时，这些
buffer 仍会随模型写入 ONNX initializer。

脚本使用两组具体输入分别调用一次 `torch.onnx.export`：

1. Prefill 输入只有 `[1, 32, 256]` 的 `hidden_states`，此时
   `past_key/past_value` 为 `None`，trace 得到不包含历史 KV 拼接的 prefill 图。
2. Decode 输入包含 `[1, 1, 256]` 的新 token，以及 prefill 产生的
   `[1, 2, 32, 32]` KV cache。此时 trace 会记录 past KV 与当前 KV 的
   `Concat`，输出 cache 长度增长到 33。

导出没有设置 `dynamic_axes`，所以 batch、sequence 和 past length 都被固定在
示例输入的 shape 上。这也是文件名中 `seq32` 和 `past32` 的含义。两个 ONNX
共享同一套 block 权重和计算语义，但因为 trace 时走过的 Python 分支不同，最终是
两张独立的静态图，而不是一张带可选 KV 输入的动态图。

导出的主要参数为：

```python
torch.onnx.export(
    model,
    inputs,
    path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=input_names,
    output_names=["hidden_out", "present_key", "present_value"],
)
```

导出需要默认环境中的 PyTorch，以及 `qairt-2.47` 环境中的 ONNX 和 ONNX
Runtime。当前机器可直接执行：

```bash
PYTHONPATH=/home/lingbok/anaconda3/envs/qairt-2.47/lib/python3.12/site-packages \
python -m tiny_llm_block.export_onnx
```

脚本会：

- 生成 `model/` 中的 prefill 和 decode ONNX。
- 生成 `test_data/` 中的 FP32 raw 输入、期望输出和 QNN input-list。
- 使用 `onnx.checker.check_model` 检查 ONNX 合法性。
- 使用 ONNX Runtime 执行两张图，并以 `rtol=2e-5`、`atol=2e-6` 对齐 NumPy
  reference。

`model/` 目录不是手工放入的模型文件，而是由上面的 `export_onnx.py` 产生：

| 文件 | 生成位置 | 来源 |
|---|---|---|
| `tiny_block_prefill_seq32.onnx` | `args.model_dir / "tiny_block_prefill_seq32.onnx"` | 用 `[1, 32, 256]` hidden input trace 出来的 prefill ONNX |
| `tiny_block_decode_past32.onnx` | `args.model_dir / "tiny_block_decode_past32.onnx"` | 用 `[1, 1, 256]` hidden input 和 `[1, 2, 32, 32]` past KV trace 出来的 decode ONNX |

默认输出目录是 `tiny_llm_block/model/`，由脚本中的
`DEFAULT_MODEL_DIR = ROOT / "model"` 决定。需要放到其他位置时可以显式传入：

```bash
PYTHONPATH=/home/lingbok/anaconda3/envs/qairt-2.47/lib/python3.12/site-packages \
python -m tiny_llm_block.export_onnx \
  --model-dir /tmp/tiny_llm_block_model \
  --data-dir /tmp/tiny_llm_block_test_data
```

## 第三阶段：QNN FP32 HTP

下面以 QAIRT `2.47.0.260601`、Android NDK r28 和 Snapdragon 8 Gen 3
（Hexagon v75）为例，从仓库根目录完成转换、编译、部署和执行。这个 block 只使用
QNN builtin op，不依赖自定义 OpPackage，因此运行时不需要 `--op_packages`。

### 1. 初始化主机环境

```bash
export QAIRT_SDK_ROOT=/home/lingbok/Qualcomm/qairt/2.47.0.260601
export ANDROID_NDK_ROOT=/home/lingbok/android/android-ndk-r28
export QNN_PHONE_ROOT=/data/local/tmp/qnn

source "$QAIRT_SDK_ROOT/bin/envsetup.sh"
export PATH="$ANDROID_NDK_ROOT:$PATH"
```

先运行第二阶段的导出命令，确认下面两张 ONNX 已存在：

```text
tiny_llm_block/model/tiny_block_prefill_seq32.onnx
tiny_llm_block/model/tiny_block_decode_past32.onnx
```

### 2. ONNX 转换为 QNN model source

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

每张图会生成一个 `.cpp` 和一个保存静态权重的 `.bin`。`--preserve_io layout`
保证设备端仍然按照 ONNX 的 `[batch, sequence, hidden]` 顺序读取和写出 raw
数据。

converter 对 `model/` 中 ONNX 的处理可以理解为四步：

1. 读取 ONNX graph：包括输入输出名、固定 shape、节点拓扑，以及 initializer
   中的权重和常量。这里的权重来自 `torch.onnx.export` 写入 ONNX 的 buffer。
2. 将 ONNX op 映射为 QNN builtin op：例如 MatMul/linear 会变成
   `FullyConnected` 或 `MatMul`，RMSNorm 会映射成 QNN `RmsNorm`，Softmax、
   Reshape、Transpose、Concat、Slice、Tile 等也会变成对应的 QNN op。
3. 生成 C++ model source：`.cpp` 里会出现大量 `addTensor_*` 和 `addNode_*`
   函数，描述每个 QNN tensor 的 dtype、rank、dimension，以及每个 op node 的
   输入、输出和参数。后面 `QnnModel_composeGraphs` 会按顺序调用这些函数来
   compose graph。
4. 拆出静态数据：大块 initializer、权重和常量写入同名 `.bin`；`.cpp` 中只保留
   graph 构建逻辑和少量小常量。`*_net.json` 则记录 converter 后的 graph
   metadata，方便人工检查。

以 prefill 为例，输入 ONNX 是：

```text
tiny_llm_block/model/tiny_block_prefill_seq32.onnx
```

运行 converter 后得到：

```text
tiny_llm_block/generated/tiny_block_prefill.cpp
tiny_llm_block/generated/tiny_block_prefill.bin
tiny_llm_block/generated/tiny_block_prefill_net.json
```

decode 图同理，只是输入 ONNX 换成
`tiny_llm_block/model/tiny_block_decode_past32.onnx`，输出前缀换成
`tiny_block_decode`。

转换完成后，`generated/` 目录中的文件来源如下：

| 文件 | 生成命令 | 作用 |
|---|---|---|
| `tiny_block_prefill.cpp` | `qnn-onnx-converter -i ...prefill... -o ...prefill.cpp` | prefill 图的 QNN C++ model source |
| `tiny_block_prefill.bin` | 同一次 prefill converter | prefill 图的静态权重和常量数据 |
| `tiny_block_prefill_net.json` | 同一次 prefill converter | converter 导出的网络结构描述，便于检查 graph 和 tensor 信息 |
| `tiny_block_decode.cpp` | `qnn-onnx-converter -i ...decode... -o ...decode.cpp` | decode 图的 QNN C++ model source |
| `tiny_block_decode.bin` | 同一次 decode converter | decode 图的静态权重和常量数据 |
| `tiny_block_decode_net.json` | 同一次 decode converter | converter 导出的网络结构描述，便于检查 graph 和 tensor 信息 |

也就是说，`generated/` 不是手写代码目录，而是由第二阶段的 ONNX 继续通过
`qnn-onnx-converter` 自动生成。后续 `qnn-model-lib-generator` 只消费其中的
`.cpp` 和 `.bin`；`*_net.json` 不参与编译，但适合用来排查 converter 后的
节点、输入输出名和 shape。

### 3. 编译 Android model library

`qnn-model-lib-generator` 把 converter 生成的 model source 和权重编译为
Android ARM64 共享库。模型库运行在 Android ARM 进程中，负责通过 QNN API
compose graph；真正的图执行由后面选择的 HTP backend 完成。

```bash
qnn-model-lib-generator \
  -c tiny_llm_block/generated/tiny_block_prefill.cpp \
  -b tiny_llm_block/generated/tiny_block_prefill.bin \
  -t aarch64-android \
  -l tiny_block_prefill \
  -o tiny_llm_block/model_libs

qnn-model-lib-generator \
  -c tiny_llm_block/generated/tiny_block_decode.cpp \
  -b tiny_llm_block/generated/tiny_block_decode.bin \
  -t aarch64-android \
  -l tiny_block_decode \
  -o tiny_llm_block/model_libs
```

产物为：

```text
tiny_llm_block/model_libs/aarch64-android/libtiny_block_prefill.so
tiny_llm_block/model_libs/aarch64-android/libtiny_block_decode.so
```

### 4. 准备手机端 QNN runtime

以下步骤只需在首次配置 QAIRT runtime 或更换 SDK 版本后执行。Snapdragon 8
Gen 3 使用 v75 HTP skel；其他芯片需要改为对应架构目录。

```bash
adb shell "mkdir -p \
  $QNN_PHONE_ROOT/bin \
  $QNN_PHONE_ROOT/lib \
  $QNN_PHONE_ROOT/dsp"

adb push \
  "$QAIRT_SDK_ROOT/bin/aarch64-android/qnn-net-run" \
  "$QNN_PHONE_ROOT/bin/"
adb push \
  "$QAIRT_SDK_ROOT/lib/aarch64-android/"*.so \
  "$QNN_PHONE_ROOT/lib/"
adb push \
  "$QAIRT_SDK_ROOT/lib/hexagon-v75/unsigned/"* \
  "$QNN_PHONE_ROOT/dsp/"
adb shell "chmod 755 $QNN_PHONE_ROOT/bin/qnn-net-run"
```

可以先用 `qnn-platform-validator` 单独检查 HTP runtime；完整的 SDK 部署和排错
说明见 `docs/QNN_learning/qnn-sdk.md`。

### 5. 部署 block 和输入

设备目录采用以下布局：

```text
/data/local/tmp/qnn/
  bin/                  qnn-net-run
  lib/                  ARM64 QNN runtime
  dsp/                  HTP skel
  tiny_llm_block/
    lib/                两个 block model library
    input/              FP32 raw 和 device input-list
    output_prefill/
    output_decode/
```

创建目录并推送模型和输入：

```bash
adb shell "mkdir -p \
  $QNN_PHONE_ROOT/tiny_llm_block/lib \
  $QNN_PHONE_ROOT/tiny_llm_block/input"

adb push \
  tiny_llm_block/model_libs/aarch64-android/libtiny_block_prefill.so \
  tiny_llm_block/model_libs/aarch64-android/libtiny_block_decode.so \
  "$QNN_PHONE_ROOT/tiny_llm_block/lib/"

adb push \
  tiny_llm_block/test_data/prefill_hidden.raw \
  tiny_llm_block/test_data/decode_hidden.raw \
  tiny_llm_block/test_data/decode_past_key.raw \
  tiny_llm_block/test_data/decode_past_value.raw \
  tiny_llm_block/test_data/device_prefill_input_list.txt \
  tiny_llm_block/test_data/device_decode_input_list.txt \
  "$QNN_PHONE_ROOT/tiny_llm_block/input/"
```

两个 device input-list 使用相对于 `tiny_llm_block/` 工作目录的路径：

```text
# device_prefill_input_list.txt
hidden_states:=input/prefill_hidden.raw

# device_decode_input_list.txt（三个输入在同一行）
hidden_states:=input/decode_hidden.raw past_key:=input/decode_past_key.raw past_value:=input/decode_past_value.raw
```

Decode 的 past KV 是 NumPy prefill reference 产生的 cache。这样可以独立测试
固定 `past_length=32` 的 decode 图，不必先在设备上运行 prefill 再改写 input
list。

### 6. 在 HTP 上运行 prefill 和 decode

`LD_LIBRARY_PATH` 供 Android linker 查找 ARM64 runtime，`ADSP_LIBRARY_PATH`
供 FastRPC/CDSP 查找 HTP skel。前者使用冒号分隔，后者在这里使用分号分隔。

运行 prefill：

```bash
adb shell '
cd /data/local/tmp/qnn/tiny_llm_block
export LD_LIBRARY_PATH="$PWD/lib:$PWD/../lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/../dsp;$PWD/../lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
rm -rf output_prefill
../bin/qnn-net-run \
  --backend ../lib/libQnnHtp.so \
  --model lib/libtiny_block_prefill.so \
  --input_list input/device_prefill_input_list.txt \
  --output_dir output_prefill \
  --input_data_type float \
  --output_data_type float_only \
  --perf_profile burst \
  --log_level info
'
```

运行 decode：

```bash
adb shell '
cd /data/local/tmp/qnn/tiny_llm_block
export LD_LIBRARY_PATH="$PWD/lib:$PWD/../lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/../dsp;$PWD/../lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
rm -rf output_decode
../bin/qnn-net-run \
  --backend ../lib/libQnnHtp.so \
  --model lib/libtiny_block_decode.so \
  --input_list input/device_decode_input_list.txt \
  --output_dir output_decode \
  --input_data_type float \
  --output_data_type float_only \
  --perf_profile burst \
  --log_level info
'
```

这里的 `--backend ../lib/libQnnHtp.so` 才是选择 HTP backend 的关键。
`--model` 指向的是在线 compose graph 的 ARM64 model library，不是 HTP context
binary，因此不能改用 `--retrieve_context`。

成功执行后，每个输出目录应包含：

```text
Result_0/hidden_out.raw
Result_0/present_key.raw
Result_0/present_value.raw
```

### 7. 拉回输出并检查精度

```bash
mkdir -p \
  tiny_llm_block/device_output/prefill \
  tiny_llm_block/device_output/decode

adb pull \
  "$QNN_PHONE_ROOT/tiny_llm_block/output_prefill/Result_0" \
  tiny_llm_block/device_output/prefill/
adb pull \
  "$QNN_PHONE_ROOT/tiny_llm_block/output_decode/Result_0" \
  tiny_llm_block/device_output/decode/

python tiny_llm_block/compare_qnn_output.py \
  prefill tiny_llm_block/device_output/prefill/Result_0
python tiny_llm_block/compare_qnn_output.py \
  decode tiny_llm_block/device_output/decode/Result_0
```

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

### 8. HTP profiling

QNN profiling 的流程分成三步：

1. 在设备端运行带 `--profiling_level detailed` 的 `qnn-net-run`，生成
   `qnn-profiling-data_0.log`。
2. 用 `adb pull` 把 profile log 拉回主机。
3. 在主机端用 `qnn-profile-viewer` 转成 CSV，再用 `compare_profile.py`
   汇总关键指标。

注意工具名是 `qnn-profile-viewer`，不是 `qnn-profiler-viewer`。它是主机端工具，
通常位于：

```text
$QAIRT_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer
```

先确认设备已连接：

```bash
adb devices
```

然后重新运行 prefill profiling。这里仍然使用 `--perf_profile burst` 固定性能档位，
并用 `--num_inferences 10` 连续运行 10 次，后续统计时会跳过第一次 warm-up：

```bash
adb shell '
cd /data/local/tmp/qnn/tiny_llm_block
export LD_LIBRARY_PATH="$PWD/lib:$PWD/../lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/../dsp;$PWD/../lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
rm -rf output_prefill_profile
../bin/qnn-net-run \
  --backend ../lib/libQnnHtp.so \
  --model lib/libtiny_block_prefill.so \
  --input_list input/device_prefill_input_list.txt \
  --output_dir output_prefill_profile \
  --input_data_type float \
  --output_data_type float_only \
  --perf_profile burst \
  --profiling_level detailed \
  --num_inferences 10 \
  --log_level info
'
```

再运行 decode profiling：

```bash
adb shell '
cd /data/local/tmp/qnn/tiny_llm_block
export LD_LIBRARY_PATH="$PWD/lib:$PWD/../lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/../dsp;$PWD/../lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
rm -rf output_decode_profile
../bin/qnn-net-run \
  --backend ../lib/libQnnHtp.so \
  --model lib/libtiny_block_decode.so \
  --input_list input/device_decode_input_list.txt \
  --output_dir output_decode_profile \
  --input_data_type float \
  --output_data_type float_only \
  --perf_profile burst \
  --profiling_level detailed \
  --num_inferences 10 \
  --log_level info
'
```

运行成功后，设备端目录中会出现：

```text
/data/local/tmp/qnn/tiny_llm_block/output_prefill_profile/qnn-profiling-data_0.log
/data/local/tmp/qnn/tiny_llm_block/output_decode_profile/qnn-profiling-data_0.log
```

拉回主机并转成 CSV：

```bash
mkdir -p \
  tiny_llm_block/device_output/prefill \
  tiny_llm_block/device_output/decode

adb pull \
  "$QNN_PHONE_ROOT/tiny_llm_block/output_prefill_profile/qnn-profiling-data_0.log" \
  tiny_llm_block/device_output/prefill/
adb pull \
  "$QNN_PHONE_ROOT/tiny_llm_block/output_decode_profile/qnn-profiling-data_0.log" \
  tiny_llm_block/device_output/decode/

qnn-profile-viewer \
  --input_log tiny_llm_block/device_output/prefill/qnn-profiling-data_0.log \
  --output tiny_llm_block/device_output/prefill/profile.csv
qnn-profile-viewer \
  --input_log tiny_llm_block/device_output/decode/qnn-profiling-data_0.log \
  --output tiny_llm_block/device_output/decode/profile.csv

python tiny_llm_block/compare_profile.py
```

如果 `qnn-profiling-data_0.log` 已经在本地存在，就不需要重新跑设备端，只需要从
`qnn-profile-viewer` 开始：

```bash
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-profile-viewer \
  --input_log tiny_llm_block/device_output/prefill/qnn-profiling-data_0.log \
  --output tiny_llm_block/device_output/prefill/profile.csv

/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-profile-viewer \
  --input_log tiny_llm_block/device_output/decode/qnn-profiling-data_0.log \
  --output tiny_llm_block/device_output/decode/profile.csv

python tiny_llm_block/compare_profile.py
```

如果当前 shell 找不到 `qnn-profile-viewer`，先初始化 QAIRT 环境，或者直接使用完整路径：

```bash
export QAIRT_SDK_ROOT=/home/lingbok/Qualcomm/qairt/2.47.0.260601
source "$QAIRT_SDK_ROOT/bin/envsetup.sh"

"$QAIRT_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer" \
  --input_log tiny_llm_block/device_output/prefill/qnn-profiling-data_0.log \
  --output tiny_llm_block/device_output/prefill/profile.csv
```

#### CSV 内容怎么读

`qnn-profile-viewer` 生成的 CSV 前几行是版本和来源信息，真正的数据表从下面这一行
开始：

```text
Msg Timestamp, Message, Time, Unit of Measurement, Timing Source, Event Level, Event Identifier
```

各列含义：

| 列 | 含义 |
|---|---|
| `Msg Timestamp` | profiling 事件时间戳 |
| `Message` | 阶段名，例如 `INIT`、`COMPOSE GRAPHS`、`FINALIZE`、`EXECUTE`、`DE-INIT` |
| `Time` | 耗时、cycle 数或计数值 |
| `Unit of Measurement` | 单位，例如 `US`、`CYCLES`、`COUNT`、`INF/SEC` |
| `Timing Source` | 统计来源，常见为 `NETRUN` 或 `BACKEND` |
| `Event Level` | `ROOT` 表示整图级事件，`SUB-EVENT` 表示子事件或单个 op |
| `Event Identifier` | 事件名字，例如 `QNN accelerator (execute) time` 或 `_Softmax:OpId_127 (cycles)` |

最常看的不是 `INIT` 和 `FINALIZE`，而是 `EXECUTE`。一次 inference 会对应一组
`EXECUTE` 行；使用 `--num_inferences 10` 时，CSV 里会有 10 组这样的记录。

整图性能先看 `EXECUTE + ROOT`：

```text
EXECUTE,3301,US,NETRUN,ROOT,Graph 0: tiny_block_decode
EXECUTE,1851,US,BACKEND,ROOT,QNN accelerator (execute) time
EXECUTE,149241,CYCLES,BACKEND,ROOT,Accelerator (execute) time (cycles)
EXECUTE,3278,US,BACKEND,ROOT,QNN (execute) time
```

这些行的含义：

| 指标 | 含义 |
|---|---|
| `NETRUN` | `qnn-net-run` 看到的端到端 graph execute 时间，包含更多运行框架、RPC、I/O 或调度开销 |
| `QNN (execute) time` | QNN backend 侧执行一次 graph 的时间 |
| `QNN accelerator (execute) time` | QNN 统计的 accelerator execute 相关时间 |
| `Accelerator (execute) time (cycles)` | HTP accelerator 上执行的 cycle 数，适合比较计算负载 |
| `Number of HVX threads used` | backend 本次执行使用的 HVX 线程数 |

定位热点再看 `EXECUTE + SUB-EVENT + CYCLES`：

```text
EXECUTE,13807,CYCLES,BACKEND,SUB-EVENT,_Transpose_3:OpId_119 (cycles)
EXECUTE,7492,CYCLES,BACKEND,SUB-EVENT,_MatMul_3:OpId_120 (cycles)
EXECUTE,6984,CYCLES,BACKEND,SUB-EVENT,_Softmax:OpId_127 (cycles)
```

这些行表示某个 QNN op 在 accelerator 上消耗的 cycle。`OpId_*` 是 converter
生成的 graph node 编号，名字来自 ONNX/QNN 中间图，不一定直接等于 PyTorch
模块名。很多 `Reshape`、部分 layout 转换或被融合的节点可能显示为 0 cycles，
这通常表示该节点没有单独形成可计时的 accelerator kernel，或者被其他 op/fusion
吸收。

`compare_profile.py` 会读取两个 `profile.csv`，筛选 `EXECUTE + ROOT` 中的三类
指标，并跳过第一次 warm-up 后取 median：

```text
Accelerator (execute) time (cycles)
QNN accelerator (execute) time
NETRUN graph execute time
```

| 模式 | accelerator cycles | QNN accelerator | NetRun |
|---|---:|---:|---:|
| prefill seq=32 | 360,575 | 2,126 us | 3,691 us |
| decode token=1, past=32 | 149,241 | 1,850 us | 3,301 us |

decode 的 accelerator cycles 明显少于 prefill，但 QNN accelerator 和 NetRun
延迟下降有限，说明单 token 阶段已明显受到 KV I/O、layout conversion、调度和
小算子固定开销影响。

## 后续阶段

1. 比较 W8A8 与 W8A16 的逐层误差和 cycles。
2. 扩展 prefill sequence length 与 KV cache length sweep。
3. 将 Linear 逐步替换为仓库中的自定义 MatMul OpPackage。
