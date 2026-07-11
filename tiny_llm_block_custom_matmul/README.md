# Tiny LLM Block Custom MatMul

本目录用于把 `tiny_llm_block` 中的 QNN builtin MatMul/Linear 路径，逐步替换为
仓库里的自定义 QNN HTP MatMul OpPackage。它不是直接上完整 Qwen，而是先在一个
固定 shape、可验证、可 profiling 的 tiny decoder block 上完成闭环。

## 为什么从这里开始

直接修改 Qwen 级别的大模型会同时遇到模型切分、KV cache、量化、layout、动态
shape、converter 支持和自定义 op 接口等多个变量。`tiny_llm_block` 已经具备：

- 固定 shape 的 prefill/decode ONNX。
- NumPy/PyTorch/ONNX/QNN builtin 多级 reference。
- Android HTP 运行脚本和 raw 输入输出。
- `qnn-profile-viewer` CSV profiling 结果。

因此这里的目标是先做一个“小而完整”的算子替换实验：

```text
tiny_llm_block builtin QNN graph
  -> 选择一个 MatMul/Linear 节点
  -> 替换为 custom HTP MatMul OpPackage
  -> 比较 raw 输出正确性
  -> 比较 qnn-profile-viewer CSV 性能
  -> 再推广到更多 Linear，最后再迁移到 Qwen block
```

## Baseline 输入

当前 baseline 来自：

```text
tiny_llm_block/model/tiny_block_prefill_seq32.onnx
tiny_llm_block/model/tiny_block_decode_past32.onnx
tiny_llm_block/generated/
tiny_llm_block/model_libs/
tiny_llm_block/test_data/
tiny_llm_block/device_output/
```

其中 `tiny_llm_block` 保持为 builtin QNN baseline；本目录只保存替换实验相关的
分析脚本、改图产物、编译产物和 profiling 结果。

## 候选自定义算子

第一阶段优先复用：

```text
qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store_multithread
```

这个 OpPackage 的核心接口是：

```text
lhs:    FP16 [B, H, M, K]
rhs:    FP16 [B, H, K, N]
output: FP32 [B, H, M, N]
```

它适合替换 transformer 中形状规整的 projection/MLP MatMul，但和
`tiny_llm_block` 当前 FP32 ONNX graph 之间仍有几个接口差异需要处理：

- ONNX projection 通常是 rank-3 `[B, S, hidden] x [hidden, out]`，custom op
  需要 rank-4 `[B, H, M, K] x [B, H, K, N]`。
- custom op 输入是 FP16，baseline block 是 FP32；替换点前后需要明确 Cast 策略。
- QNN builtin converter 可能把某些 linear 映射成 `FullyConnected`，不一定保留
  原始 ONNX `MatMul` 名字。
- 当前 custom op 不支持 broadcasting 和 transpose，weight layout 必须提前对齐。

所以第一版实验不要一次替换所有 projection，而是先挑一个 shape 最简单、输出容易
对齐的 MatMul 做单点替换。

## 推荐阶段

### 1. 盘点 ONNX MatMul

先列出 prefill/decode ONNX 中所有 MatMul/Gemm 节点及其 shape：

```bash
PYTHONPATH=/home/lingbok/anaconda3/envs/qairt-2.47/lib/python3.12/site-packages \
python tiny_llm_block_custom_matmul/tools/inspect_onnx_matmuls.py \
  tiny_llm_block/model/tiny_block_prefill_seq32.onnx

PYTHONPATH=/home/lingbok/anaconda3/envs/qairt-2.47/lib/python3.12/site-packages \
python tiny_llm_block_custom_matmul/tools/inspect_onnx_matmuls.py \
  tiny_llm_block/model/tiny_block_decode_past32.onnx
```

盘点时重点看：

- `MatMul` 的左输入是否是 activation。
- 右输入是否是 initializer weight。
- `M/K/N` 是否匹配 custom op 的 `[B,H,M,K] x [B,H,K,N]` 约定。
- 输出是否后接 Add/Bias、Reshape、Transpose 或残差路径。

### 2. 选择第一个替换目标

优先选择 MLP 或 attention projection 中最规整的 dense MatMul：

```text
[1, seq, 256] x [256, 256]
[1, seq, 256] x [256, 768]
[1, seq, 768] x [768, 256]
```

prefill 的 `seq=32` 通常比 decode 的 `seq=1` 更适合先验证性能，因为 M 方向有更多
行可以让 8-row HVX kernel 和 QHPI self-slicing 发挥作用。

### 3. 定义替换图

第一版可以先做手写 QNN model source，而不是马上改 ONNX converter：

```text
APP_WRITE hidden/input
  -> Cast FP32 to FP16
  -> Reshape to [B, 1, M, K]
  -> custom MatMulQhpiHvx8RowFp32StoreMultithread
  -> Reshape back to [B, M, N]
  -> 后续 builtin op 或 APP_READ
```

等单点实验跑通以后，再考虑：

- 通过 ONNX custom domain 表达自定义 op。
- 用 converter package config 接入 custom op。
- 或者直接生成/修改 QNN model C++ source，把指定 node 替换成 custom op。

### 4. 正确性比较

每次替换后都要和 baseline 比较：

```text
tiny_llm_block/test_data/*_expected.raw
tiny_llm_block/device_output/*/Result_0/*.raw
```

如果引入 FP16 输入，误差会比纯 FP32 builtin 大。第一版建议记录：

- `max_abs_error`
- `mean_abs_error`
- `allclose(atol, rtol)`
- 是否出现 NaN/Inf
- 替换点前后的 tensor shape 和 dtype

### 5. 性能比较

运行 `qnn-net-run` 时打开：

```text
--profiling_level detailed
--num_inferences 10
```

再用：

```bash
qnn-profile-viewer \
  --input_log qnn-profiling-data_0.log \
  --output profile.csv
```

比较时优先看 graph-level wall time：

```text
identifier=Accelerator (execute) time
```

custom op 自身的 cycles 在多线程 self-slicing 下可能是 worker 聚合值，不要直接当成
端到端延迟。

## 目录约定

```text
tiny_llm_block_custom_matmul/
  README.md
  tools/          # 分析/辅助脚本
  model/          # 后续保存改图后的 ONNX 或中间模型
  generated/      # 后续保存 custom graph 的 QNN source/bin/json
  model_libs/     # 后续保存 Android model library
  test_data/      # 后续保存替换实验的 raw/input-list
  device_output/  # 后续保存设备输出和 profiling CSV
```

## 当前第一步

先运行 `tools/inspect_onnx_matmuls.py`，确定 prefill/decode 中哪些 MatMul 最适合作为
第一个 custom op 替换点。拿到这张表以后，再进入 graph rewriting 或手写 QNN model
source。

## 当前 ONNX 盘点结果

`tiny_block_prefill_seq32.onnx` 中有 9 个 MatMul：

| 节点 | 输入 shape | 权重/右输入 shape | 输出 shape | 说明 |
|---|---|---|---|---|
| `/MatMul` | `[1, 32, 256]` | `q_proj_weight [256, 256]` | `[1, 32, 256]` | Q projection |
| `/MatMul_1` | `[1, 32, 256]` | `k_proj_weight [256, 64]` | `[1, 32, 64]` | K projection |
| `/MatMul_2` | `[1, 32, 256]` | `v_proj_weight [256, 64]` | `[1, 32, 64]` | V projection |
| `/MatMul_3` | `[1, 8, 32, 32]` | dynamic K transpose | inferred dynamic | attention score |
| `/MatMul_4` | softmax output | dynamic V | inferred dynamic | attention value |
| `/MatMul_5` | `[1, 32, 256]` | `o_proj_weight [256, 256]` | `[1, 32, 256]` | output projection |
| `/MatMul_6` | `[1, 32, 256]` | `gate_proj_weight [256, 768]` | `[1, 32, 768]` | MLP gate projection |
| `/MatMul_7` | `[1, 32, 256]` | `up_proj_weight [256, 768]` | `[1, 32, 768]` | MLP up projection |
| `/MatMul_8` | `[1, 32, 768]` | `down_proj_weight [768, 256]` | `[1, 32, 256]` | MLP down projection |

`tiny_block_decode_past32.onnx` 也有同样 9 个 MatMul，但 sequence 维度是
`1`，例如 projection 变成 `[1, 1, 256] x [256, N]`。decode 的 M 太小，
不适合作为第一版 8-row custom op 性能验证入口。

## 第一版替换建议

第一版建议从 prefill 的 `/MatMul`，也就是 `q_proj_weight [256, 256]` 开始：

```text
baseline ONNX shape:
  [1, 32, 256] x [256, 256] -> [1, 32, 256]

custom op shape:
  lhs [1, 1, 32, 256]
  rhs [1, 1, 256, 256]
  out [1, 1, 32, 256]
```

选择它的原因：

- `M=32` 能被 8-row tile 整除，可以直接验证 multi-row kernel。
- `K=256, N=256` 和现有 custom op demo 的核心 shape 最接近。
- 权重是 initializer，不需要处理动态右输入。
- 输出 rank 和 hidden size 都不变，后续接回 graph 最简单。

如果这个点跑通，再扩展到 `/MatMul_5` 的 `o_proj_weight [256, 256]`，然后再做
`gate/up/down` 这些更大的 MLP projection。attention score/value 的两个动态
MatMul 放到后面，因为它们涉及 mask、Softmax、KV layout 和动态右输入。

## q_proj Standalone Fixture

为了先验证 custom MatMul 的接口和数值，不必一开始就改完整 block。当前目录提供：

```bash
python tiny_llm_block_custom_matmul/tools/generate_q_proj_fixture.py
```

它复用 `tiny_llm_block.export_onnx` 的同一套输入生成逻辑：

- block 权重：`TinyDecoderBlock(seed=42)`。
- prefill hidden input：`np.random.default_rng(7)` 生成 `[1, 32, 256]`。
- lhs：对 hidden 做 `input RMSNorm` 后 reshape 为 `[1, 1, 32, 256]`。
- rhs：`q_proj_weight` reshape 为 `[1, 1, 256, 256]`。

生成内容位于：

```text
tiny_llm_block_custom_matmul/test_data/q_proj_prefill/
  lhs.raw                    # FP32 app input, 后续 QNN graph 内部 Cast 到 FP16
  rhs.raw                    # FP32 app input, 后续 QNN graph 内部 Cast 到 FP16
  lhs_fp16.raw               # 调试用 FP16 lhs
  rhs_fp16.raw               # 调试用 FP16 rhs
  expected_fp32_baseline.raw # 原始 FP32 matmul reference
  expected_fp16_input.raw    # FP16 input + FP32 matmul reference
  expected_q13_kernel.raw    # 按当前 HVX Q13 kernel 路径模拟的 reference
  input_list.txt
```

当前生成结果：

```text
lhs shape: [1, 1, 32, 256]
rhs shape: [1, 1, 256, 256]
expected shape: [1, 1, 32, 256]
FP16-input reference vs Q13-kernel reference:
  max_abs_error  = 0.0026402026
  mean_abs_error = 0.0004553714
```

后续和设备输出比较时，优先对齐 `expected_q13_kernel.raw`，因为现有 HVX kernel 会把
FP16 输入转换成 Q13，累加后再转回 FP32。

## q_proj Standalone 已跑通

### 1. 生成 model library

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c tiny_llm_block_custom_matmul/model/q_proj_prefill_custom_matmul_model.cpp \
  -t aarch64-android \
  -l q_proj_prefill_custom_matmul_model \
  -o tiny_llm_block_custom_matmul/model_libs
```

输出：

```text
tiny_llm_block_custom_matmul/model_libs/aarch64-android/libq_proj_prefill_custom_matmul_model.so
```

### 2. 推送到设备

```bash
adb shell mkdir -p \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/lib \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/input \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/output_q_proj

adb push \
  tiny_llm_block_custom_matmul/model_libs/aarch64-android/libq_proj_prefill_custom_matmul_model.so \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/lib/

adb push \
  tiny_llm_block_custom_matmul/test_data/q_proj_prefill/lhs.raw \
  tiny_llm_block_custom_matmul/test_data/q_proj_prefill/rhs.raw \
  tiny_llm_block_custom_matmul/test_data/q_proj_prefill/sample_app_input_list.txt \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/input/

adb push \
  qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store_multithread/htp/MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage/build/aarch64-android/libQnnMatMulQhpiHvx8RowFp32StoreMultithreadOpPackage.so \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/lib/

adb push \
  qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store_multithread/htp/MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage/build/hexagon-v75/libQnnMatMulQhpiHvx8RowFp32StoreMultithreadOpPackage.so \
  /data/local/tmp/qnn/dsp/
```

这里 ARM package 放在实验目录的 `lib/`，HTP package 放在 `/data/local/tmp/qnn/dsp/`。
运行时 `--op_packages` 中 ARM 项使用完整相对路径，HTP 项使用裸文件名。

### 3. 在 HTP 上运行

推荐从 `/data/local/tmp/qnn` 作为工作目录启动 `qnn-net-run`。这样
`--op_packages` 中 ARM package 的相对路径和 HTP package 的裸文件名都能正确解析。

```bash
adb shell '
cd /data/local/tmp/qnn
rm -rf tiny_llm_block_custom_matmul/output_q_proj_netrun
export LD_LIBRARY_PATH="$PWD/tiny_llm_block_custom_matmul/lib:$PWD/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
./bin/qnn-net-run \
  --backend lib/libQnnHtp.so \
  --model tiny_llm_block_custom_matmul/lib/libq_proj_prefill_custom_matmul_model.so \
  --input_list tiny_llm_block_custom_matmul/input/sample_app_input_list.txt \
  --output_dir tiny_llm_block_custom_matmul/output_q_proj_netrun \
  --input_data_type float \
  --output_data_type float_only \
  --perf_profile burst \
  --profiling_level detailed \
  --num_inferences 10 \
  --log_level info \
  --op_packages tiny_llm_block_custom_matmul/lib/libQnnMatMulQhpiHvx8RowFp32StoreMultithreadOpPackage.so:MatMulQhpiHvx8RowFp32StoreMultithreadOpPackageInterfaceProvider:CPU,libQnnMatMulQhpiHvx8RowFp32StoreMultithreadOpPackage.so:MatMulQhpiHvx8RowFp32StoreMultithreadOpPackageInterfaceProvider:HTP
'
```

`qnn-net-run` 成功日志：

```text
Composing Graphs
Finalizing Graphs
Executing Graphs
Finished Executing Graphs
```

也可以用 `qnn-sample-app` 做更详细的注册日志检查：

```bash
adb shell '
cd /data/local/tmp/qnn
rm -rf tiny_llm_block_custom_matmul/output_q_proj
export LD_LIBRARY_PATH="$PWD/tiny_llm_block_custom_matmul/lib:$PWD/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
./bin/qnn-sample-app \
  --backend lib/libQnnHtp.so \
  --model tiny_llm_block_custom_matmul/lib/libq_proj_prefill_custom_matmul_model.so \
  --op_packages tiny_llm_block_custom_matmul/lib/libQnnMatMulQhpiHvx8RowFp32StoreMultithreadOpPackage.so:MatMulQhpiHvx8RowFp32StoreMultithreadOpPackageInterfaceProvider:CPU,libQnnMatMulQhpiHvx8RowFp32StoreMultithreadOpPackage.so:MatMulQhpiHvx8RowFp32StoreMultithreadOpPackageInterfaceProvider:HTP \
  --input_list tiny_llm_block_custom_matmul/input/sample_app_input_list.txt \
  --output_dir tiny_llm_block_custom_matmul/output_q_proj \
  --input_data_type float \
  --output_data_type float_only \
  --log_level info
'
```

成功日志中的关键行：

```text
Loaded package MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage
Matched kernel 'matmulqhpihvx8rowfp32storemultithread_float_16_Execute'
QnnGraph_execute done. status 0x0
```

### 4. 拉回输出并比较

```bash
adb pull \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/output_q_proj_netrun/Result_0 \
  tiny_llm_block_custom_matmul/device_output/q_proj_prefill_netrun/

adb pull \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/output_q_proj_netrun/qnn-profiling-data_0.log \
  tiny_llm_block_custom_matmul/device_output/q_proj_prefill_netrun/

/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-profile-viewer \
  --input_log tiny_llm_block_custom_matmul/device_output/q_proj_prefill_netrun/qnn-profiling-data_0.log \
  --output tiny_llm_block_custom_matmul/device_output/q_proj_prefill_netrun/profile.csv

python tiny_llm_block_custom_matmul/tools/compare_q_proj_output.py \
  --output tiny_llm_block_custom_matmul/device_output/q_proj_prefill_netrun/output.raw
```

当前设备输出与 `expected_q13_kernel.raw` 的比较结果：

```text
elements: 8192
actual range: -1.2756941318511963 1.1606714725494385
expected range: -1.2756941318511963 1.1606714725494385
nonzero: 8192
max_abs_error: 1.1920928955078125e-07
mean_abs_error: 3.14321368932724e-09
allclose_1e-3: True
nan_count: 0
inf_count: 0
```

当前 `qnn-profile-viewer` 结果摘要：

```text
Graph: tinyBlockQProjCustomMatMul
NetRun average:                    6527 us
QNN accelerator average:           5209 us
Accelerator execute average:       4823 us
Accelerator cycles average:        3636002 cycles
HVX threads used:                  4
MatMul custom op average cycles:   3604780 cycles
CastLhsToFp16 average cycles:      3730 cycles
CastRhsToFp16 average cycles:      12792 cycles
```

这说明第一版 `q_proj` standalone custom MatMul 已经完成：

```text
tiny block q_proj fixture
  -> QNN model source
  -> custom HTP OpPackage
  -> Android HTP 执行
  -> 本地 reference 对齐
```

## 完整 prefill graph 替换 q_proj

standalone q_proj 跑通后，下一步是把 custom MatMul 接回完整
`tiny_block_prefill` graph。这里没有重新导出 ONNX，而是 patch converter 已经生成的
QNN model source：

```text
tiny_llm_block/generated/tiny_block_prefill.cpp
  -> tiny_llm_block_custom_matmul/tools/patch_prefill_q_proj_custom.py
  -> tiny_llm_block_custom_matmul/generated/tiny_block_prefill_q_proj_custom.cpp
```

patch 内容只改 q_proj 路径：

```text
原始 converter graph:
  _Mul_2_output_0
    -> _MatMul_pre_reshape [32,256]
    -> FullyConnected(q_proj_weight, q_proj_bias)
    -> _Add_1_output_0_fc [32,256]
    -> _MatMul_post_reshape [1,32,8,32]

替换后:
  _Mul_2_output_0
    -> _MatMul_pre_reshape [32,256]
    -> Reshape [1,1,32,256]
    -> Cast FP16
    -> Cast q_proj_weight [1,1,256,256] to FP16
    -> MatMulQhpiHvx8RowFp32StoreMultithread
    -> Reshape [32,256]
    -> _MatMul_post_reshape [1,32,8,32]
```

`q_proj_bias` 在 tiny block 中全 0，所以替换时没有额外 Add bias；这样语义差异主要
来自 FP16 输入和当前 HVX kernel 的 Q13 计算路径。

### 1. 生成 patched QNN source

```bash
python tiny_llm_block_custom_matmul/tools/patch_prefill_q_proj_custom.py
```

输出：

```text
tiny_llm_block_custom_matmul/generated/tiny_block_prefill_q_proj_custom.cpp
```

### 2. 编译完整 prefill custom model library

继续复用原始 converter 生成的权重 bin：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c tiny_llm_block_custom_matmul/generated/tiny_block_prefill_q_proj_custom.cpp \
  -b tiny_llm_block/generated/tiny_block_prefill.bin \
  -t aarch64-android \
  -l tiny_block_prefill_q_proj_custom \
  -o tiny_llm_block_custom_matmul/model_libs
```

输出：

```text
tiny_llm_block_custom_matmul/model_libs/aarch64-android/libtiny_block_prefill_q_proj_custom.so
```

### 3. 推送并运行完整 graph

从 `/data/local/tmp/qnn` 启动，input-list 使用相对该目录的路径：

```text
tiny_llm_block_custom_matmul/test_data/prefill_from_qnn_root_input_list.txt
```

内容：

```text
hidden_states:=tiny_llm_block/input/prefill_hidden.raw
```

推送：

```bash
adb shell mkdir -p \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/lib \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/input

adb push \
  tiny_llm_block_custom_matmul/model_libs/aarch64-android/libtiny_block_prefill_q_proj_custom.so \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/lib/

adb push \
  tiny_llm_block_custom_matmul/test_data/prefill_from_qnn_root_input_list.txt \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/input/
```

运行：

```bash
adb shell '
cd /data/local/tmp/qnn
rm -rf tiny_llm_block_custom_matmul/output_prefill_q_proj_custom
export LD_LIBRARY_PATH="$PWD/tiny_llm_block_custom_matmul/lib:$PWD/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
./bin/qnn-net-run \
  --backend lib/libQnnHtp.so \
  --model tiny_llm_block_custom_matmul/lib/libtiny_block_prefill_q_proj_custom.so \
  --input_list tiny_llm_block_custom_matmul/input/prefill_from_qnn_root_input_list.txt \
  --output_dir tiny_llm_block_custom_matmul/output_prefill_q_proj_custom \
  --input_data_type float \
  --output_data_type float_only \
  --perf_profile burst \
  --profiling_level detailed \
  --num_inferences 10 \
  --log_level info \
  --op_packages tiny_llm_block_custom_matmul/lib/libQnnMatMulQhpiHvx8RowFp32StoreMultithreadOpPackage.so:MatMulQhpiHvx8RowFp32StoreMultithreadOpPackageInterfaceProvider:CPU,libQnnMatMulQhpiHvx8RowFp32StoreMultithreadOpPackage.so:MatMulQhpiHvx8RowFp32StoreMultithreadOpPackageInterfaceProvider:HTP
'
```

成功日志：

```text
Composing Graphs
Finalizing Graphs
Executing Graphs
Finished Executing Graphs
```

### 4. 拉回并比较

```bash
adb pull \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/output_prefill_q_proj_custom/Result_0 \
  tiny_llm_block_custom_matmul/device_output/prefill_q_proj_custom/

adb pull \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/output_prefill_q_proj_custom/qnn-profiling-data_0.log \
  tiny_llm_block_custom_matmul/device_output/prefill_q_proj_custom/

/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-profile-viewer \
  --input_log tiny_llm_block_custom_matmul/device_output/prefill_q_proj_custom/qnn-profiling-data_0.log \
  --output tiny_llm_block_custom_matmul/device_output/prefill_q_proj_custom/profile.csv

python tiny_llm_block/compare_qnn_output.py \
  prefill tiny_llm_block_custom_matmul/device_output/prefill_q_proj_custom
```

当前完整 graph 输出误差：

| 输出 | max error | mean error | cosine |
|---|---:|---:|---:|
| hidden_out | 0.0247808099 | 0.0032691909 | 0.999762475 |
| present_key | 0.0008549094 | 0.0001221451 | 0.999999940 |
| present_value | 0.0005049631 | 0.0001066229 | 1.000000000 |

`present_key/value` 没有经过 q_proj，因此仍接近原始 builtin baseline；`hidden_out`
经过 attention 和 MLP，q_proj 的 FP16/Q13 误差会被后续 Softmax 和残差路径放大。

当前 `qnn-profile-viewer` 摘要：

```text
Graph: tiny_block_prefill
NetRun average:                    8494 us
QNN accelerator average:           6998 us
Accelerator execute average:       5107 us
Accelerator cycles average:        3935655 cycles
HVX threads used:                  4
q_proj custom MatMul avg cycles:   3610771 cycles
```

对比 `tiny_llm_block` builtin prefill baseline：

```text
builtin prefill:
  accelerator cycles: 360575
  QNN accelerator:    2126 us
  NetRun:             3691 us

q_proj custom prefill:
  accelerator cycles: 3935655
  QNN accelerator:    6998 us
  NetRun:             8494 us
```

这版替换证明了 custom OpPackage 可以接回 transformer block 的真实数据流，但性能还
不是优化目标结果。当前 custom q_proj 自身占用约 3.61M cycles，是下一阶段要优化的
主要热点。

## q_proj LHS-prepack 版本

上面的 multithread 版本中，q_proj custom op 仍在 MatMul reduction 内层重复做：

```text
FP16 LHS scalar -> Q13 -> splat
```

仓库已有一个 `LHS prepack + FP32 store` 版本，可以先验证这个优化方向是否对
tiny block 的 q_proj 有收益：

```text
qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_prepack_fp32_store
```

这个版本的 MatMul 本身不是 multithread，但它可以把 LHS 的 Q13 转换搬到单独的
`MatMulQhpiHvxPackLhs` 节点中：

```text
_MatMul_pre_reshape [32,256]
  -> Reshape [1,1,32,256]
  -> Cast FP16
  -> MatMulQhpiHvxPackLhs
  -> MatMulQhpiHvx8RowLhsPrepackFp32Store
  -> Reshape [32,256]
```

### 1. 生成 LHS-prepack patched source

```bash
python tiny_llm_block_custom_matmul/tools/patch_prefill_q_proj_lhs_prepack.py
```

输出：

```text
tiny_llm_block_custom_matmul/generated/tiny_block_prefill_q_proj_lhs_prepack.cpp
```

### 2. 编译 model library

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c tiny_llm_block_custom_matmul/generated/tiny_block_prefill_q_proj_lhs_prepack.cpp \
  -b tiny_llm_block/generated/tiny_block_prefill.bin \
  -t aarch64-android \
  -l tiny_block_prefill_q_proj_lhs_prepack \
  -o tiny_llm_block_custom_matmul/model_libs
```

输出：

```text
tiny_llm_block_custom_matmul/model_libs/aarch64-android/libtiny_block_prefill_q_proj_lhs_prepack.so
```

### 3. 推送 OpPackage 和 model

```bash
adb push \
  tiny_llm_block_custom_matmul/model_libs/aarch64-android/libtiny_block_prefill_q_proj_lhs_prepack.so \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/lib/

adb push \
  qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_prepack_fp32_store/htp/MatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage/build/aarch64-android/libQnnMatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage.so \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/lib/

adb push \
  qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_prepack_fp32_store/htp/MatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage/build/hexagon-v75/libQnnMatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage.so \
  /data/local/tmp/qnn/dsp/
```

### 4. 运行 LHS-prepack 完整 graph

```bash
adb shell '
cd /data/local/tmp/qnn
rm -rf tiny_llm_block_custom_matmul/output_prefill_q_proj_lhs_prepack
export LD_LIBRARY_PATH="$PWD/tiny_llm_block_custom_matmul/lib:$PWD/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
./bin/qnn-net-run \
  --backend lib/libQnnHtp.so \
  --model tiny_llm_block_custom_matmul/lib/libtiny_block_prefill_q_proj_lhs_prepack.so \
  --input_list tiny_llm_block_custom_matmul/input/prefill_from_qnn_root_input_list.txt \
  --output_dir tiny_llm_block_custom_matmul/output_prefill_q_proj_lhs_prepack \
  --input_data_type float \
  --output_data_type float_only \
  --perf_profile burst \
  --profiling_level detailed \
  --num_inferences 10 \
  --log_level info \
  --op_packages tiny_llm_block_custom_matmul/lib/libQnnMatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage.so:MatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackageInterfaceProvider:CPU,libQnnMatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage.so:MatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackageInterfaceProvider:HTP
'
```

拉回并比较：

```bash
adb pull \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/output_prefill_q_proj_lhs_prepack/Result_0 \
  tiny_llm_block_custom_matmul/device_output/prefill_q_proj_lhs_prepack/

adb pull \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/output_prefill_q_proj_lhs_prepack/qnn-profiling-data_0.log \
  tiny_llm_block_custom_matmul/device_output/prefill_q_proj_lhs_prepack/

/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-profile-viewer \
  --input_log tiny_llm_block_custom_matmul/device_output/prefill_q_proj_lhs_prepack/qnn-profiling-data_0.log \
  --output tiny_llm_block_custom_matmul/device_output/prefill_q_proj_lhs_prepack/profile.csv

python tiny_llm_block/compare_qnn_output.py \
  prefill tiny_llm_block_custom_matmul/device_output/prefill_q_proj_lhs_prepack
```

LHS-prepack 与 multithread 版本的数值结果一致：

| 输出 | max error | mean error | cosine |
|---|---:|---:|---:|
| hidden_out | 0.0247808099 | 0.0032691909 | 0.999762475 |
| present_key | 0.0008549094 | 0.0001221451 | 0.999999940 |
| present_value | 0.0005049631 | 0.0001066229 | 1.000000000 |

### 5. 统一 profiling 对比

为了避免 `qnn-profile-viewer` 的 average 受第一轮 warm-up 和电源/VTCM acquire 影响，
用下面脚本统一读取 CSV，并跳过第一轮后取 median：

```bash
python tiny_llm_block_custom_matmul/tools/compare_prefill_profiles.py
```

当前结果：

```text
builtin                       root_cycles=   360575 qnn_us= 2126 netrun_us= 3691 q_proj_cycles=        0 gate_proj_cycles=     3963
custom_multithread            root_cycles=  3933728 qnn_us= 6717 netrun_us= 8241 q_proj_cycles=  3618124 gate_proj_cycles=     2731
custom_lhs_prepack            root_cycles=  1682834 qnn_us= 7960 netrun_us=10762 q_proj_cycles=  1338650 gate_proj_cycles=     2574
custom_lhs_prepack_mt         root_cycles=  2670623 qnn_us= 6074 netrun_us= 7560 q_proj_cycles=  2289029 gate_proj_cycles=     5413
custom_both_prepack           root_cycles=  1752130 qnn_us= 8120 netrun_us=10922 q_proj_cycles=  1401686 gate_proj_cycles=     2576
custom_gate_lhs_prepack       root_cycles=  4465563 qnn_us=15083 netrun_us=17784 q_proj_cycles=        0 gate_proj_cycles=  4042572
custom_gate_lhs_prepack_mt    root_cycles=  6941176 qnn_us= 8979 netrun_us=11649 q_proj_cycles=        0 gate_proj_cycles=  6560063
```

结论：

- LHS-prepack 明显降低了 q_proj 内核本身的 cycles：
  `3.62M -> 1.34M`，约 `2.70x`。
- `PackLhs` 只有约 `8K cycles`，成本很小。
- `custom_lhs_prepack_mt` 是组合版本：

```text
LHS prepack + FP32 store + QHPI self-slicing multithread
```

它把 `MatMulQhpiHvx8RowLhsPrepackFp32Store` 的 kernel 改成
`.multithreaded = true`，并用 `qhpi_num_slices(handle)` /
`qhpi_slice_number(handle)` 划分 8-row tiles。

组合版本的结果比较微妙：

- 相比第一版 `custom_multithread`，`q_proj_cycles` 从 `3.62M` 降到
  `2.29M`，`NetRun` 从 `8241 us` 降到 `7560 us`。
- 相比单线程 `custom_lhs_prepack`，`q_proj_cycles` 反而从 `1.34M`
  上升到 `2.29M`，但 `NetRun` 从 `10762 us` 降到 `7560 us`。
- 对 `M=32, K=256, N=256` 这种很小的 prefill q_proj 来说，多线程
  self-slicing 会引入调度/同步和带宽竞争成本；它改善 wall time，但没有让
  kernel cycles 继续下降。

### 6. q_proj both-prepack 版本

继续验证 RHS/weight prepack，把 q_proj 替换为：

```text
hidden_states
  -> Reshape [1,1,32,256]
  -> Cast FP16
  -> MatMulQhpiHvxPackLhs
  -> lhs_q13_bits

q_proj_weight [1,1,256,256]
  -> Cast FP16
  -> MatMulQhpiHvxPackRhs
  -> rhs_q13_bits

lhs_q13_bits + rhs_q13_bits
  -> MatMulQhpiHvx8RowBothPrepackFp32Store
  -> Reshape [32,256]
```

生成、编译和运行链路：

```bash
python tiny_llm_block_custom_matmul/tools/patch_prefill_q_proj_both_prepack.py

PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c tiny_llm_block_custom_matmul/generated/tiny_block_prefill_q_proj_both_prepack.cpp \
  -b tiny_llm_block/generated/tiny_block_prefill.bin \
  -t aarch64-android \
  -l tiny_block_prefill_q_proj_both_prepack \
  -o tiny_llm_block_custom_matmul/model_libs
```

设备端运行时需要注册 both-prepack package：

```text
libQnnMatMulQhpiHvx8RowBothPrepackFp32StoreOpPackage.so:
MatMulQhpiHvx8RowBothPrepackFp32StoreOpPackageInterfaceProvider:CPU

libQnnMatMulQhpiHvx8RowBothPrepackFp32StoreOpPackage.so:
MatMulQhpiHvx8RowBothPrepackFp32StoreOpPackageInterfaceProvider:HTP
```

正确性：

```text
prefill hidden: max=2.47808099e-02 mean=3.26919090e-03 cosine=0.999762475
prefill key: max=8.54909420e-04 mean=1.22145138e-04 cosine=0.999999940
prefill value: max=5.04963100e-04 mean=1.06622887e-04 cosine=1.000000000
```

both-prepack 的实测结果：

```text
custom_lhs_prepack  q_proj_cycles=1.338M netrun=10762 us
custom_both_prepack q_proj_cycles=1.402M netrun=10922 us
```

也就是说，RHS prepack 在 full prefill graph 里没有带来收益，反而略慢。这里有两个
值得记录的观察：

- `pack_rhs_cycles` 只有约 `296 cycles`，说明这个 RHS pack 节点本身在 execute
  阶段非常轻，可能因为 `q_proj_weight` 是静态权重，QNN/HTP 对这条路径做了很强
  的常量处理或调度优化。
- 真正的 MatMul kernel 没有下降：`1.338M -> 1.402M`。对当前 8-row kernel 来说，
  RHS 的 inline FP16/Q13 转换可能已经能和 MAC 流水重叠；拆成独立 RHS bit-carrier
  反而增加了读中间 tensor 和 graph 调度成本。

### 7. gate_proj LHS-prepack 版本

为了测试更大的 projection，把 MLP 里的 `gate_proj` 从 builtin `FullyConnected`
替换为 custom MatMul。这个节点的 shape 是：

```text
input:  [32, 256]
weight: [768, 256]  // FullyConnected layout: [out, in]
output: [32, 768]
```

和 q_proj 不同，`gate_proj_weight` 不是方阵，不能直接 reshape 成 custom MatMul
要的 `[K, N]`。patch 里先把 weight 变成 `[1,1,768,256]`，再插入：

```text
Transpose perm=[0,1,3,2]
```

得到 `[1,1,256,768]` 后再 Cast FP16，接入
`MatMulQhpiHvx8RowLhsPrepackFp32Store`。

生成脚本：

```bash
python tiny_llm_block_custom_matmul/tools/patch_prefill_gate_proj_lhs_prepack.py
python tiny_llm_block_custom_matmul/tools/patch_prefill_gate_proj_lhs_prepack_multithread.py
```

正确性：

```text
prefill hidden: max=9.99987125e-04 mean=8.09368066e-05 cosine=0.999999821
prefill key: max=8.54909420e-04 mean=1.22145138e-04 cosine=0.999999940
prefill value: max=5.04963100e-04 mean=1.06622887e-04 cosine=1.000000000
```

实测结果：

```text
builtin gate_proj                 gate_proj_cycles=0.004M netrun= 3691 us
custom_gate_lhs_prepack           gate_proj_cycles=4.043M netrun=17784 us
custom_gate_lhs_prepack_mt        gate_proj_cycles=6.560M netrun=11649 us
```

结论：

- 非方阵 weight transpose 路径是正确的，数值没有问题。
- `gate_proj` 的 custom kernel cycles 约为 q_proj LHS-prepack 的 3 倍，
  这和 `N=768` 是 `N=256` 的 3 倍一致，说明当前 kernel 主要按 N 线性放大。
- multithread 版本让 wall time 从 `17784 us` 降到 `11649 us`，但 kernel
  cycles 从 `4.04M` 增到 `6.56M`；它改善了端到端等待/调度表现，但没有改善
  kernel 本身效率。
- builtin `FullyConnected` 的 `gate_proj_cycles` 只有几千 cycles，说明 QNN HTP
  builtin 对静态 weight、layout、tiling 或 fusion 做了非常强的内部优化。

### 8. q_proj LHS tile-cache 融合实验

前面的 LHS-prepack 实验把 `PackLhs` 做成了一个独立 graph 节点。这个版本把
`PackLhs + MatMul` 融合到同一个 custom op 内部：

```text
FP16 LHS/RHS input
-> op 内部按 8 行 tile 把 LHS 转成 Q13 小缓存
-> HVX MatMul 内层循环直接读 Q13 LHS cache
-> FP32 output
```

对应的 op package：

```text
qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_tile_cache_fp32_store/
```

对应的 graph patch：

```bash
python tiny_llm_block_custom_matmul/tools/patch_prefill_q_proj_lhs_tile_cache.py
```

它生成：

```text
tiny_llm_block_custom_matmul/generated/tiny_block_prefill_q_proj_lhs_tile_cache.cpp
tiny_llm_block_custom_matmul/model_libs/aarch64-android/libtiny_block_prefill_q_proj_lhs_tile_cache.so
```

按 `tiny_llm_block/compare_qnn_output.py` 的 reference 口径做数值对比：

```text
prefill hidden: max=2.47808099e-02 mean=3.26919090e-03 cosine=0.999762475
prefill key: max=8.54909420e-04 mean=1.22145138e-04 cosine=0.999999940
prefill value: max=5.04963100e-04 mean=1.06622887e-04 cosine=1.000000000
```

这个误差来自 q_proj custom kernel 内部的 FP16->Q13 近似，和前面 custom q_proj
实验属于同一类误差。相对设备 builtin prefill 输出，`present_key` 和 `present_value`
没有被替换，所以是逐元素一致；主要差异集中在 `hidden_out`。

横向 profiling 摘要：

```text
builtin                  root_cycles=   360575 qnn_us= 2126 netrun_us= 3691 q_proj_cycles=        0
custom_lhs_prepack       root_cycles=  1682834 qnn_us= 7960 netrun_us=10762 q_proj_cycles=  1338650
custom_both_prepack      root_cycles=  1752130 qnn_us= 8120 netrun_us=10922 q_proj_cycles=  1401686
custom_lhs_tile_cache    root_cycles=   480235 qnn_us= 4821 netrun_us= 6238 q_proj_cycles=   146244
```

结论：

- tile-cache 版本明显有效，q_proj custom kernel 从百万级 cycles 降到约 `146k`
  cycles。
- graph-level `root_cycles` 从 `1.68M` 降到 `480k`，已经接近 builtin 的
  `360k`。
- 端到端 `qnn_us/netrun_us` 仍慢于 builtin，说明剩余差距主要不只是 MatMul 内核，
  还包括外部 OpPackage 调度、额外 Cast/Reshape、以及 builtin 内部预处理/fusion
  路径。
- 这个版本比“独立 PackLhs 节点”更值得继续作为 custom q_proj 优化基线。

### 9. 实验产物索引和复现模板

当前已经跑过的完整 prefill graph 替换实验如下：

| 实验 | patch 脚本 | generated source | model lib 名称 | OpPackage |
|---|---|---|---|---|
| q_proj custom multithread | `tools/patch_prefill_q_proj_custom.py` | `generated/tiny_block_prefill_q_proj_custom.cpp` | `libtiny_block_prefill_q_proj_custom.so` | `MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage` |
| q_proj LHS-prepack | `tools/patch_prefill_q_proj_lhs_prepack.py` | `generated/tiny_block_prefill_q_proj_lhs_prepack.cpp` | `libtiny_block_prefill_q_proj_lhs_prepack.so` | `MatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage` |
| q_proj LHS-prepack multithread | `tools/patch_prefill_q_proj_lhs_prepack_multithread.py` | `generated/tiny_block_prefill_q_proj_lhs_prepack_multithread.cpp` | `libtiny_block_prefill_q_proj_lhs_prepack_multithread.so` | `MatMulQhpiHvx8RowLhsPrepackFp32StoreMultithreadOpPackage` |
| q_proj both-prepack | `tools/patch_prefill_q_proj_both_prepack.py` | `generated/tiny_block_prefill_q_proj_both_prepack.cpp` | `libtiny_block_prefill_q_proj_both_prepack.so` | `MatMulQhpiHvx8RowBothPrepackFp32StoreOpPackage` |
| q_proj LHS tile-cache | `tools/patch_prefill_q_proj_lhs_tile_cache.py` | `generated/tiny_block_prefill_q_proj_lhs_tile_cache.cpp` | `libtiny_block_prefill_q_proj_lhs_tile_cache.so` | `MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage` |
| gate_proj LHS-prepack | `tools/patch_prefill_gate_proj_lhs_prepack.py` | `generated/tiny_block_prefill_gate_proj_lhs_prepack.cpp` | `libtiny_block_prefill_gate_proj_lhs_prepack.so` | `MatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage` |
| gate_proj LHS-prepack multithread | `tools/patch_prefill_gate_proj_lhs_prepack_multithread.py` | `generated/tiny_block_prefill_gate_proj_lhs_prepack_multithread.cpp` | `libtiny_block_prefill_gate_proj_lhs_prepack_multithread.so` | `MatMulQhpiHvx8RowLhsPrepackFp32StoreMultithreadOpPackage` |

统一生成方式：

```bash
python tiny_llm_block_custom_matmul/tools/<patch_script>.py
```

统一编译方式：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c tiny_llm_block_custom_matmul/generated/<generated_source>.cpp \
  -b tiny_llm_block/generated/tiny_block_prefill.bin \
  -t aarch64-android \
  -l <model_lib_name_without_lib_prefix_and_so_suffix> \
  -o tiny_llm_block_custom_matmul/model_libs
```

设备端统一运行方式：

```bash
adb push \
  tiny_llm_block_custom_matmul/model_libs/aarch64-android/lib<model_lib>.so \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/lib/

adb push \
  qnn_custom_ops/<custom_op_dir>/htp/<OpPackage>/build/aarch64-android/libQnn<OpPackage>.so \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/lib/

adb push \
  qnn_custom_ops/<custom_op_dir>/htp/<OpPackage>/build/hexagon-v75/libQnn<OpPackage>.so \
  /data/local/tmp/qnn/dsp/

adb shell '
cd /data/local/tmp/qnn
rm -rf tiny_llm_block_custom_matmul/<device_output_dir>
export LD_LIBRARY_PATH="$PWD/tiny_llm_block_custom_matmul/lib:$PWD/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
./bin/qnn-net-run \
  --backend lib/libQnnHtp.so \
  --model tiny_llm_block_custom_matmul/lib/lib<model_lib>.so \
  --input_list tiny_llm_block_custom_matmul/input/prefill_from_qnn_root_input_list.txt \
  --output_dir tiny_llm_block_custom_matmul/<device_output_dir> \
  --input_data_type float \
  --output_data_type float_only \
  --perf_profile burst \
  --profiling_level detailed \
  --num_inferences 10 \
  --log_level info \
  --op_packages tiny_llm_block_custom_matmul/lib/libQnn<OpPackage>.so:<OpPackage>InterfaceProvider:CPU,libQnn<OpPackage>.so:<OpPackage>InterfaceProvider:HTP
'
```

统一拉回和分析方式：

```bash
mkdir -p tiny_llm_block_custom_matmul/device_output/<local_output_dir>

adb pull \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/<device_output_dir>/Result_0 \
  tiny_llm_block_custom_matmul/device_output/<local_output_dir>/

adb pull \
  /data/local/tmp/qnn/tiny_llm_block_custom_matmul/<device_output_dir>/qnn-profiling-data_0.log \
  tiny_llm_block_custom_matmul/device_output/<local_output_dir>/qnn-profiling-data_0.log

/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-profile-viewer \
  --input_log tiny_llm_block_custom_matmul/device_output/<local_output_dir>/qnn-profiling-data_0.log \
  --output tiny_llm_block_custom_matmul/device_output/<local_output_dir>/profile.csv

python tiny_llm_block/compare_qnn_output.py \
  prefill tiny_llm_block_custom_matmul/device_output/<local_output_dir>

python tiny_llm_block_custom_matmul/tools/compare_prefill_profiles.py
```

其中 `<device_output_dir>` 和 `<local_output_dir>` 当前使用：

| 实验 | output dir |
|---|---|
| q_proj custom multithread | `output_prefill_q_proj_custom` / `prefill_q_proj_custom` |
| q_proj LHS-prepack | `output_prefill_q_proj_lhs_prepack` / `prefill_q_proj_lhs_prepack` |
| q_proj LHS-prepack multithread | `output_prefill_q_proj_lhs_prepack_multithread` / `prefill_q_proj_lhs_prepack_multithread` |
| q_proj both-prepack | `output_prefill_q_proj_both_prepack` / `prefill_q_proj_both_prepack` |
| q_proj LHS tile-cache | `output_prefill_q_proj_lhs_tile_cache` / `prefill_q_proj_lhs_tile_cache` |
| gate_proj LHS-prepack | `output_prefill_gate_proj_lhs_prepack` / `prefill_gate_proj_lhs_prepack` |
| gate_proj LHS-prepack multithread | `output_prefill_gate_proj_lhs_prepack_multithread` / `prefill_gate_proj_lhs_prepack_multithread` |

### 10. Profiling 口径注意事项

为了进一步确认 builtin `FullyConnected` 为什么看起来只有几千 cycles，增加了一个
subevent 分析脚本：

```bash
python tiny_llm_block_custom_matmul/tools/analyze_profile_subevents.py
```

当前输出摘要：

```text
== builtin ==
root_cycles: 360575
  60482  _MatMul_3
  11425  _MatMul_4
   3963  _MatMul_6   // gate_proj builtin FullyConnected
   2862  _MatMul_7   // up_proj builtin FullyConnected
   2283  _MatMul_8   // down_proj builtin FullyConnected
      0  _MatMul     // q_proj builtin FullyConnected

== gate_lhs ==
root_cycles: 4465563
4042572  _MatMul_6  // custom gate_proj
```

这里有一个重要结论：builtin graph 的 per-node subevent cycles 不能和 custom op
的 per-node cycles 做一比一公平对比。原因是：

- builtin `FullyConnected` 可能被 QNN HTP graph optimizer 改写、融合、预打包或
  映射到内部高效路径。
- 一些 builtin projection 节点在 CSV 中显示为 `0 cycles`，但它们不可能没有计算；
  这说明 subevent 计数并不一定完整归因到原始 node 名称。
- custom op 是外部 OpPackage，profiling 更直接地把耗时记在 custom node 上。

因此后续比较优先级应该是：

```text
1. graph-level root_cycles / qnn_us / netrun_us
2. custom op 自身 cycles，用来找 custom kernel 内部瓶颈
3. builtin subevent cycles，只作为定位参考，不作为绝对性能真相
```

所以这个实验的价值不是“已经超过 builtin”，而是证明了完整替换链路：

```text
converter 生成 QNN C++
-> patch q_proj
-> custom package CPU/HTP 注册
-> Android qnn-net-run
-> qnn-profile-viewer
-> 数值和性能对比
```

下一步如果继续优化，优先级应该是继续减少 custom op 的图级开销和静态 weight 开销，
而不是继续盲目加线程。更可能有效的方向包括：

- 对静态 weight 做真正离线 prepack，把 RHS Q13 bits 直接写进 model bin，
  而不是 runtime graph 里再加一个 `PackRhs` 节点。
- 尝试把 q_proj 的 Cast/Reshape 也吸收到 custom op 或 converter patch 中进一步减少
  graph 节点。
- 继续研究 QNN builtin 的 graph-level 优化和权重预处理路径，但注意不要只看
  builtin subevent cycles，因为这类计数可能没有完整归因。
