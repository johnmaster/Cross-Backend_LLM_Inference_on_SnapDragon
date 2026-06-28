# W4FP16 Blockwise MatMul

## 先纠正术语

QAIRT 2.47 HTP 文档明确列出的是：

```text
W8/W4/W2 FP16-activation Block Quantization
BW_FLOAT_BLOCK encoding
```

因此本实验应称为 **W4FP16a**，不是 W4A16。后者通常表示 INT16
activation，不能与 FP16 activation 混用。

## Block 定义

RHS shape：

```text
[1, 1, K=256, N=256]
```

沿 K 轴分块：

```text
axis = 2
block_size = 32 / 64 / 128
```

每个 block、每个输出列有独立 symmetric INT4 scale：

```text
scale = max(abs(block)) / 7
q = clamp(round(weight / scale), -8, 7)
weight_hat = q * scale
```

## 生成 overrides

```bash
python qnn_quantization/11_w4fp16_blockwise_matmul/generate_overrides.py
```

生成 JSON schema 2.0.0 block encodings：

- block 32：2048 个 scale
- block 64：1024 个 scale
- block 128：512 个 scale

## NumPy 分析

```bash
python qnn_quantization/11_w4fp16_blockwise_matmul/analyze_blocks.py
```

| Block size | 权重+scale | 相对 FP16 压缩 | mean error | cosine | SQNR |
|---:|---:|---:|---:|---:|---:|
| 32 | 36 KiB | 3.56x | 0.02177 | 0.9977186 | 23.34 dB |
| 64 | 34 KiB | 3.76x | 0.02233 | 0.9976037 | 23.12 dB |
| 128 | 33 KiB | 3.88x | 0.02251 | 0.9975513 | 23.02 dB |

block 越小，每个 scale 覆盖的权重越少，因此误差较低；代价是 scale 数量和
metadata 增加。本测试中 block 32 精度最好，但三个 block size 的差异不大。

## QAIRT 2.47 工具链验证

尝试了三条路径。

### v2 quantization overrides

`qnn-onnx-converter` 在处理合法的 block zero-point 数组时失败：

```text
TypeError: type numpy.ndarray doesn't define __round__ method
```

`run_converter_with_block_fix.py` 只为这个数组检查增加 NumPy 支持，不修改 SDK
安装文件。修复后 converter 可以完成，但随后报告：

```text
Constant folded static tensor rhs.nchw from s4 to f16
```

检查生成 C++ 可确认 RHS 是 `QNN_DATATYPE_FLOAT_16`，因此该模型不是 W4。

### qairt-converter + qairt-quantizer

backend-aware DLC 路径同样把 RHS 转为 FP16。`qairt-dlc-info` 显示：

```text
rhs: Float_16, No encoding info
```

再次运行 `qairt-quantizer` 时，block encoding 已经丢失，无法恢复。

### ONNX INT4 QDQ

`generate_int4_qdq.py` 生成了通过 ONNX checker 的 opset 21 模型：

```text
INT4 packed RHS
DequantizeLinear(axis=2, block_size=32)
FP16 MatMul
```

packed INT4 权重是 32768 bytes。但 QAIRT 2.47 ONNX frontend 无法读取
`TensorProto.INT4`：

```text
no supported data type: 22
Node RhsInt4BlockDequantize: shape
```

## 当前结论

当前 SDK 的 HTP runtime 文档声称支持 W4FP16 block quantization，但本机
QAIRT 2.47 converter 前端无法从手写 ONNX/overrides 可靠生成该模型。

所以目前可以得到可信的 NumPy 精度与存储结果，但不能报告 HTP cycles。把 FP16
fallback 模型当作 W4 profiling 会得到错误结论。

真实部署应优先使用 Qualcomm 已支持的模型来源和量化流程，例如官方支持的
QAT/LLM 导出工具、可保留 block encoding 的 DLC，或升级到修复 INT4 ONNX/v2
block encoding 的 QAIRT 版本。拿到真正的 block-encoded DLC 后，再执行
`qairt-dlc-prepare` 和 `qnn-net-run --profiling_level detailed`。
