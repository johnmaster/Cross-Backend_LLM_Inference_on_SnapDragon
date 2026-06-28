# Quantization Error Decomposition

本实验不更换模型和数据，而是拆分
`06_onnx_qdq_matmul` 中三种误差来源：

1. RHS per-axis INT8 weight
2. UINT8 activation
3. UINT8 output requantization

## 量化公式

QNN scale-offset encoding：

```text
real = (quantized + offset) * scale
quantized = round(real / scale - offset)
```

本实验直接使用 converter 生成模型中的参数：

```text
LHS:    scale=0.0039213779382407665, offset=-128
RHS:    256 个 per-axis scale, offset=0, axis=3
OUTPUT: scale=0.03154205158352852,   offset=-124
```

## 执行

```bash
python qnn_quantization/07_quantization_error_decomposition/analyze.py
```

## 结果

| 阶段 | max error | mean error | cosine | SQNR |
|---|---:|---:|---:|---:|
| 仅 per-axis weight | 0.01671 | 0.00125 | 0.9999924 | 48.16 dB |
| activation + weight | 0.02507 | 0.00177 | 0.9999848 | 45.15 dB |
| activation + weight + output | 0.03450 | 0.00818 | 0.9998370 | 34.87 dB |
| QNN HTP output | 0.03450 | 0.00820 | 0.9998358 | 34.83 dB |

数学模型和 HTP 输出最多相差一个 output LSB。这种差异可能来自 HTP 的整数
累加、requantization 舍入方式或融合执行细节，属于预期范围。

## 结论

当前实验中：

- per-axis weight 本身的精度很好。
- activation INT8 增加了一部分误差，但仍保持约 45 dB SQNR。
- output UINT8 把 SQNR 从约 45 dB 降至约 35 dB，是主要误差来源。

因此，看到 per-axis 相对 per-tensor 的提升不大，并不代表 per-axis 没有效果；
其收益被 output requantization 的误差遮住了。

真实网络通常不会在每个算子后都转回 FP32。相邻量化算子共享 INT8 tensor，
最终应衡量整网任务精度，而不是要求每个中间 tensor 都逼近 FP32。
