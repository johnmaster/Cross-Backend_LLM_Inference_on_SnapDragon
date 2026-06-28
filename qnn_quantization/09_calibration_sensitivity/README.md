# Calibration Sensitivity

本实验研究 calibration 数据如何影响 W8A8 MatMul 精度。

保持不变：

- FP32 ONNX 模型
- RHS per-axis INT8 overrides
- 正常测试输入
- HTP 设备与执行参数

只改变 converter 的 calibration 输入：

| Calibration | 输入范围 |
|---|---:|
| representative | 约 `[-0.5, 0.5]` |
| narrow | 约 `[-0.125, 0.125]` |
| outlier | 正常分布，但一个元素为 `8.0` |

## 生成数据

```bash
python qnn_quantization/09_calibration_sensitivity/generate_calibration_data.py
```

## 转换方法

三组均使用与 `06_onnx_qdq_matmul` 相同的转换参数，只替换
`--input_list`：

```bash
qnn-onnx-converter \
  -i qnn_quantization/06_onnx_qdq_matmul/model/fp32_matmul.onnx \
  -o qnn_quantization/09_calibration_sensitivity/generated/narrow.cpp \
  --input_list \
    qnn_quantization/09_calibration_sensitivity/calibration/narrow/converter_input_list.txt \
  --quantization_overrides \
    qnn_quantization/06_onnx_qdq_matmul/model/quantization_overrides.json \
  --act_bitwidth 8 \
  --weights_bitwidth 8 \
  --param_quantizer_schema symmetric
```

将 `narrow` 替换为 `outlier` 可生成离群值模型。代表性模型复用实验 `06`。

converter 生成的 LHS encoding：

| Calibration | scale | offset |
|---|---:|---:|
| representative | 0.00392138 | -128 |
| narrow | 0.00098034 | -128 |
| outlier | 0.03333325 | -15 |

## 比较

```bash
python qnn_quantization/09_calibration_sensitivity/compare_output.py
```

| Calibration | 测试输入越界比例 | 使用的量化级数 | mean error | SQNR |
|---|---:|---:|---:|---:|
| representative | 0.00% | 256 | 0.00820 | 34.83 dB |
| narrow | 75.17% | 256 | 0.20788 | 3.69 dB |
| outlier | 0.00% | 31 | 0.01376 | 28.62 dB |

## 结论

范围过窄时，scale 很小、分辨率看似很高，但真实输入大量超出范围并被夹到
`qmin/qmax`。这种 clipping 会产生灾难性误差。

范围被离群值拉大时，不发生 clipping，但 scale 过大，正常数据只使用少数量化
级，产生明显 rounding error。

因此 calibration 的目标不是简单覆盖越大的范围越好，而是让统计数据代表真实
部署分布，在 clipping error 和 quantization resolution 之间取得平衡。

真实模型应使用多条有代表性的样本，并比较 min-max、percentile、entropy 等
校准策略。测试集不能作为校准集，否则会产生数据泄漏并高估部署精度。
