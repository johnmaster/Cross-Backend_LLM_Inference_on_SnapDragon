# Percentile Calibration

本实验继续使用 `09_calibration_sensitivity` 中含单个 `8.0` 离群值的
calibration 数据，比较：

- min-max
- percentile 99.9
- percentile 99.99

RHS per-axis weight、正常测试输入和 HTP 环境保持不变。

## 转换

```bash
qnn-onnx-converter \
  -i qnn_quantization/06_onnx_qdq_matmul/model/fp32_matmul.onnx \
  -o qnn_quantization/10_percentile_calibration/generated/percentile.cpp \
  --input_list \
    qnn_quantization/09_calibration_sensitivity/calibration/outlier/converter_input_list.txt \
  --quantization_overrides \
    qnn_quantization/06_onnx_qdq_matmul/model/quantization_overrides.json \
  --act_bitwidth 8 \
  --weights_bitwidth 8 \
  --param_quantizer_schema symmetric \
  --act_quantizer_calibration percentile \
  --percentile_calibration_value 99.9
```

将最后一项改为 `99.99` 可生成更保守的 percentile 模型。

## 生成的 encoding

| Calibration | LHS scale | LHS offset | output scale |
|---|---:|---:|---:|
| outlier min-max | 0.03333325 | -15 | 0.03005265 |
| percentile 99.9 | 0.00390624 | -124 | 0.02165904 |
| percentile 99.99 | 0.00390624 | -124 | 0.02817436 |

percentile 成功避免单个离群值把 LHS scale 放大，但 99.9 和 99.99 对 output
范围的保留程度不同。

## 比较

```bash
python qnn_quantization/10_percentile_calibration/compare_output.py
```

| Calibration | input sat | levels | output sat | mean error | SQNR |
|---|---:|---:|---:|---:|---:|
| representative min-max | 0.000% | 256 | 0.000% | 0.00820 | 34.83 dB |
| outlier min-max | 0.000% | 31 | 0.003% | 0.01376 | 28.62 dB |
| outlier percentile 99.9 | 1.422% | 253 | 0.189% | 0.00673 | 27.09 dB |
| outlier percentile 99.99 | 1.422% | 253 | 0.018% | 0.00753 | 34.61 dB |

## 如何理解

min-max 保留所有离群值，因此没有输入 clipping，但正常数据只使用 31 个量化
级，rounding error 较大。

99.9 percentile 的平均误差最低，说明大多数元素更准确；但少量 output
clipping 很严重，平方误差被尾部主导，因此 SQNR 反而最低。

99.99 percentile 保留更宽的 output 范围，SQNR 接近代表性 min-max，同时平均
误差仍优于代表性 min-max，是这组数据上更均衡的选择。

不能只凭 calibration 方法名称判断好坏。分类任务应查看任务精度，生成模型还应
关注异常 token、长尾通道和最终输出质量；MSE、SQNR、平均误差也可能给出不同
排序。percentile 数值必须在独立验证集上选择，不能用测试集调参。
