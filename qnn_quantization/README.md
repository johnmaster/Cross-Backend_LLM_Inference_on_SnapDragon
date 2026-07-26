# QNN Quantization

本目录保存 QNN 量化实验、设备结果和复现命令。量化定义与参数以各实验 README
中的实际实现为准。

## 实验索引

| 目录 | 内容 |
|---|---|
| [`01_int8_per_tensor`](01_int8_per_tensor/README.md) | INT8 per-tensor NumPy reference |
| [`02_qnn_quantize_dequantize`](02_qnn_quantize_dequantize/README.md) | QNN HTP Quantize/Dequantize |
| [`03_int8_per_axis`](03_int8_per_axis/README.md) | INT8 per-axis NumPy reference |
| [`04_qnn_int8_per_axis`](04_qnn_int8_per_axis/README.md) | QNN per-axis encoding 支持边界 |
| [`05_quantized_matmul`](05_quantized_matmul/README.md) | FP32、per-tensor 与 per-axis MatMul |
| [`06_onnx_qdq_matmul`](06_onnx_qdq_matmul/README.md) | ONNX QDQ converter 到 HTP |
| [`07_quantization_error_decomposition`](07_quantization_error_decomposition/README.md) | weight、activation 与 output 误差拆分 |
| [`08_w8a16_matmul`](08_w8a16_matmul/README.md) | W8A8 与 W8A16 |
| [`09_calibration_sensitivity`](09_calibration_sensitivity/README.md) | 校准集和离群值敏感性 |
| [`10_percentile_calibration`](10_percentile_calibration/README.md) | min-max 与 percentile 校准 |
| [`11_w4fp16_blockwise_matmul`](11_w4fp16_blockwise_matmul/README.md) | W4FP16 blockwise 与 QAIRT 2.47 支持边界 |
