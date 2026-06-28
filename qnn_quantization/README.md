# QNN Quantization

本目录用于学习和实验 QNN 量化。

## 学习顺序

1. INT8 symmetric per-tensor
2. INT8 asymmetric per-tensor
3. INT8 per-axis
4. INT16 / Q13 fixed-point
5. INT4 blockwise weight quantization
6. Quantized MatMul and Linear

当前 MatMul 对比实验见
[`05_quantized_matmul`](05_quantized_matmul/README.md)，包含 FP32、INT8
per-tensor，以及 INT8 activation + per-axis weight 三条路径。

converter 生成并在 HTP 上成功运行的 per-axis MatMul 见
[`06_onnx_qdq_matmul`](06_onnx_qdq_matmul/README.md)。

weight、activation 和 output 三阶段误差拆分见
[`07_quantization_error_decomposition`](07_quantization_error_decomposition/README.md)。

W8A8 与 W8A16 mixed-precision 对比见
[`08_w8a16_matmul`](08_w8a16_matmul/README.md)。

校准集范围、离群值和量化精度的关系见
[`09_calibration_sensitivity`](09_calibration_sensitivity/README.md)。

min-max 与 99.9/99.99 percentile 校准对比见
[`10_percentile_calibration`](10_percentile_calibration/README.md)。

W4FP16 block-size 精度、存储分析和 QAIRT 2.47 converter 支持边界见
[`11_w4fp16_blockwise_matmul`](11_w4fp16_blockwise_matmul/README.md)。

## 实验要求

每个量化实验应记录：

- 数据类型和量化编码
- scale、offset 和量化轴
- 量化与反量化公式
- calibration 方法
- FP32 reference
- 最大误差和平均误差
- cosine similarity 和 SQNR
- tensor 大小
- HTP accelerator execute time
- kernel cycles 和转换开销

## 计划目录

```text
qnn_quantization/
  int8_per_tensor/
  int8_per_axis/
  int16_q13/
  int4_blockwise/
  quantized_matmul/
```
