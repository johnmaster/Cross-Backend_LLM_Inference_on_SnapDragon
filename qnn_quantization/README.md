# QNN Quantization

本目录用于学习和实验 QNN 量化。

## QNN 官方量化约定

QNN 文档把量化描述为：用给定 bitwidth，把一组 floating point values 映射到一组
fixed point integers。对 `bw` bit unsigned fixed-point 来说，整数范围是：

```text
qmin = 0
qmax = 2^bw - 1
```

QNN 先从输入数据计算 true range，再得到用于量化的 encoding range：

```text
true_min = min(input)
true_max = max(input)
encoding_min <= true_min
encoding_max >= true_max
```

文档里强调三个要求：

- 覆盖完整输入范围。
- encoding range 至少为 `0.0001`。
- 浮点值 0 必须能被某个 fixed-point integer 精确表示。

因此 `encoding_min` 和 `encoding_max` 不一定等于真实数据的 min/max。QNN 会根据
数据分布调整它们：

- 输入全为正数时，`encoding_min = 0.0`，0 由 fixed-point value 0 表示。
- 输入全为负数时，`encoding_max = 0.0`，0 由最大 fixed-point value 表示。
- 输入同时有负数和正数时，range 可能被轻微平移，使 0 落在某个整数格点上。

encoding 参数为：

```text
scale  = (encoding_max - encoding_min) / (2^bw - 1)
offset = round(encoding_min / scale)
```

用常见非对称量化公式看，同一件事是：

```text
scale = (encoding_max - encoding_min) / (qmax - qmin)
zero_point_real = qmin - encoding_min / scale
zero_point = round(zero_point_real)

q  = clamp(round(x / scale) + zero_point, qmin, qmax)
dq = (q - zero_point) * scale
```

问题就在 `zero_point_real` 通常不是整数，但 fixed-point value 必须是整数，所以
实际只能使用 `round` 后的 `zero_point`。一旦 `zero_point` 被取整，再把
`qmin/qmax` 代回反量化公式，得到的实际可表示边界就是：

```text
aligned_encoding_min = (qmin - zero_point) * scale
aligned_encoding_max = (qmax - zero_point) * scale
```

对 QNN 文档里的 unsigned fixed-point，`qmin = 0`，并且
`offset = -zero_point`，所以同一组式子会写成：

```text
aligned_encoding_min = offset * scale
aligned_encoding_max = (2^bw - 1 + offset) * scale
range_shift          = aligned_encoding_min - encoding_min
```

这个 `range_shift` 不是额外又做了一次神秘修正，而是 `zero_point/offset`
取整以后，反量化网格相对于原始 `encoding_min` 的实际偏移。
因为 QNN 的反量化公式是：

```text
real_value = (quantized + offset) * scale
```

当 `real_value = 0` 时：

```text
0 = (quantized_zero + offset) * scale
quantized_zero = -offset
```

所以只要 `offset` 是整数，浮点 0 就会落在 fixed-point integer `-offset` 上。
对 unsigned 8-bit 来说，这个整数必须落在 `[0, 255]`。

官方文档中的 `[-5.1, 5.1]` 例子可以这样算：

```text
bw = 8
qmax = 255
encoding_min = -5.1
encoding_max =  5.1
scale = (5.1 - (-5.1)) / 255 = 0.04

offset = round(encoding_min / scale)
       = round(-5.1 / 0.04)
       = round(-127.5)
       = -128

quantized_zero = -offset = 128

aligned_encoding_min = offset * scale
                     = -128 * 0.04
                     = -5.12

aligned_encoding_max = (255 + offset) * scale
                     = (255 - 128) * 0.04
                     = 5.08

range_shift = -5.12 - (-5.1) = -0.02
```

所以文档里说的 “shift by `-0.02`” 来自：

```text
range_shift = round(encoding_min / scale) * scale - encoding_min
```

这个平移不会改变 step size，只是把整个 fixed-point 网格挪动到让 0 正好落在
一个整数 index 上。量化时仍然会对结果做 clamp，因此超出最终
`aligned_encoding_min/max` 的值会被夹到 `[0, 2^bw - 1]`。

这里的 `offset` 是 QNN 的 scale-offset 约定，不是很多框架里说的
`zero_point`。二者关系是：

```text
zero_point = -offset
offset = -zero_point
```

因此 QNN quantize/dequantize 可以写成：

```text
quantized = clamp(round(float_value / scale - offset), qmin, qmax)
real_value = (quantized + offset) * scale
```

也可以写成官方文档中的 `encoding_min/max` 形式：

```text
quantized = round((2^bw - 1) * (float_value - encoding_min)
                  / (encoding_max - encoding_min))
quantized = clamp(quantized, 0, 2^bw - 1)
```

这两种写法等价，因为 `scale = (encoding_max - encoding_min)/(2^bw - 1)`，
且 `offset ~= encoding_min / scale`。这里的 `real_value` 指量化整数按
`scale + offset` 反量化后代表的浮点值，不一定等于量化前的原始 FP32 值；round
和 clamp 会带来误差。

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
