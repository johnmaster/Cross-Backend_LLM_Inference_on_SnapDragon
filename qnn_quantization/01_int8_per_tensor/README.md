# INT8 Per-Tensor Quantization

第一个量化实验比较：

```text
FP32 reference
INT8 symmetric per-tensor
UINT8 asymmetric per-tensor
```

## 目的

- 理解 scale、zero point 和 QNN offset
- 观察舍入与饱和
- 比较对称和非对称量化误差
- 建立后续 QNN 实验使用的 NumPy reference

## 对称 INT8

```text
qmin = -128
qmax = 127
scale = max(abs(x)) / 127
zero_point = 0
q = clip(round(x / scale), qmin, qmax)
x_dequant = q * scale
```

## 非对称 UINT8

```text
qmin = 0
qmax = 255
scale = (max(x) - min(x)) / 255
zero_point = clip(round(qmin - min(x) / scale), qmin, qmax)
q = clip(round(x / scale) + zero_point, qmin, qmax)
x_dequant = (q - zero_point) * scale
```

QNN 使用：

```text
real_value = (quantized_value + offset) * scale
```

因此：

```text
QNN offset = -zero_point
```

## 执行

```bash
python3 qnn_quantization/01_int8_per_tensor/compare.py
```

脚本使用固定随机种子，并分别测试：

- 以零为中心的分布
- 明显偏离零的非对称分布

输出指标：

- scale
- zero point / QNN offset
- 最大绝对误差
- 平均绝对误差
- cosine similarity
- SQNR
- 饱和元素数量
- 数据存储大小

## 下一步

建立 QNN 图：

```text
FP32 -> Quantize -> INT8/UINT8 -> Dequantize -> FP32
```

然后将 QNN 输出与本目录的 NumPy reference 比较。
