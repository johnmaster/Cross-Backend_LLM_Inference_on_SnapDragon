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
scale = (encoding_max - encoding_min) / 255
zero_point = clip(round(qmin - encoding_min / scale), qmin, qmax)
q = clip(round(x / scale) + zero_point, qmin, qmax)
x_dequant = (q - zero_point) * scale
```

QNN 官方文档使用 `encoding-min/max` 和 `offset` 表达同一件事：

```text
scale  = (encoding_max - encoding_min) / (2^bw - 1)
offset = round(encoding_min / scale)

quantized_value = clamp(round(x / scale - offset), qmin, qmax)
real_value = (quantized_value + offset) * scale
```

因此：

```text
QNN offset = -zero_point
```

注意 `encoding_min/max` 不一定等于输入数据的真实 min/max。QNN 会保证完整覆盖
输入范围、range 至少为 `0.0001`，并调整 encoding range 让浮点 0 可以被某个
fixed-point integer 精确表示。

这个调整可以从普通非对称公式直接看出来。理论上：

```text
zero_point_real = qmin - encoding_min / scale
```

但 `zero_point` 必须是整数，所以实际会使用：

```text
zero_point = round(zero_point_real)
```

把这个取整后的 `zero_point` 代回 `dq = (q - zero_point) * scale`，实际可表示
边界就变成：

```text
aligned_encoding_min = (qmin - zero_point) * scale
aligned_encoding_max = (qmax - zero_point) * scale
range_shift = aligned_encoding_min - encoding_min
```

QNN 只是把 `zero_point` 换成相反符号的 `offset`：

```text
offset = -zero_point
```

此时浮点 0 对应的 fixed-point integer 是：

```text
quantized_zero = -offset
```

例如 `encoding_min=-5.1`、`encoding_max=5.1`、`qmax=255` 时，
`scale=10.2/255=0.04`，`offset=round(-5.1/0.04)=-128`，
所以 `aligned_encoding_min=-5.12`，`aligned_encoding_max=5.08`，
整个 range 被平移了 `-0.02`，0 正好由整数 128 表示。

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
