# INT8 Per-Axis Quantization

本实验比较权重矩阵的 per-tensor 与 per-axis 对称 INT8 量化。

## 权重形状

```text
RHS[K,N] = [256,64]
```

每个输出列使用不同的动态范围。量化轴选择：

```text
axis = 1
numScaleOffsets = N = 64
```

这意味着每个输出 channel 有独立的 scale：

```text
scale[j] = max(abs(rhs[:,j])) / 127
```

## QNN 表达

```cpp
Qnn_ScaleOffset_t channelQuantParams[64];

Qnn_QuantizeParams_t quantizeParams = {
    QNN_DEFINITION_DEFINED,
    QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET,
    {.axisScaleOffsetEncoding = {
        .axis = 1,
        .numScaleOffsets = 64,
        .scaleOffset = channelQuantParams,
    }},
};
```

每个 `channelQuantParams[j]`：

```cpp
channelQuantParams[j].scale = channelScale[j];
channelQuantParams[j].offset = 0;
```

若使用 UINT8 storage 表达 signed symmetric INT8，则 offset 应为 `-128`。

## 执行

```bash
python3 qnn_quantization/03_int8_per_axis/compare.py
```

输出：

- per-tensor 与 per-axis scale
- 最大和平均绝对误差
- cosine similarity
- SQNR
- 每个 channel 的误差改善倍数
- INT8 数据与 scale metadata 大小

## 为什么通常更准确

per-tensor scale 由动态范围最大的 channel 决定。较小的 channel 只能使用
很少的量化级别。

per-axis 为每列单独选择 scale，使小范围 channel 也可以充分使用 INT8 的
有效范围。代价是额外保存 64 个 scale，并且 kernel 反量化时需要按输出列
选择 scale。

## 下一步

建立 QNN `AXIS_SCALE_OFFSET` Quantize graph，直接导出 UINT8 raw tensor，
并与本目录的 NumPy per-axis reference 做 1 LSB 比较。
