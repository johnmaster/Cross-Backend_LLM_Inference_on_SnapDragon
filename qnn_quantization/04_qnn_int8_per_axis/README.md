# QNN INT8 Per-Axis Quantization

本实验探索 QNN HTP 对以下 encoding 的支持边界：

```text
per-tensor SCALE_OFFSET
per-axis AXIS_SCALE_OFFSET
```

## 输入

```text
weights shape: [256,64]
axis:          1
channels:      64
```

64 个输出列具有从 `0.02` 到 `1.0` 的不同动态范围。

## QNN 参数

per-tensor：

```cpp
QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
scale  = 1.0 / 127.0
offset = -128
```

per-axis：

```cpp
QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
axis            = 1
numScaleOffsets = 64
scaleOffset     = per_channel_scale_array
```

每个 channel：

```text
scale[j]  = channel_range[j] / 127
offset[j] = -128
```

使用 `QNN_DATATYPE_UFIXED_POINT_8` 存储数学上的 signed symmetric INT8：

```text
stored_uint8 = signed_int8 + 128
real_value   = (stored_uint8 - 128) * scale
```

## 生成数据

```bash
python3 qnn_quantization/04_qnn_int8_per_axis/generate_data.py
```

## 编译

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qnn_quantization/04_qnn_int8_per_axis/model/qnn_int8_per_axis_model.cpp \
  -t aarch64-android \
  -l qnn_int8_per_axis_model \
  -o qnn_quantization/04_qnn_int8_per_axis/model_libs
```

## 推送

```bash
adb shell 'mkdir -p \
  /data/local/tmp/qnn/qnn_int8_per_axis/lib \
  /data/local/tmp/qnn/qnn_int8_per_axis/input \
  /data/local/tmp/qnn/qnn_int8_per_axis/output'

adb push \
  qnn_quantization/04_qnn_int8_per_axis/model_libs/aarch64-android/libqnn_int8_per_axis_model.so \
  /data/local/tmp/qnn/qnn_int8_per_axis/lib/

adb push \
  qnn_quantization/04_qnn_int8_per_axis/input/. \
  /data/local/tmp/qnn/qnn_int8_per_axis/input/
```

## 执行实验

```bash
adb shell 'cd /data/local/tmp/qnn && \
rm -rf qnn_int8_per_axis/output/* && \
export LD_LIBRARY_PATH="$PWD/qnn_int8_per_axis/lib:$PWD/lib:$LD_LIBRARY_PATH" && \
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp" && \
./bin/qnn-sample-app \
  --backend lib/libQnnHtp.so \
  --model qnn_int8_per_axis/lib/libqnn_int8_per_axis_model.so \
  --input_list qnn_int8_per_axis/input/input_list.txt \
  --output_dir qnn_int8_per_axis/output \
  --input_data_type float \
  --output_data_type float_only \
  --log_level info'
```

## 设备结果

```text
Op Dequantize does not support per-channel quant tensor
Failed to construct common node
Graph Prepare failure
```

这不是 scale 数组生命周期或 axis 维度错误。model 能编译，Quantize 节点
也能通过 validation；失败发生在 HTP 构建 per-axis Dequantize 节点时。

之前尝试把 per-axis UINT8 tensor 直接声明为 APP_READ：

```text
graph 执行成功
输出只包含 0/1
不能按预期解释为 per-channel raw UINT8
```

SampleApp 的 graph-output I/O 主要支持 per-tensor `scaleOffsetEncoding`，
不能用这条路径可靠验证 per-axis tensor。

## 当前状态

```text
数据生成：通过
model 编译：通过
per-tensor Q/DQ：支持
per-axis Quantize validation：支持
per-axis Dequantize：HTP 2.47 不支持
```

## 结论

```text
AXIS_SCALE_OFFSET 不能通过独立 Quantize -> Dequantize 小图学习完整语义。
```

在 HTP 的实际模型中，per-axis encoding 通常附着在静态 Conv/MatMul
权重上，由计算算子直接消费，而不是经过独立 Dequantize。

因此下一步应建立：

```text
FP32 activation
UINT8 per-axis static weight
QNN MatMul
FP32 output
```

比较：

```text
per-tensor weight MatMul
per-axis weight MatMul
FP32 reference
```

## 学习重点

1. axis 是 tensor dimension index，不是随意的 channel 编号。
2. `numScaleOffsets` 必须等于所选 axis 的维度。
3. `scaleOffset` 数组必须在 graph finalize 期间保持有效。
4. per-axis 增加少量 metadata，换取小动态范围 channel 的更高精度。
5. backend 支持某种 tensor encoding，不代表每个算子都支持该 encoding。
6. 应通过目标计算算子的 datatype/quantization constraints 验证支持范围。
