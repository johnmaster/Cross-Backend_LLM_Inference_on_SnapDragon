# QNN Quantize and Dequantize

本实验把第一步的 NumPy per-tensor 量化放入真正的 QNN HTP 图。

## 图结构

同一个 FP32 input 分成两条支路：

```text
                -> symmetric Quantize -> raw UINT8 output -> Dequantize -> FP32
FP32 input ----+
                -> asymmetric Quantize -> raw UINT8 output -> Dequantize -> FP32
```

数学上的 signed symmetric INT8 使用 UINT8 storage 表达：

```text
stored_byte = signed_int8 + 128
scale       = 0.013284672902325007
QNN offset  = -128
```

非对称 UINT8：

```text
scale       = 0.007043378727108825
zero_point  = 15
QNN offset  = -15
```

QNN 的反量化公式是：

```text
real_value = (quantized_value + offset) * scale
```

## 生成输入

```bash
python3 qnn_quantization/02_qnn_quantize_dequantize/generate_data.py
```

## 编译

```bash
rm -rf qnn_quantization/02_qnn_quantize_dequantize/model_libs

PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qnn_quantization/02_qnn_quantize_dequantize/model/qnn_quantize_dequantize_model.cpp \
  -t aarch64-android \
  -l qnn_quantize_dequantize_model \
  -o qnn_quantization/02_qnn_quantize_dequantize/model_libs
```

## 执行

量化 tensor 是 graph output，因此必须使用 `native_only` 才能保存 raw bytes：

```bash
adb shell 'cd /data/local/tmp/qnn && \
rm -rf qnn_quantize_dequantize/output/* && \
export LD_LIBRARY_PATH="$PWD/qnn_quantize_dequantize/lib:$PWD/lib:$LD_LIBRARY_PATH" && \
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp" && \
./bin/qnn-sample-app \
  --backend lib/libQnnHtp.so \
  --model qnn_quantize_dequantize/lib/libqnn_quantize_dequantize_model.so \
  --input_list qnn_quantize_dequantize/input/input_list.txt \
  --output_dir qnn_quantize_dequantize/output \
  --input_data_type float \
  --output_data_type native_only \
  --log_level info'
```

## 比较

拉回输出后执行：

```bash
python3 qnn_quantization/02_qnn_quantize_dequantize/compare_outputs.py
```

当前结果：

```text
Symmetric:
  different:          673 / 32768
  max_lsb_difference: 1

Asymmetric:
  different:          1343 / 32768
  max_lsb_difference: 1
```

所有 QNN raw quantized values 都在 NumPy reference 的 1 LSB 范围内。

## 学到的 QNN 行为

1. `QNN_DATATYPE_UFIXED_POINT_8` 描述物理存储类型。
2. `QNN_QUANTIZATION_ENCODING_SCALE_OFFSET` 描述数值解释方式。
3. QNN offset 等于常见公式中的负 zero point。
4. HTP Quantize 的 reciprocal 和 rounding 路径不完全等于 NumPy
   `float32 division + rint`，边界值可能相差 1 LSB。
5. 当前 SampleApp 不能把 `SFIXED_POINT_8` 作为 native graph output 导出，
   因此实验使用 UINT8 storage 表达 signed symmetric INT8。
6. 只观察 Quantize->Dequantize FP32 输出可能具有误导性。Graph prepare
   可以折叠或改变 Q/DQ 数据通路；学习量化时应同时导出 raw quantized tensor。

## 下一步

学习 per-axis quantization，让每个 channel 使用独立 scale：

```text
QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
```
