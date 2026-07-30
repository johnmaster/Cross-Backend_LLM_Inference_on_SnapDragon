# Quantized MatMul 对比

本实验使用同一组 `128x256 * 256x256` 数据比较：

1. FP32 输入的 QNN 内置 MatMul（HTP 内部使用 FP16）
2. INT8 activation + INT8 per-tensor weight MatMul
3. INT8 activation + INT8 per-axis static weight MatMul

## 量化定义

激活和 per-tensor 权重使用 `QNN_DATATYPE_UFIXED_POINT_8`：

```text
q_u8 = clamp(round(real / scale), -128, 127) + 128
real = (q_u8 - 128) * scale
offset = -128
```

per-axis 权重沿 RHS 最后一维（输出列，`axis=3`）量化。每个输出列有独立
scale，静态数据保存为有符号 INT8，offset 为 0：

```text
q_i8[j] = clamp(round(weight[j] / scale[j]), -128, 127)
weight[j] = q_i8[j] * scale[j]
```

测试数据刻意让不同输出列具有 `0.02` 到 `0.5` 的不同范围，以体现
per-axis 相对于 per-tensor 的精度优势。

## 文件

- `generate_data.py`：生成输入、量化数据、NumPy reference 和量化参数。
- `model/fp32_matmul_model.cpp`：FP32 I/O、内部 FP16 MatMul。
- `model/int8_per_tensor_matmul_model.cpp`：输入和权重均为 per-tensor INT8。
- `model/int8_per_axis_weight_matmul_model.cpp`：静态 per-axis INT8 权重模型。
- `compare_outputs.py`：计算最大误差、平均误差、余弦相似度、SQNR 和 LSB 差异。
- `model_bin/`：per-axis 静态权重及其 model binary。
- `reference/`：NumPy FP32 和量化参考结果。

## 生成数据

```bash
python3 qnn_quantization/05_quantized_matmul/generate_data.py

tar -cf qnn_quantization/05_quantized_matmul/model_bin/per_axis_weight.bin \
  -C qnn_quantization/05_quantized_matmul/model_bin \
  rhs_per_axis_weight.raw
```

### `rhs_per_axis_weight` 的来源

`int8_per_axis_weight_matmul_model.cpp` 中的：

```cpp
rhs.v1.clientBuf = {
    .data = BINVARSTART(rhs_per_axis_weight),
    .dataSize = BINLEN(rhs_per_axis_weight),
};
```

这里的 `rhs_per_axis_weight` 不是实际文件名，而是
`qnn-model-lib-generator` 根据静态权重文件生成的 C/C++ 链接符号。对应的实际
文件是：

```text
model_bin/rhs_per_axis_weight.raw
```

`generate_data.py` 通过以下代码生成该文件：

```python
rhs_axis_i8.tofile(ROOT / "model_bin" / "rhs_per_axis_weight.raw")
```

随后，前面的 `tar` 命令将它打包进：

```text
model_bin/per_axis_weight.bin
```

完整关系如下：

```text
rhs_per_axis_weight.raw
        │
        │ tar -cf per_axis_weight.bin
        ▼
per_axis_weight.bin
        │
        │ qnn-model-lib-generator -b
        ▼
BINVARSTART(rhs_per_axis_weight)
BINLEN(rhs_per_axis_weight)
```

编译 model library 时，`-b model_bin/per_axis_weight.bin` 会将其中的
`rhs_per_axis_weight.raw` 嵌入动态库，并生成上面的符号。运行时
`QNN_TENSOR_TYPE_STATIC` 类型的 RHS 通过 `clientBuf` 直接引用这段嵌入数据，
不需要在 `input_list` 中提供 RHS，也不需要把 raw 或 bin 文件单独推送到设备。

## 编译

FP32：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qnn_quantization/05_quantized_matmul/model/fp32_matmul_model.cpp \
  -t aarch64-android -l fp32_matmul_model \
  -o qnn_quantization/05_quantized_matmul/model_libs/fp32
```

INT8 per-tensor：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qnn_quantization/05_quantized_matmul/model/int8_per_tensor_matmul_model.cpp \
  -t aarch64-android -l int8_per_tensor_matmul_model \
  -o qnn_quantization/05_quantized_matmul/model_libs/int8_per_tensor
```

INT8 per-axis weight：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qnn_quantization/05_quantized_matmul/model/int8_per_axis_weight_matmul_model.cpp \
  -b qnn_quantization/05_quantized_matmul/model_bin/per_axis_weight.bin \
  -t aarch64-android -l int8_per_axis_weight_matmul_model \
  -o qnn_quantization/05_quantized_matmul/model_libs/int8_per_axis
```

## 部署到设备

以下命令假设 QNN runtime、`qnn-sample-app` 和 HTP DSP 库已经部署到
`/data/local/tmp/qnn`。

```bash
adb shell 'mkdir -p \
  /data/local/tmp/qnn/qnn_quantized_matmul/lib \
  /data/local/tmp/qnn/qnn_quantized_matmul/input \
  /data/local/tmp/qnn/qnn_quantized_matmul/output/fp32 \
  /data/local/tmp/qnn/qnn_quantized_matmul/output/int8_per_tensor \
  /data/local/tmp/qnn/qnn_quantized_matmul/output/int8_per_axis'

adb push \
  qnn_quantization/05_quantized_matmul/model_libs/fp32/aarch64-android/libfp32_matmul_model.so \
  /data/local/tmp/qnn/qnn_quantized_matmul/lib/

adb push \
  qnn_quantization/05_quantized_matmul/model_libs/int8_per_tensor/aarch64-android/libint8_per_tensor_matmul_model.so \
  /data/local/tmp/qnn/qnn_quantized_matmul/lib/

adb push \
  qnn_quantization/05_quantized_matmul/model_libs/int8_per_axis/aarch64-android/libint8_per_axis_weight_matmul_model.so \
  /data/local/tmp/qnn/qnn_quantized_matmul/lib/

adb push \
  qnn_quantization/05_quantized_matmul/input/. \
  /data/local/tmp/qnn/qnn_quantized_matmul/input/
```

`per_axis_weight.bin` 已由 `qnn-model-lib-generator` 嵌入
`libint8_per_axis_weight_matmul_model.so`，运行时不需要单独推送。

## 执行 FP32 MatMul

```bash
adb shell 'cd /data/local/tmp/qnn && \
rm -rf qnn_quantized_matmul/output/fp32/* && \
export LD_LIBRARY_PATH="$PWD/qnn_quantized_matmul/lib:$PWD/lib:$LD_LIBRARY_PATH" && \
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp" && \
./bin/qnn-sample-app \
  --backend lib/libQnnHtp.so \
  --model qnn_quantized_matmul/lib/libfp32_matmul_model.so \
  --input_list qnn_quantized_matmul/input/fp32_input_list.txt \
  --output_dir qnn_quantized_matmul/output/fp32 \
  --input_data_type float \
  --output_data_type float_only \
  --log_level info'
```

## 执行 INT8 per-tensor MatMul

输入文件已经是模型原生的 UINT8 数据，因此必须使用
`--input_data_type native`。输出保留 UINT8 原始值，供比较脚本自行反量化。

```bash
adb shell 'cd /data/local/tmp/qnn && \
rm -rf qnn_quantized_matmul/output/int8_per_tensor/* && \
export LD_LIBRARY_PATH="$PWD/qnn_quantized_matmul/lib:$PWD/lib:$LD_LIBRARY_PATH" && \
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp" && \
./bin/qnn-sample-app \
  --backend lib/libQnnHtp.so \
  --model qnn_quantized_matmul/lib/libint8_per_tensor_matmul_model.so \
  --input_list qnn_quantized_matmul/input/per_tensor_input_list.txt \
  --output_dir qnn_quantized_matmul/output/int8_per_tensor \
  --input_data_type native \
  --output_data_type native_only \
  --log_level info'
```

## 执行 INT8 per-axis static-weight MatMul

这个命令用于复现后文记录的 HTP Graph Prepare 崩溃：

```bash
adb shell 'cd /data/local/tmp/qnn && \
rm -rf qnn_quantized_matmul/output/int8_per_axis/* && \
export LD_LIBRARY_PATH="$PWD/qnn_quantized_matmul/lib:$PWD/lib:$LD_LIBRARY_PATH" && \
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp" && \
./bin/qnn-sample-app \
  --backend lib/libQnnHtp.so \
  --model qnn_quantized_matmul/lib/libint8_per_axis_weight_matmul_model.so \
  --input_list qnn_quantized_matmul/input/per_axis_input_list.txt \
  --output_dir qnn_quantized_matmul/output/int8_per_axis \
  --input_data_type native \
  --output_data_type native_only \
  --log_level info'
```

## 拉取输出并比较

```bash
mkdir -p \
  qnn_quantization/05_quantized_matmul/device_output/fp32 \
  qnn_quantization/05_quantized_matmul/device_output/int8_per_tensor

adb pull \
  /data/local/tmp/qnn/qnn_quantized_matmul/output/fp32/Result_0/output.raw \
  qnn_quantization/05_quantized_matmul/device_output/fp32/output.raw

adb pull \
  /data/local/tmp/qnn/qnn_quantized_matmul/output/int8_per_tensor/Result_0/output_native.raw \
  qnn_quantization/05_quantized_matmul/device_output/int8_per_tensor/output_native.raw

python3 qnn_quantization/05_quantized_matmul/compare_outputs.py
```

## 当前结果

FP32 和 INT8 per-tensor 已在 HTP 上成功执行。相对 NumPy FP32 reference：

| 实验 | max abs error | mean abs error | cosine similarity | SQNR |
|---|---:|---:|---:|---:|
| QNN FP16 MatMul | 0.0018971 | 0.0001082 | 0.99999976 | 69.28 dB |
| QNN INT8 per-tensor | 0.0394292 | 0.0091150 | 0.99978596 | 33.68 dB |

QNN per-tensor 输出与 NumPy 量化 reference 最多相差 1 LSB，说明量化参数、
raw 数据解释和 MatMul 图是吻合的。

## Per-axis 遇到的问题

在 QAIRT 2.47 和当前 HTP 设备上，手写 model library 的 per-axis static RHS
会在图 prepare 阶段进入 `libQnnHtpPrepare.so` 后发生空指针崩溃。

已经验证过两种表达：

- `AXIS_SCALE_OFFSET` + `SFIXED_POINT_8` + offset 0
- `BW_AXIS_SCALE_OFFSET` + bitwidth 8 + `SFIXED_POINT_8` + symmetric offset

两者均在 prepare 阶段崩溃，静态 binary 的符号、长度和数据尺寸已核对正确。
因此当前结果不能解释为“HTP 不支持 per-axis MatMul”，只能说明这个版本中，
直接手写 model-library 元数据的路径不可用或触发了 HTP prepare 缺陷。

后续应使用 ONNX QDQ/量化配置，经 QAIRT converter 生成模型，再验证
`INT8 activation + per-axis weight MatMul`。

## 比较脚本行为

脚本会自动检测 `device_output/int8_per_axis/output_native.raw`；不存在时会明确
报告 per-axis HTP 输出不可用，不会误用旧输出。
