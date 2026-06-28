# ONNX Converter Per-axis MatMul

本实验验证：

```text
FP32 ONNX
  -> calibration + quantization overrides
  -> QNN INT8 activation + per-axis INT8 weight MatMul
  -> HTP
```

矩阵尺寸与 `05_quantized_matmul` 相同：

```text
LHS: [1, 1, 128, 256]
RHS: [1, 1, 256, 256]
OUT: [1, 1, 128, 256]
```

## 为什么新增这个实验

手写 model library 的 per-axis static RHS 会在 QAIRT 2.47 的
`libQnnHtpPrepare.so` 中崩溃。converter 生成的模型可以正常 finalize 和
execute，证明 HTP 支持该量化组合，问题在手写模型没有完整复现 converter
生成的张量、layout 和 graph metadata。

## 环境

```bash
conda activate qairt-2.47
source /home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/envsetup.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

QAIRT 2.47 的 Python/C++ binding 与 NumPy 2.x、protobuf 7.x 不兼容。本实验
使用：

```bash
python -m pip install \
  "numpy==1.26.4" \
  "protobuf>=4.25,<5" \
  "onnx>=1.16,<1.18" \
  onnxruntime PyYAML onnxsim
```

## 生成模型和量化配置

```bash
python qnn_quantization/06_onnx_qdq_matmul/generate_qdq_model.py
```

脚本生成两种 ONNX：

- `int8_per_axis_qdq_matmul.onnx`：用于学习 QDQ 表达。
- `fp32_matmul.onnx`：用于 QAIRT calibration + overrides 的实际部署流程。

`quantization_overrides.json` 为 RHS 的 256 个输出列分别提供 symmetric INT8
范围。仅提供 QDQ 模型时，QAIRT 2.47 会把静态 RHS Dequantize 折叠成 FP32
常量，因此不能据此声称执行了 per-axis INT8 MatMul。

## 转换

```bash
qnn-onnx-converter \
  -i qnn_quantization/06_onnx_qdq_matmul/model/fp32_matmul.onnx \
  -o qnn_quantization/06_onnx_qdq_matmul/generated/int8_per_axis_calibrated.cpp \
  --input_list qnn_quantization/06_onnx_qdq_matmul/input/converter_input_list.txt \
  --quantization_overrides \
    qnn_quantization/06_onnx_qdq_matmul/model/quantization_overrides.json \
  --act_bitwidth 8 \
  --weights_bitwidth 8 \
  --param_quantizer_schema symmetric
```

检查生成代码，应该看到：

```text
lhs: QNN_DATATYPE_UFIXED_POINT_8 + SCALE_OFFSET
rhs: QNN_DATATYPE_SFIXED_POINT_8 + AXIS_SCALE_OFFSET, axis=3, count=256
output: QNN_DATATYPE_UFIXED_POINT_8 + SCALE_OFFSET
```

converter 日志中的 `Total MACs: 256` 统计不准确；生成代码中的输入、输出和
RHS 维度均已核对为目标尺寸，设备执行结果也与完整 NumPy MatMul 对齐。

## 编译

library name 使用短名，避免 `qnn-model-lib-generator` 的 Android.mk 名称截断：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
qnn-model-lib-generator \
  -c qnn_quantization/06_onnx_qdq_matmul/generated/int8_per_axis_calibrated.cpp \
  -b qnn_quantization/06_onnx_qdq_matmul/generated/int8_per_axis_calibrated.bin \
  -t aarch64-android \
  -l axis_matmul_model \
  -o qnn_quantization/06_onnx_qdq_matmul/model_libs
```

## 设备执行

```bash
adb shell 'cd /data/local/tmp/qnn && \
export LD_LIBRARY_PATH="$PWD/qnn_qdq_matmul/lib:$PWD/lib:$LD_LIBRARY_PATH" && \
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp" && \
./bin/qnn-sample-app \
  --backend lib/libQnnHtp.so \
  --model qnn_qdq_matmul/lib/libaxis_matmul_model.so \
  --input_list qnn_qdq_matmul/input/input_list.txt \
  --output_dir qnn_qdq_matmul/output \
  --input_data_type float \
  --output_data_type float_only \
  --log_level info'
```

## 精度结果

```bash
python qnn_quantization/06_onnx_qdq_matmul/compare_output.py
```

| 实验 | max abs error | mean abs error | cosine | SQNR |
|---|---:|---:|---:|---:|
| QNN converter per-axis | 0.03450 | 0.00820 | 0.9998358 | 34.83 dB |
| NumPy per-axis weight | 0.02482 | 0.00176 | 0.9999850 | 45.21 dB |
| QNN per-tensor | 0.03943 | 0.00911 | 0.9997860 | 33.68 dB |

QNN per-axis 比 per-tensor 更准确，但优势小于 NumPy 权重量化对比。原因是 QNN
图的 output 也量化为 UINT8，输出 requantization 误差成为主要误差来源。下一步
应分别比较 INT8 output 与更高精度 output，拆分 weight、activation 和 output
三处量化误差。
