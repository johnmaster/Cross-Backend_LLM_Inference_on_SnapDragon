# W8A8 与 W8A16 MatMul

本实验保持以下条件不变：

- 相同 FP32 ONNX MatMul
- 相同输入和 calibration 数据
- 相同 RHS per-axis INT8 encoding
- 相同 HTP 设备

唯一变量是 activation 和 output 位宽：

```text
W8A8:  INT8 per-axis weight + UINT8 activation/output
W8A16: INT8 per-axis weight + UINT16 activation/output
```

## 目的

`07_quantization_error_decomposition` 显示，W8A8 的主要误差来自 activation 和
output，而不是 per-axis weight。本实验用 W8A16 验证这个判断。

## 转换

先加载环境：

```bash
conda activate qairt-2.47
source /home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/envsetup.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

生成 W8A16：

```bash
qnn-onnx-converter \
  -i qnn_quantization/06_onnx_qdq_matmul/model/fp32_matmul.onnx \
  -o qnn_quantization/08_w8a16_matmul/generated/w8a16_matmul.cpp \
  --input_list \
    qnn_quantization/06_onnx_qdq_matmul/input/converter_input_list.txt \
  --quantization_overrides \
    qnn_quantization/06_onnx_qdq_matmul/model/quantization_overrides.json \
  --act_bitwidth 16 \
  --weights_bitwidth 8 \
  --param_quantizer_schema symmetric
```

生成代码中应看到：

```text
LHS:    QNN_DATATYPE_UFIXED_POINT_16
RHS:    QNN_DATATYPE_SFIXED_POINT_8 + AXIS_SCALE_OFFSET
OUTPUT: QNN_DATATYPE_UFIXED_POINT_16
```

## 编译

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
qnn-model-lib-generator \
  -c qnn_quantization/08_w8a16_matmul/generated/w8a16_matmul.cpp \
  -b qnn_quantization/08_w8a16_matmul/generated/w8a16_matmul.bin \
  -t aarch64-android \
  -l w8a16_model \
  -o qnn_quantization/08_w8a16_matmul/model_libs
```

## 精度比较

```bash
python qnn_quantization/08_w8a16_matmul/compare_output.py
```

设备执行已成功。最终数字由上述脚本生成。

| 模型 | max error | mean error | cosine | SQNR |
|---|---:|---:|---:|---:|
| W8A8 per-axis | 0.03450 | 0.00820 | 0.9998358 | 34.83 dB |
| W8A16 per-axis | 0.01671 | 0.00125 | 0.9999925 | 48.16 dB |

W8A16 基本恢复到“仅 per-axis 权重量化”时约 48 dB 的水平，验证了 W8A8
中的主要附加误差来自 activation/output 量化。

## 代价

W8A16 的 activation/output tensor 是 W8A8 的两倍大小。本次 HTP prepare
日志中，I/O tensor allocation 从约 `65536` 增加到 `131072` 字节。

这说明 mixed precision 是精度、内存带宽和执行速度之间的折中。不能仅根据
精度选择 W8A16。

## Profiling

使用官方 `qnn-net-run --profiling_level detailed --num_inferences 5`，再通过
`qnn-profile-viewer` 生成 CSV。首轮包含冷启动影响，因此排除第 1 次，对后 4
次取 median：

```bash
python qnn_quantization/08_w8a16_matmul/compare_profile.py
```

| 模型 | accelerator cycles median | MatMul cycles median |
|---|---:|---:|
| W8A8 | 12,129.5 | 7,326.5 |
| W8A16 | 29,909.5 | 15,581.0 |

W8A16 相对 W8A8：

```text
总 accelerator cycles: 2.47x
MatMul cycles:          2.13x
```

因此本例中，W8A16 将 SQNR 从 `34.83 dB` 提升到 `48.16 dB`，代价是 MatMul
cycles 超过两倍，activation/output 内存也增加一倍。这正是 mixed precision
需要权衡的精度、速度和内存三角。
