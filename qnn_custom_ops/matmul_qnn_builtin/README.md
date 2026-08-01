# QNN Builtin MatMul 基线

本目录使用 QNN 内置的 `qti.aisw::MatMul` 构建 FP16 MatMul，用于和
`matmul_qhpi_*` 自定义算子比较正确性与性能。该示例不需要注册自定义 Op
Package。

## Graph 与 Shape

Graph 边界使用 FP32，MatMul 内部使用 FP16：

```text
lhs FP32 ── Cast FP16 ──┐
                        ├─ qti.aisw::MatMul ─ Cast FP32 ─ output
rhs FP32 ── Cast FP16 ──┘
```

```text
lhs    [1,1,128,256]
rhs    [1,1,256,256]
output [1,1,128,256]
```

输入和参考输出与 `matmul_qhpi_hvx` 使用相同的数据，便于直接比较 builtin
MatMul 与自定义 HVX kernel。

## 目录内容

```text
matmul_qnn_builtin/
├── model/matmul_qnn_builtin_model.cpp       # QNN graph 定义
├── model_libs/aarch64-android/               # 生成的 model library
├── input/                                    # FP32 输入和 input_list
├── test_data/expected_float.raw              # FP32 参考输出
├── device_output/                            # 普通运行结果
└── device_output_profile/                    # detailed profiling 结果
```

`model_libs/`、`device_output/` 和 `device_output_profile/` 都是生成产物。

## 构建 Model Library

在仓库根目录执行：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qnn_custom_ops/matmul_qnn_builtin/model/matmul_qnn_builtin_model.cpp \
  -t aarch64-android \
  -l matmul_qnn_builtin_model \
  -o qnn_custom_ops/matmul_qnn_builtin/model_libs
```

生成文件：

```text
qnn_custom_ops/matmul_qnn_builtin/model_libs/aarch64-android/
libmatmul_qnn_builtin_model.so
```

## 设备运行

```bash
adb shell 'mkdir -p /data/local/tmp/qnn/matmul_qnn_builtin/{lib,input,output}'

adb push \
  qnn_custom_ops/matmul_qnn_builtin/model_libs/aarch64-android/libmatmul_qnn_builtin_model.so \
  /data/local/tmp/qnn/matmul_qnn_builtin/lib/

adb push \
  qnn_custom_ops/matmul_qnn_builtin/input/lhs.raw \
  qnn_custom_ops/matmul_qnn_builtin/input/rhs.raw \
  qnn_custom_ops/matmul_qnn_builtin/input/input_list.txt \
  /data/local/tmp/qnn/matmul_qnn_builtin/input/
```

假设 QNN backend 和 `qnn-net-run` 已部署到 `/data/local/tmp/qnn`：

```bash
adb shell 'cd /data/local/tmp/qnn && \
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH" && \
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp" && \
./bin/qnn-net-run \
  --backend lib/libQnnHtp.so \
  --model matmul_qnn_builtin/lib/libmatmul_qnn_builtin_model.so \
  --input_list matmul_qnn_builtin/input/input_list.txt \
  --output_dir matmul_qnn_builtin/output \
  --output_data_type float_only'
```

## Profiling

在运行命令中增加：

```text
--profiling_level detailed
--num_inferences 20
```

仓库保存的 `device_output_profile/profile.csv` 包含 `QnnBuiltinMatMul_0` 的
逐次执行周期，可作为自定义 MatMul 的 builtin 性能基线。比较时应使用相同的
shape、输入、运行次数和 profiling 配置，并排除首轮初始化影响。

## 正确性检查

```bash
python3 -c 'import numpy as np; \
actual=np.fromfile("qnn_custom_ops/matmul_qnn_builtin/device_output/Result_0/output.raw", dtype=np.float32); \
expected=np.fromfile("qnn_custom_ops/matmul_qnn_builtin/test_data/expected_float.raw", dtype=np.float32); \
diff=np.abs(actual-expected); \
print("max_abs_error", float(diff.max())); \
print("mean_abs_error", float(diff.mean())); \
print("allclose", bool(np.allclose(actual, expected, atol=1e-3, rtol=1e-3)))'
```
