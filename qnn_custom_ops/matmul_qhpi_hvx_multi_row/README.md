# MatMul QHPI HVX Multi-Row 优化实验

本目录是 `../matmul_qhpi_hvx` 的独立优化版本，用于验证“一次计算多行，
复用 RHS 数据”对矩阵乘法性能的影响。

本文档记录了优化动机、实现过程、构建与部署命令、正确性测试、性能测试、
测试结果，以及实验期间遇到的问题。后续增加其他 MatMul 优化时，应保留
当前目录作为单一优化变量，避免直接叠加无关优化。

## 1. 实验目标

矩阵形状为：

```text
lhs    [B,H,M,K] = [1,1,128,256]
rhs    [B,H,K,N] = [1,1,256,256]
output [B,H,M,N] = [1,1,128,256]
```

总计算量为：

```text
M * K * N = 128 * 256 * 256 = 8,388,608 MACs
```

`N=256` 能被当前 64 列 HVX tile 整除，因此每个输出行包含 4 个完整的
64 列 tile，不会进入列方向 scalar tail。`M=128` 也能被 4 行 tile
整除，因此正式性能测试不会进入行方向 tail。

Baseline：

```text
qnn_custom_ops/matmul_qhpi_hvx
```

优化版本：

```text
qnn_custom_ops/matmul_qhpi_hvx_multi_row
```

两者使用完全相同的输入数据、矩阵尺寸、FP16/Q13 数值路径和输出格式。
主要差异只有 HVX kernel 的行 tile 大小。

## 2. Baseline 的主要问题

Baseline 每次计算一个 `1x64` 输出 tile。

对于每一个输出行和每一个 reduction 位置，Baseline 会执行：

1. 将一个 LHS FP16 标量转换为 Q13；
2. 将连续 64 个 RHS FP16 值逐个转换为 Q13；
3. 将 64 个 Q13 RHS 值加载为一个 HVX vector；
4. 广播 LHS Q13 标量；
5. 执行一次 64 lane 的 int16 x int16 -> int32 HVX multiply-accumulate。

核心结构可以简化为：

```cpp
for (row = 0; row < M; ++row) {
  for (col = 0; col < N; col += 64) {
    for (reduction = 0; reduction < K; ++reduction) {
      convert_rhs_fp16_to_q13(rhs[reduction][col : col + 64]);
      load_rhs_vector();
      accumulate_one_row();
    }
  }
}
```

计算相邻 4 个输出行时，它们使用的是同一段 RHS 数据，但 Baseline 会将
这 64 个 RHS 值重复转换和加载 4 次。

四行对应的工作量大致为：

```text
RHS FP16 -> Q13 转换：64 * 4 = 256 次
RHS HVX vector 加载： 4 次
HVX multiply-accumulate：4 次
```

其中 multiply-accumulate 不能减少，因为四个输出行必须有四份独立累加器；
但 RHS 转换和 RHS 加载可以在四行之间共享。

## 3. Multi-Row 优化设计

本实现将完整 HVX tile 从 `1x64` 改为 `4x64`。

每个 reduction 位置执行：

1. 将 64 个 RHS FP16 值转换为 Q13；
2. 只加载一次 RHS HVX vector；
3. 分别读取四个输出行对应的 LHS 标量；
4. 将四个 LHS 标量广播成四个 HVX vector；
5. 使用同一个 RHS vector 更新四个独立的 `HVX_VectorPair` accumulator。

简化后的循环结构为：

```cpp
for (row = 0; row + 4 <= M; row += 4) {
  for (col = 0; col + 64 <= N; col += 64) {
    acc0 = acc1 = acc2 = acc3 = 0;

    for (reduction = 0; reduction < K; ++reduction) {
      rhs_vec = convert_and_load_rhs_once();

      acc0 = vmpyacc(acc0, broadcast(lhs[row + 0]), rhs_vec);
      acc1 = vmpyacc(acc1, broadcast(lhs[row + 1]), rhs_vec);
      acc2 = vmpyacc(acc2, broadcast(lhs[row + 2]), rhs_vec);
      acc3 = vmpyacc(acc3, broadcast(lhs[row + 3]), rhs_vec);
    }

    store_four_output_rows();
  }
}
```

四行对应的工作量变为：

```text
RHS FP16 -> Q13 转换：64 次
RHS HVX vector 加载： 1 次
HVX multiply-accumulate：4 次
```

因此 RHS 转换和加载理论上减少到 Baseline 的四分之一。LHS 转换、四组
multiply-accumulate、输出转换和存储仍然必须执行，所以整体加速上限小于
或接近 4 倍。

## 4. Tail 处理

实现仍然支持任意合法的 `M` 和 `N`：

- `M` 剩余不足 4 行时，调用 Baseline 的 `1x64` HVX 路径；
- `N` 剩余不足 64 列时，调用 scalar dot-product 路径。

正式测试形状 `M=128, N=256` 不会进入这两个 tail，但保留 tail 可以避免
优化版本只能运行单一固定形状。

## 5. 独立命名

为了让 Baseline 和优化版本能够同时部署到设备，两者使用不同名称：

```text
QHPI Op:
  MatMulQhpiHvxMultiRow

OpPackage:
  MatMulQhpiHvxMultiRowOpPackage

ARM/HTP library:
  libQnnMatMulQhpiHvxMultiRowOpPackage.so

Model library:
  libcustom_matmul_qhpi_hvx_multi_row_model.so

Interface provider:
  MatMulQhpiHvxMultiRowOpPackageInterfaceProvider
```

因为 model 中写入了 Package 和 Op 名称，Baseline 的 model `.so` 不能直接
用于 Multi-Row 包。矩阵形状虽然相同，Multi-Row model 仍然必须重新生成。

输入数据和 `expected_float.raw` 可以复用，因为它们不包含 OpPackage 名称。

## 6. 目录结构

```text
matmul_qhpi_hvx_multi_row/
├── config/MatMulQhpiHvxMultiRowOpPackage.xml
├── htp/MatMulQhpiHvxMultiRowOpPackage/
│   ├── Makefile
│   ├── config/MatMulQhpiHvxMultiRowOpPackage.xml
│   └── src/
│       ├── MatMulQhpiHvxMultiRowOpPackageInterface.cpp
│       └── ops/MatMulQhpiHvxMultiRow.cpp
├── input/
│   ├── input_list.txt
│   ├── lhs.raw
│   └── rhs.raw
├── model/custom_matmul_qhpi_hvx_multi_row_model.cpp
├── model_libs/
├── scripts/generate_inputs.py
├── test_data/
│   ├── expected_float.raw
│   └── expected_fp16.raw
└── device_output/
```

## 7. 编译

以下命令均从仓库根目录执行：

```bash
cd ~/Cross-Backend_LLM_Inference_on_SnapDragon

PKG=qnn_custom_ops/matmul_qhpi_hvx_multi_row/htp/MatMulQhpiHvxMultiRowOpPackage
```

### 7.1 编译 HTP v75 DSP library

```bash
make -C "$PKG" htp_v75 \
  QNN_INCLUDE=/home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN \
  HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  HEXAGON_SDK_ROOT_V75=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  HEXAGON_TOOLS_VERSION_V75=8.7.06
```

产物：

```text
$PKG/build/hexagon-v75/libQnnMatMulQhpiHvxMultiRowOpPackage.so
```

### 7.2 编译 ARM OpPackage library

```bash
make -C "$PKG" htp_aarch64 \
  QNN_INCLUDE=/home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN \
  QNN_TARGET_LIB=/home/lingbok/Qualcomm/qairt/2.47.0.260601/lib/aarch64-android \
  HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  X86_LIBNATIVE_RELEASE_DIR=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0/tools/HEXAGON_Tools/8.7.06/Tools \
  ANDROID_NDK_ROOT=/home/lingbok/android/android-ndk-r28
```

产物：

```text
$PKG/build/aarch64-android/libQnnMatMulQhpiHvxMultiRowOpPackage.so
```

必须显式指定 `X86_LIBNATIVE_RELEASE_DIR`。生成的 Makefile 默认会尝试使用
Hexagon SDK 6.5 的 libnative，但当前环境实际使用 SDK 5.5.5。未指定时，
ARM 编译会出现 `HVX_Vector`、`HVX_VectorPair` 和 HVX intrinsic 未定义。

### 7.3 生成 model library

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qnn_custom_ops/matmul_qhpi_hvx_multi_row/model/custom_matmul_qhpi_hvx_multi_row_model.cpp \
  -t aarch64-android \
  -l custom_matmul_qhpi_hvx_multi_row_model \
  -o qnn_custom_ops/matmul_qhpi_hvx_multi_row/model_libs
```

产物：

```text
qnn_custom_ops/matmul_qhpi_hvx_multi_row/model_libs/aarch64-android/libcustom_matmul_qhpi_hvx_multi_row_model.so
```

### 7.4 确认生成了四组 HVX multiply-accumulate

```bash
/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0/tools/HEXAGON_Tools/8.7.06/Tools/bin/hexagon-llvm-objdump \
  -d "$PKG/build/hexagon-v75/libQnnMatMulQhpiHvxMultiRowOpPackage.so" |
  grep -E "vmpy|vmpyacc"
```

反汇编中已经观察到四个不同的 vector-pair accumulator，例如：

```text
v3:2.w += vmpy(...)
v7:6.w += vmpy(...)
v9:8.w += vmpy(...)
v5:4.w += vmpy(...)
```

这说明编译器确实生成了四行并行累加，而不是将源码重新折叠成单行。

## 8. 推送到设备

```bash
adb shell 'mkdir -p \
  /data/local/tmp/qnn/custom_matmul_qhpi_hvx_multi_row/lib \
  /data/local/tmp/qnn/custom_matmul_qhpi_hvx_multi_row/dsp \
  /data/local/tmp/qnn/custom_matmul_qhpi_hvx_multi_row/input \
  /data/local/tmp/qnn/custom_matmul_qhpi_hvx_multi_row/output'
```

```bash
adb push \
  "$PKG/build/aarch64-android/libQnnMatMulQhpiHvxMultiRowOpPackage.so" \
  /data/local/tmp/qnn/custom_matmul_qhpi_hvx_multi_row/lib/libQnnMatMulQhpiHvxMultiRowOpPackage_Cpu.so

adb push \
  "$PKG/build/hexagon-v75/libQnnMatMulQhpiHvxMultiRowOpPackage.so" \
  /data/local/tmp/qnn/custom_matmul_qhpi_hvx_multi_row/dsp/libQnnMatMulQhpiHvxMultiRowOpPackage_Htp.so

adb push \
  qnn_custom_ops/matmul_qhpi_hvx_multi_row/model_libs/aarch64-android/libcustom_matmul_qhpi_hvx_multi_row_model.so \
  /data/local/tmp/qnn/custom_matmul_qhpi_hvx_multi_row/lib/

adb push \
  qnn_custom_ops/matmul_qhpi_hvx_multi_row/input/. \
  /data/local/tmp/qnn/custom_matmul_qhpi_hvx_multi_row/input/
```

## 9. 执行

```bash
adb shell 'cd /data/local/tmp/qnn && \
rm -rf custom_matmul_qhpi_hvx_multi_row/output/* && \
export LD_LIBRARY_PATH="$PWD/custom_matmul_qhpi_hvx_multi_row/lib:$PWD/lib:$LD_LIBRARY_PATH" && \
export ADSP_LIBRARY_PATH="$PWD/custom_matmul_qhpi_hvx_multi_row/dsp;$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp" && \
./bin/qnn-sample-app \
  --backend lib/libQnnHtp.so \
  --model custom_matmul_qhpi_hvx_multi_row/lib/libcustom_matmul_qhpi_hvx_multi_row_model.so \
  --op_packages custom_matmul_qhpi_hvx_multi_row/lib/libQnnMatMulQhpiHvxMultiRowOpPackage_Cpu.so:MatMulQhpiHvxMultiRowOpPackageInterfaceProvider:CPU,libQnnMatMulQhpiHvxMultiRowOpPackage_Htp.so:MatMulQhpiHvxMultiRowOpPackageInterfaceProvider:HTP \
  --input_list custom_matmul_qhpi_hvx_multi_row/input/input_list.txt \
  --output_dir custom_matmul_qhpi_hvx_multi_row/output \
  --input_data_type float \
  --output_data_type float_only \
  --log_level info'
```

成功日志应包含：

```text
Loaded package MatMulQhpiHvxMultiRowOpPackage
Matched kernel 'matmulqhpihvxmultirow_float_16_Execute'
QnnGraph_execute done. status 0x0
```

`--op_packages` 必须同时注册：

1. ARM/CPU package：使用设备上的完整路径；
2. HTP package：使用 DSP 搜索路径中的 library basename。

仅传入一个 package 时，可能出现 HTP Skel 侧注册失败。

## 10. 正确性测试

拉回输出：

```bash
rm -rf qnn_custom_ops/matmul_qhpi_hvx_multi_row/device_output

adb pull \
  /data/local/tmp/qnn/custom_matmul_qhpi_hvx_multi_row/output \
  qnn_custom_ops/matmul_qhpi_hvx_multi_row/device_output
```

比较 float32 output：

```bash
python3 - <<'PY'
import numpy as np

output = np.fromfile(
    "qnn_custom_ops/matmul_qhpi_hvx_multi_row/device_output/Result_0/output.raw",
    dtype=np.float32,
)
expected = np.fromfile(
    "qnn_custom_ops/matmul_qhpi_hvx_multi_row/test_data/expected_float.raw",
    dtype=np.float32,
)

difference = np.abs(output - expected)
print("elements:", output.size)
print("max_abs_error:", float(difference.max()))
print("mean_abs_error:", float(difference.mean()))
print("allclose:", bool(np.allclose(output, expected, atol=1e-3, rtol=1e-3)))
print("nan:", int(np.isnan(output).sum()))
print("inf:", int(np.isinf(output).sum()))
PY
```

实测结果：

```text
elements:       32768
max_abs_error:  0.0009766817092895508
mean_abs_error: 0.00007510452996939421
allclose:       True
NaN / Inf:      0 / 0
```

该误差与 Baseline 完全一致，说明 Multi-Row 只复用了 RHS 转换和加载，没有
改变每个输出元素的 Q13 乘加顺序与数值语义。

## 11. 官方逐 Op Profiling

### 11.1 为什么不能直接使用预编译 SampleApp 查看事件

QAIRT 2.47 的预编译 `qnn-sample-app` 可以成功创建 detailed profile：

```text
Profiling turned on; level = 2
QnnProfile_create
```

但 SampleApp 源码使用 `QNN_DEBUG` 打印 `QnnProfile_getEvents()` 返回的事件。
设备上的 release binary 即使使用 `--log_level verbose`，也没有显示这些行。

尝试使用：

```text
--serialize_profile_logs
--system_library lib/libQnnSystem.so
```

会失败：

```text
QNN System function pointers are not populated.
Profiling Initialization failure
```

原因是当前 `libQnnSystem.so` 提供的 interface 中没有填充 profile
serialization 所需的三个 function pointer。这个问题只影响 profile 日志的
序列化，不代表 backend detailed profiling 本身不可用。

### 11.2 Profile SampleApp

仓库中保存了一份 QAIRT 2.47 SampleApp：

```text
qnn_custom_ops/tools/qnn_sample_app_profile
```

主要改动：

```cpp
QNN_DEBUG("Printing Event Info ...");
```

改为具有固定前缀的 INFO 日志：

```cpp
QNN_INFO("QNN_PROFILE_EVENT ...");
```

编译：

```bash
cd qnn_custom_ops/tools/qnn_sample_app_profile

/home/lingbok/android/android-ndk-r28/ndk-build \
  APP_ALLOW_MISSING_DEPS=true \
  APP_ABI=arm64-v8a \
  NDK_PROJECT_PATH=./ \
  NDK_APPLICATION_MK=make/Application.mk \
  APP_BUILD_SCRIPT=make/Android.mk \
  QNN_SDK_ROOT=/home/lingbok/Qualcomm/qairt/2.47.0.260601
```

推送时使用新名字，不覆盖原始工具：

```bash
adb push \
  libs/arm64-v8a/qnn-sample-app \
  /data/local/tmp/qnn/bin/qnn-sample-app-profile

adb shell 'chmod 755 /data/local/tmp/qnn/bin/qnn-sample-app-profile'
```

### 11.3 Multi-Row 性能测试命令

```bash
adb shell 'cd /data/local/tmp/qnn && \
export LD_LIBRARY_PATH="$PWD/custom_matmul_qhpi_hvx_multi_row/lib:$PWD/lib:$LD_LIBRARY_PATH" && \
export ADSP_LIBRARY_PATH="$PWD/custom_matmul_qhpi_hvx_multi_row/dsp;$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp" && \
./bin/qnn-sample-app-profile \
  --backend lib/libQnnHtp.so \
  --model custom_matmul_qhpi_hvx_multi_row/lib/libcustom_matmul_qhpi_hvx_multi_row_model.so \
  --op_packages custom_matmul_qhpi_hvx_multi_row/lib/libQnnMatMulQhpiHvxMultiRowOpPackage_Cpu.so:MatMulQhpiHvxMultiRowOpPackageInterfaceProvider:CPU,libQnnMatMulQhpiHvxMultiRowOpPackage_Htp.so:MatMulQhpiHvxMultiRowOpPackageInterfaceProvider:HTP \
  --input_list custom_matmul_qhpi_hvx_multi_row/input/input_list.txt \
  --output_dir custom_matmul_qhpi_hvx_multi_row/profile_output \
  --input_data_type float \
  --output_data_type float_only \
  --profiling_level detailed \
  --num_inferences 20 \
  --log_level info' 2>&1 |
grep 'identifier=MatMulQhpiHvxMultiRow_0.*cycles'
```

### 11.4 Baseline 性能测试命令

Baseline 使用同一个 `qnn-sample-app-profile`，但必须切换回 Baseline 的
model、package、input list 和 ADSP 路径：

```bash
adb shell 'cd /data/local/tmp/qnn && \
export LD_LIBRARY_PATH="$PWD/custom_matmul_qhpi_hvx/lib:$PWD/lib:$LD_LIBRARY_PATH" && \
export ADSP_LIBRARY_PATH="$PWD/custom_matmul_qhpi_hvx/dsp;$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp" && \
./bin/qnn-sample-app-profile \
  --backend lib/libQnnHtp.so \
  --model custom_matmul_qhpi_hvx/lib/libcustom_matmul_qhpi_hvx_model.so \
  --op_packages custom_matmul_qhpi_hvx/lib/libQnnMatMulQhpiHvxOpPackage_Cpu.so:MatMulQhpiHvxOpPackageInterfaceProvider:CPU,libQnnMatMulQhpiHvxOpPackage_Htp.so:MatMulQhpiHvxOpPackageInterfaceProvider:HTP \
  --input_list custom_matmul_qhpi_hvx/input/input_list.txt \
  --output_dir custom_matmul_qhpi_hvx/profile_output \
  --input_data_type float \
  --output_data_type float_only \
  --profiling_level detailed \
  --num_inferences 20 \
  --log_level info' 2>&1 |
grep 'identifier=MatMulQhpiHvx_0.*cycles'
```

事件示例：

```text
QNN_PROFILE_EVENT type=404 value=205504786 identifier=MatMulQhpiHvxMultiRow_0:OpId_18 (cycles) unit=3
```

其中：

- `type=404`：逐 Op profile event；
- `value`：该 Op 的 DSP cycles；
- `unit=3`：单位为 cycles；
- `identifier`：对应 graph 中的具体节点。

不要使用终端左侧的累计毫秒数作为 kernel 时间。它包含模型加载、Package
注册、graph prepare、RPC、输入输出文件处理和资源清理。

## 12. 正式性能结果

测试环境：

```text
SoC:             SM8650
HTP architecture: v75
QAIRT:           2.47.0.260601
Shape:           [1,1,128,256] x [1,1,256,256]
Profiling:       QNN detailed per-Op cycles
Runs:            20
Warm-up:         丢弃第 1 次
Statistics:      剩余 19 次取中位数
```

统计结果：

| 实现 | 中位数 cycles | 最小 cycles | 最大 cycles |
| --- | ---: | ---: | ---: |
| Baseline `1x64` | 793,889,648 | 782,945,312 | 801,008,396 |
| Multi-Row `4x64` | 205,504,786 | 205,388,952 | 205,643,915 |

加速比：

```text
speedup
  = baseline_median / multi_row_median
  = 793,889,648 / 205,504,786
  = 3.8631x
```

cycles 降低比例：

```text
cycle_reduction
  = 1 - multi_row_median / baseline_median
  = 74.1142%
```

此前使用 3 次执行得到过约 `3.72x` 的初步结果。该数据只用于确认优化方向，
最终应以去除手工计时探针后的 20 次官方逐 Op profiling 结果 `3.8631x` 为准。

结果接近理论上的 4 倍，说明 Baseline 的主要瓶颈确实是每个输出行重复进行
RHS FP16 -> Q13 转换和加载。没有达到正好 4 倍，是因为以下工作没有减少：

- 四个输出行仍需要四次 LHS 转换；
- 仍需要四组 HVX multiply-accumulate；
- 四组 accumulator 都需要转换并写回；
- 存在循环控制、地址计算和函数执行开销。

## 13. 实验期间遇到的问题

### 13.1 ADSP_LIBRARY_PATH 分隔符

`ADSP_LIBRARY_PATH` 在该环境中必须使用分号：

```text
path1;path2;path3
```

使用 Linux 常见的冒号会导致 DSP Skel 无法找到 HTP library，常见错误包括：

```text
DspTransport.openSession qnn_open failed
Failed to load skel
```

### 13.2 必须同时注册 CPU 和 HTP package

只传一个 package 时，ARM 侧可能加载成功，但 context 创建时 HTP Skel 注册
失败。最终使用：

```text
CPU library:完整路径:InterfaceProvider:CPU,
HTP library basename:InterfaceProvider:HTP
```

### 13.3 Model 不能复用

Baseline model 内部绑定：

```text
MatMulQhpiHvxOpPackage
MatMulQhpiHvx
```

Multi-Row model 必须绑定：

```text
MatMulQhpiHvxMultiRowOpPackage
MatMulQhpiHvxMultiRow
```

因此必须重新运行 `qnn-model-lib-generator`。

### 13.4 ARM 编译找不到 HVX 类型

默认 Makefile 指向当前环境中不存在的 Hexagon SDK 6.5 libnative，导致：

```text
unknown type name 'HVX_Vector'
unknown type name 'HVX_VectorPair'
use of undeclared identifier 'Q6_Vh_vsplat_R'
```

显式设置 SDK 5.5.5 的 `X86_LIBNATIVE_RELEASE_DIR` 后解决。

### 13.5 Profile 序列化不可用

`--profiling_level detailed` 本身可用，但 `--serialize_profile_logs` 因
QNN System serialization function pointers 未填充而失败。解决方案不是放弃
QNN profile，而是重新编译 SampleApp，将已有的 profile event 提升到 INFO
日志。

### 13.6 手工 HAP_perf/FARF 探针不可见

曾尝试在 Execute 内使用 `HAP_perf_get_pcycles()` 和 `FARF(ALWAYS)`。
计时代码可以编译运行，但当前 HTP 日志路径没有将 FARF 行输出到 SampleApp
终端。更重要的是，手工探针自身也会被官方逐 Op cycles 计入。

在官方 profile event 成功输出后，手工探针已从两个 kernel 中完全移除，
正式结果来自纯净内核。

## 14. 后续优化方向

Multi-Row 已经消除了四行之间重复的 RHS 转换，但每个 `4x64` tile 仍会在
每次执行时进行 scalar FP16 -> Q13 转换。后续可以分别建立独立目录验证：

```text
matmul_qhpi_hvx_vector_convert
matmul_qhpi_hvx_rhs_prepack
matmul_qhpi_hvx_vector_store
matmul_qhpi_hvx_multithread
matmul_qhpi_hvx_combined
```

建议保持每个目录只有一个主要优化变量，并始终使用相同输入、相同 shape、
相同 profile app 和相同统计方法进行比较。
