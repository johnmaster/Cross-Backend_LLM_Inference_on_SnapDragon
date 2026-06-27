# MatMul QHPI HVX Multi-Row + Vector Convert

本目录是在 `../matmul_qhpi_hvx_multi_row` 基础上的累计优化版本：

1. 保留 `4x64` multi-row tile，在四行之间复用 RHS；
2. 将每次 64 个 RHS FP16 -> Q13 标量转换替换为 HVX 向量转换。

测试矩阵保持不变：

```text
[1,1,128,256] x [1,1,256,256] -> [1,1,128,256]
```

## 1. 优化动机

Multi-row 版本已经把 RHS 转换次数降低到原来的四分之一，但每个 reduction
位置仍然包含以下标量循环：

```cpp
alignas(128) int16_t rhs_q13[64];

for (uint32_t lane = 0; lane < 64; ++lane) {
  rhs_q13[lane] = floatToQ13(static_cast<float>(rhs[lane]));
}

const HVX_Vector rhs_vec = vmemu(rhs_q13);
```

对于一个 `4x64` tile：

```text
K = 256
每个 K 位置转换 64 个 RHS
总计 256 * 64 = 16,384 次标量转换
```

整个矩阵还有 32 个四行 tile 和 4 个列 tile，因此标量转换仍然是主要热点。

## 2. 向量转换实现

当前实现直接加载 64 个 FP16：

```cpp
const HVX_Vector fp16_values = vmemu(values);
```

然后调用 QAIRT HTP helper：

```cpp
return hnnx::s16_from_hf_rnd_sat<13>(fp16_values);
```

该 helper 使用 qf32 中间结果，并通过 HVX 指令完成：

- FP16/QF32 multiply；
- rounding；
- int16 saturation；
- overflow compare 和 select。

转换后的 Q13 vector 直接传给四个 accumulator：

```cpp
acc0 = Q6_Ww_vmpyacc_WwVhVh(acc0, lhs0, rhs_vec);
acc1 = Q6_Ww_vmpyacc_WwVhVh(acc1, lhs1, rhs_vec);
acc2 = Q6_Ww_vmpyacc_WwVhVh(acc2, lhs2, rhs_vec);
acc3 = Q6_Ww_vmpyacc_WwVhVh(acc3, lhs3, rhs_vec);
```

因此不再需要：

```text
rhs_q13[64] stack buffer
64 次 scalar FP16 -> float
64 次 scalar float -> Q13
一次从 stack 重新加载 RHS vector
```

## 3. 数值风险

QAIRT 2.47 的 `hvx_mathops.h` 注释说明：

```text
s16_from_hf_rnd_sat<FBITS>()
完整 sweep-test 范围为 FBITS=-2..9
```

本实现需要 Q13，因此使用 `FBITS=13`。头文件同时说明 `FBITS>=10` 可能出现
内部舍入误差。

这意味着：

- 当前 benchmark 输入已经通过正确性测试；
- 当前输出与 multi-row 输出逐字节一致；
- 这还不能证明所有 FP16 bit pattern 都与 scalar Q13 转换完全一致；
- 扩展到模型真实权重前，应增加 FP16 全 bit-pattern 或分层边界测试；
- 如果发现差异，需要实现专用的、经过验证的 HVX Q13 conversion。

这个限制必须保留在实验结论中，不能因为性能提升明显而忽略。

## 4. 失败过的转换方案

最初尝试了两阶段转换：

```cpp
scaled = Q6_Vhf_vmpy_VhfVhf(fp16_values, fp16(16.0));
q13 = hnnx::s16_from_hf_rnd_sat<9>(scaled);
```

数学上：

```text
2^4 * 2^9 = 2^13
```

而且 `FBITS=9` 在 QAIRT 完整 sweep-test 范围内。但该方案在 SM8650/v75
设备上输出全零：

```text
output min: 0
output max: 0
nonzero:    0
```

反汇编中确实存在：

```text
v1.hf = vmpy(v0.hf, ...)
```

说明它不是“编译器没有生成指令”，而是 direct half-float HVX multiply 在
当前运行环境中再次出现“指令可编译、运行结果为零”的问题。该问题和早期
FP16 HVX MatMul prototype 的现象一致。

最终版本绕过 direct half-float multiply，改用 qf32 中间路径。

## 5. 独立命名

```text
Directory:
  matmul_qhpi_hvx_multi_row_vector_convert

QHPI Op:
  MatMulQhpiHvxMultiRowVectorConvert

OpPackage:
  MatMulQhpiHvxMultiRowVectorConvertOpPackage

Interface provider:
  MatMulQhpiHvxMultiRowVectorConvertOpPackageInterfaceProvider

Model:
  libcustom_matmul_qhpi_hvx_multi_row_vector_convert_model.so
```

新 model 必须重新生成，因为 model 内部绑定 Package 和 Op 名称。

## 6. 编译

从仓库根目录执行：

```bash
PKG=qnn_custom_ops/matmul_qhpi_hvx_multi_row_vector_convert/htp/MatMulQhpiHvxMultiRowVectorConvertOpPackage
```

HTP v75：

```bash
make -C "$PKG" htp_v75 \
  QNN_INCLUDE=/home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN \
  HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  HEXAGON_SDK_ROOT_V75=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  HEXAGON_TOOLS_VERSION_V75=8.7.06
```

ARM package：

```bash
make -C "$PKG" htp_aarch64 \
  QNN_INCLUDE=/home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN \
  QNN_TARGET_LIB=/home/lingbok/Qualcomm/qairt/2.47.0.260601/lib/aarch64-android \
  HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  X86_LIBNATIVE_RELEASE_DIR=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0/tools/HEXAGON_Tools/8.7.06/Tools \
  ANDROID_NDK_ROOT=/home/lingbok/android/android-ndk-r28
```

Model：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qnn_custom_ops/matmul_qhpi_hvx_multi_row_vector_convert/model/custom_matmul_qhpi_hvx_multi_row_vector_convert_model.cpp \
  -t aarch64-android \
  -l custom_matmul_qhpi_hvx_multi_row_vector_convert_model \
  -o qnn_custom_ops/matmul_qhpi_hvx_multi_row_vector_convert/model_libs
```

## 7. 反汇编验证

```bash
/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0/tools/HEXAGON_Tools/8.7.06/Tools/bin/hexagon-llvm-objdump \
  -d "$PKG/build/hexagon-v75/libQnnMatMulQhpiHvxMultiRowVectorConvertOpPackage.so" |
  grep -E "\\.hf|qf32|vround|vmux|vmpy"
```

已经观察到：

```text
v3:2.qf32 = vmpy(...)
v10.h = vround(...):sat
v12 = vmux(...)
v3:2.w += vmpy(...)
v19:18.w += vmpy(...)
v15:14.w += vmpy(...)
v9:8.w += vmpy(...)
```

## 8. 设备执行

设备目录：

```text
/data/local/tmp/qnn/custom_matmul_qhpi_hvx_multi_row_vector_convert
```

执行命令：

```bash
adb shell 'cd /data/local/tmp/qnn && \
rm -rf custom_matmul_qhpi_hvx_multi_row_vector_convert/output/* && \
export LD_LIBRARY_PATH="$PWD/custom_matmul_qhpi_hvx_multi_row_vector_convert/lib:$PWD/lib:$LD_LIBRARY_PATH" && \
export ADSP_LIBRARY_PATH="$PWD/custom_matmul_qhpi_hvx_multi_row_vector_convert/dsp;$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp" && \
./bin/qnn-sample-app \
  --backend lib/libQnnHtp.so \
  --model custom_matmul_qhpi_hvx_multi_row_vector_convert/lib/libcustom_matmul_qhpi_hvx_multi_row_vector_convert_model.so \
  --op_packages custom_matmul_qhpi_hvx_multi_row_vector_convert/lib/libQnnMatMulQhpiHvxMultiRowVectorConvertOpPackage_Cpu.so:MatMulQhpiHvxMultiRowVectorConvertOpPackageInterfaceProvider:CPU,libQnnMatMulQhpiHvxMultiRowVectorConvertOpPackage_Htp.so:MatMulQhpiHvxMultiRowVectorConvertOpPackageInterfaceProvider:HTP \
  --input_list custom_matmul_qhpi_hvx_multi_row_vector_convert/input/input_list.txt \
  --output_dir custom_matmul_qhpi_hvx_multi_row_vector_convert/output \
  --input_data_type float \
  --output_data_type float_only \
  --log_level info'
```

成功时会命中：

```text
matmulqhpihvxmultirowvectorconvert_float_16_Execute
```

## 9. 正确性结果

```text
elements:       32768
output min:     -1.3535157442092896
output max:      1.4541016817092896
nonzero:         32768
max_abs_error:   0.0009766817092895508
mean_abs_error:  0.00007510452996939421
allclose(1e-3):  True
NaN / Inf:       0 / 0
```

与 `matmul_qhpi_hvx_multi_row` 的输出：

```text
byte-identical: True
```

## 10. 性能测试

使用：

```text
/data/local/tmp/qnn/bin/qnn-sample-app-profile
--profiling_level detailed
--num_inferences 20
```

丢弃第一轮，剩余 19 次取中位数。

| 实现 | 中位数 cycles | 最小 cycles | 最大 cycles |
| --- | ---: | ---: | ---: |
| Baseline `1x64` | 793,889,648 | 782,945,312 | 801,008,396 |
| Multi-row `4x64` | 205,504,786 | 205,388,952 | 205,643,915 |
| Multi-row + vector convert | 15,042,197 | 14,978,439 | 15,107,041 |

相对原始 Baseline：

```text
speedup = 793,889,648 / 15,042,197 = 52.7775x
cycles reduction = 98.1053%
```

相对 Multi-row：

```text
speedup = 205,504,786 / 15,042,197 = 13.6619x
cycles reduction = 92.6804%
```

## 11. 结论

本实验说明：

1. Multi-row 消除四行之间重复的 RHS 转换，带来约 `3.86x`；
2. 剩余的 scalar FP16 -> Q13 转换仍然占据绝大多数 cycles；
3. 使用 HVX qf32 中间路径进行 64 lane Q13 转换后，总加速达到 `52.78x`；
4. 当前 benchmark 数值正确，但 `FBITS=13` 的全 FP16 范围正确性仍需专项验证；
5. direct half-float vector multiply 在当前设备上不可依赖。

下一步优先事项不是继续叠加优化，而是建立 Q13 vector conversion 的边界和
全 bit-pattern 正确性测试。只有该转换在目标输入域内被充分验证后，才适合
继续优化输出转换、RHS prepack 或多线程。

### 输出向量化失败记录

后续曾尝试使用以下 HVX 路径替换 scalar 输出转换：

```text
int32 accumulator
  -> Q6_Vsf_equals_Vw
  -> Q6_Vsf_vmpy_VsfVsf (乘 2^-26)
  -> Q6_Vhf_vcvt_VsfVsf
  -> 128-byte vector store
```

该版本能够编译、注册并成功执行，但设备输出仍然全部为零：

```text
output min: 0
output max: 0
nonzero:    0
```

因此当前 SM8650/v75 环境中不仅 direct FP16 vector multiply 不可用，
direct FP32 -> FP16 HVX conversion 也不能直接作为可靠写回路径。该失败版本
没有保留为独立优化目录，后续仍使用已经验证的 scalar 输出转换。
