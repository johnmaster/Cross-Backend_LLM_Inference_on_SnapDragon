# MatMul QHPI HVX 8-row FP32 store

本目录基于 `../matmul_qhpi_hvx_8row_vector_convert`，单独验证输出阶段优化。

## 优化思路

原 8-row kernel 在每个 `8x64` tile 计算完成后：

1. 将 8 组 32-bit accumulator 写入栈上数组；
2. 标量执行 `int32 -> float -> fp16`；
3. 逐元素写入 FP16 tensor；
4. 图中的 Cast 节点再把 FP16 输出转换成 FP32。

本版本改成：

1. `convert_s32_to_sf` 向量化执行 `int32 -> FP32`；
2. 直接调整 IEEE FP32 exponent，精确乘以 Q26 反量化系数 `2^-26`；
3. `Q6_W_vshuff_VVR(..., -4)` 恢复偶数列、奇数列的顺序；
4. 每行用两次 128-byte HVX store 写出 64 个 FP32；
5. custom op 直接输出 FP32，删除图末尾的 FP16-to-FP32 Cast。

输入和矩阵计算部分保持不变，因此可以与 8-row vector-convert 版本直接比较。

## 编译

```bash
PKG=qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store/htp/MatMulQhpiHvx8RowFp32StoreOpPackage

make -C "$PKG" htp_v75 \
  QNN_INCLUDE=/home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN \
  HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  HEXAGON_SDK_ROOT_V75=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  HEXAGON_TOOLS_VERSION_V75=8.7.06

make -C "$PKG" htp_aarch64 \
  QNN_INCLUDE=/home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN \
  QNN_TARGET_LIB=/home/lingbok/Qualcomm/qairt/2.47.0.260601/lib/aarch64-android \
  HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0 \
  X86_LIBNATIVE_RELEASE_DIR=/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0/tools/HEXAGON_Tools/8.7.06/Tools \
  ANDROID_NDK_ROOT=/home/lingbok/android/android-ndk-r28

PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store/model/custom_matmul_qhpi_hvx_8row_fp32_store_model.cpp \
  -t aarch64-android \
  -l custom_matmul_qhpi_hvx_8row_fp32_store_model \
  -o qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store/model_libs
```

## 输出文件

```text
HTP:
htp/MatMulQhpiHvx8RowFp32StoreOpPackage/build/hexagon-v75/
  libQnnMatMulQhpiHvx8RowFp32StoreOpPackage.so

ARM:
htp/MatMulQhpiHvx8RowFp32StoreOpPackage/build/aarch64-android/
  libQnnMatMulQhpiHvx8RowFp32StoreOpPackage.so

Model:
model_libs/aarch64-android/
  libcustom_matmul_qhpi_hvx_8row_fp32_store_model.so
```

## 正确性判断

设备输出已经是 FP32，可直接与 `test_data/expected_float.raw` 比较：

```bash
python3 -c 'import numpy as np; o=np.fromfile("qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store/device_output/Result_0/output.raw",np.float32); e=np.fromfile("qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store/test_data/expected_float.raw",np.float32); d=np.abs(o-e); print("max_abs_error",d.max()); print("mean_abs_error",d.mean()); print("allclose",np.allclose(o,e,atol=1e-3,rtol=1e-3))'
```

FP32 store 消除了最终 FP16 舍入，因此结果可能比 FP16-output 版本更接近参考值，但 Q13 输入量化误差仍然存在。

设备实测：

```text
elements:       32768
max_abs_error:  0.0007113218307495117
mean_abs_error: 0.00009107097139349207
allclose(1e-3): True
NaN / Inf:      0 / 0
```

## Profiling 结果

使用 `qnn-sample-app-profile --profiling_level detailed --num_inferences 20`，
丢弃第一次 warm-up：

```text
median: 12,940,799 cycles
min:    12,777,916 cycles
max:    13,112,935 cycles
```

与 8-row vector-convert 的 `14,554,119 cycles` 相比：

```text
加速比:     1.125x
cycles 降低: 11.08%
```

## 遇到的问题

第一次实现使用 `Q6_Vsf_vmpy_VsfVsf` 做 FP32 向量缩放，设备输出全部为
零。目标 QHPI/HVX 路径上的 vector FP multiply 与此前失败的 FP16 vector
store 实验表现一致。

由于反量化比例严格等于 `2^-26`，最终实现不再执行 FP multiply，而是：

1. `convert_s32_to_sf` 转成 IEEE FP32；
2. 对非零 lane 的 exponent 减 26；
3. 保留零 lane 为精确的 `0.0f`。

这样既绕开了异常的 FP multiply，也不会引入额外的乘法舍入。
