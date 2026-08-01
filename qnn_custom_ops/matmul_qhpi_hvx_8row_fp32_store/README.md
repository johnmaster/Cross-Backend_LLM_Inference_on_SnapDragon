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

### 完整数据流

这个版本不修改 `8x64` MatMul 主循环。LHS 和 RHS 仍先转换为 Q13，使用八组
`HVX_VectorPair` 将 int16 乘积累加到 int32：

```text
FP16 LHS/RHS
    -> Q13 int16
    -> int16 x int16
    -> int32 Q26 accumulator
```

优化发生在 K 维累加结束之后。原版本和当前版本的输出路径分别是：

```text
原版本：
HVX int32 accumulator
    -> 栈上 int32 数组
    -> 标量 int32 -> FP32 -> FP16
    -> FP16 tensor
    -> QNN Cast
    -> FP32 graph output

当前版本：
HVX int32 accumulator
    -> HVX int32 -> FP32
    -> HVX Q26 反量化
    -> HVX 偶数/奇数 lane 重排
    -> FP32 graph output
```

因此性能收益并不是来自减少输出数据量，而是来自消除栈中转、逐元素转换和图末尾
的 Cast。FP32 输出实际占用的内存是 FP16 输出的两倍。

### Q26 转换为 FP32

两个 Q13 数相乘后得到 Q26，所以 int32 accumulator 需要乘以 `2^-26` 才能恢复
真实数值。代码先用 `convert_s32_to_sf()` 一次将 32 个 int32 lane 转换成 IEEE
FP32：

```cpp
const HVX_Vector fp32 = convert_s32_to_sf(values);
```

目标设备上的 HVX FP32 vector multiply 会返回全零，因此不能直接执行
`fp32 * 2^-26`。由于这个比例是严格的二次幂，代码改为将 IEEE FP32 exponent
减 26：

```cpp
const HVX_Vector exponent_delta = Q6_V_vsplat_R(-218103808);
const HVX_Vector scaled = Q6_Vw_vadd_VwVw(fp32, exponent_delta);
```

其中：

```text
-218103808 = -26 * 2^23
```

FP32 exponent field 从 bit 23 开始，所以这相当于精确乘以 `2^-26`。零值没有
普通的 exponent，代码使用 predicate 和 `Q6_V_vmux_QVV()` 保留精确的
`0.0f`：

```cpp
const HVX_VectorPred is_zero = Q6_Q_vcmp_eq_VwVw(values, zero);
return Q6_V_vmux_QVV(is_zero, zero, scaled);
```

#### `ConvertQ26ToFp32()` 中的 HVX 操作

完整函数为：

```cpp
static inline HVX_Vector
matmulqhpihvx8rowfp32storeConvertQ26ToFp32(HVX_Vector values) {
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector fp32 = convert_s32_to_sf(values);
  const HVX_Vector exponent_delta = Q6_V_vsplat_R(-218103808);
  const HVX_Vector scaled = Q6_Vw_vadd_VwVw(fp32, exponent_delta);
  const HVX_VectorPred is_zero = Q6_Q_vcmp_eq_VwVw(values, zero);
  return Q6_V_vmux_QVV(is_zero, zero, scaled);
}
```

在 128 字节 HVX 模式下，一个 `HVX_Vector` 可容纳 32 个 int32 或 FP32 lane。
函数中的操作含义如下：

| 操作 | 含义 | 本函数中的作用 |
| --- | --- | --- |
| `Q6_V_vzero()` | 生成全零 vector | 同时作为 int32 零和 FP32 `0.0f` |
| `convert_s32_to_sf(values)` | 将 32 个有符号 int32 数值转换为 32 个 IEEE FP32 | 将 Q26 accumulator 转为仍带 `2^26` scale 的 FP32 |
| `Q6_V_vsplat_R(x)` | 将一个 32-bit 标量广播到全部 word lane | 为每个 FP32 lane 准备相同的 exponent 调整量 |
| `Q6_Vw_vadd_VwVw(a,b)` | 对 32 个 32-bit word lane 做整数加法 | 将 FP32 位模式的 exponent 同时减 26，不是浮点加法 |
| `Q6_Q_vcmp_eq_VwVw(a,b)` | 逐个比较 32-bit word lane，返回 predicate | 标记原始 Q26 accumulator 等于零的 lane |
| `Q6_V_vmux_QVV(q,a,b)` | predicate 为真时选 `a`，否则选 `b` | 零 lane 返回 `0.0f`，其余返回缩放结果 |

`convert_s32_to_sf()` 是 QAIRT helper；在当前 v75 上映射到
`Q6_Vsf_equals_Vw()`。这是数值转换，例如将整数 `67108864` 转成浮点数
`67108864.0f`，不是直接把整数 bit pattern 重新解释成 FP32。

#### 为什么是 `26 << 23`

这里正确的表达式是 `26 << 23`，不是 `23 << 26`：

- `26` 来自 Q13 x Q13 = Q26，恢复真实值需要乘以 `2^-26`；
- `23` 来自 IEEE FP32 exponent 字段的最低位是 bit 23。

IEEE FP32 位布局为：

```text
bit 31       bit 30 ... 23       bit 22 ... 0
┌──────────┬───────────────────┬────────────────────┐
│ sign     │ exponent          │ mantissa           │
│ 1 bit    │ 8 bits            │ 23 bits            │
└──────────┴───────────────────┴────────────────────┘
```

因此 exponent 改变 1，对整个 32-bit 位模式来说就是改变 `1 << 23`；exponent
减 26 就是从位模式中减去 `26 << 23`：

```text
26 << 23 = 26 * 8,388,608 = 218,103,808
```

代码将其写成负的广播常量，然后使用整数 vector add：

```cpp
const HVX_Vector exponent_delta = Q6_V_vsplat_R(-(26 << 23));
scaled_bits = Q6_Vw_vadd_VwVw(fp32_bits, exponent_delta);
```

源代码中的 `-218103808` 与 `-(26 << 23)` 完全相同。对于非零正常 FP32，位模式
变化可以表示为：

```text
sign + (exponent << 23) + mantissa
  -> sign + ((exponent - 26) << 23) + mantissa
```

mantissa 不变，数值由 `mantissa * 2^E` 变为 `mantissa * 2^(E-26)`，正好完成
`x * 2^-26`。例如 Q26 整数 `67108864 = 2^26` 转换为 FP32 后 exponent 减 26，
结果就是 `2^0 = 1.0f`。

FP32 零的位模式没有普通 exponent，不能直接执行上述减法，因此函数先用
`Q6_Q_vcmp_eq_VwVw()` 找出零 lane，再通过 `Q6_V_vmux_QVV()` 将它们恢复为
精确的 `0.0f`。当前输入来自 int32 accumulator，非零值除以 `2^26` 后仍处于
正常 FP32 表示范围。

### 恢复输出列顺序

`Q6_Ww_vmpyacc_WwVhVh()` 的 widening 结果分布为：

```text
Q6_V_lo_W(acc): 偶数列 0, 2, 4, ..., 62
Q6_V_hi_W(acc): 奇数列 1, 3, 5, ..., 63
```

转换成 FP32 后，使用 4 字节粒度的 shuffle 将它们恢复为连续列顺序：

```cpp
const HVX_VectorPair interleaved = Q6_W_vshuff_VVR(odd, even, -4);
```

```text
even:       0 2 4 6 ...
odd:        1 3 5 7 ...
interleaved:0 1 2 3 ... 63
```

一行 64 个 FP32 共 256 字节，而一个 HVX vector 是 128 字节，因此每行使用
两次 vector store：

```cpp
vmemu(&output[row_output_base]) = Q6_V_lo_W(interleaved);
vmemu(&output[row_output_base + 32]) = Q6_V_hi_W(interleaved);
```

### Graph 输出变化

原版本的 Custom Op 输出 FP16，然后由 QNN Cast 转换成 FP32。当前版本同时将
QHPI output signature 和 model graph output 改成 FP32：

```text
QHPI output:  QHPI_Float32
QNN output:   QNN_DATATYPE_FLOAT_32 / QNN_TENSOR_TYPE_APP_READ
```

因此 Custom Op 的两次 HVX store 直接写入最终 graph output，model 中不再需要
FP16-to-FP32 Cast。这样也消除了最终 FP16 舍入，但 LHS/RHS 的 Q13 输入量化误差
仍然存在。

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
