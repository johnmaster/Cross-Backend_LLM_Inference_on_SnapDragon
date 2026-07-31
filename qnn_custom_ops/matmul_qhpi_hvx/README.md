# QHPI HVX MatMul

本目录使用 QHPI External Op Package API 实现 FP16 批量 MatMul 自定义算子。

## 当前实现

- QHPI 算子：`MatMulQhpiHvx`
- Op Package：`MatMulQhpiHvxOpPackage`
- 内部输入和输出：`QHPI_Float16`
- Tensor layout：`QHPI_Layout_Flat4`
- 运行时资源：`QHPI_RESOURCE_HVX`
- Shape 约定：`[B,H,M,K] x [B,H,K,N] -> [B,H,M,N]`
- Graph 边界：应用侧 tensor 为 FP32，自定义算子前后使用 QNN `Cast` 与 FP16 互转

Hexagon/libnative 路径使用 HVX 整数乘加计算完整的 64 列输出 tile：

1. 将 FP16 输入转换为 Q13 定点值。
2. 使用 `Q6_Vh_vsplat_R` 广播一个 Q13 LHS 标量。
3. 使用 `vmemu` 加载连续的 64 个 Q13 RHS 列元素。
4. 使用 `Q6_Ww_vmpyacc_WwVhVh` 将 int16 × int16 累加到 int32 lane。
5. 反量化 int32 累加值并写入 FP16 输出。

不足 64 列的尾部使用 FP16 标量输入、FP32 累加的 fallback 路径，从而保证任意
`N` 都能得到正确结果。

最初使用 `Q6_Wsf_vmpyacc_WsfVhfVhf` 的 FP16 浮点 HVX 原型能够在本设备上完成
编译和调度，但运行时返回全零。因此，当前提交的实现使用上述整数 HVX 路径作为
已验证正确性的基线。

## 示例 Shape

仓库中的示例模型使用：

```text
lhs    [1,1,128,256]
rhs    [1,1,256,256]
output [1,1,128,256]
```

该 shape 特意设置 `N=256`，因此每个输出行会执行四个完整的 64 列 HVX tile。
较大的 `M` 和 `K` 便于比较不同矩阵乘优化方案。

## 文件

```text
matmul_qhpi_hvx/
├── config/MatMulQhpiHvxOpPackage.xml
├── htp/MatMulQhpiHvxOpPackage/
│   ├── Makefile
│   ├── config/MatMulQhpiHvxOpPackage.xml
│   └── src/
│       ├── MatMulQhpiHvxOpPackageInterface.cpp
│       └── ops/MatMulQhpiHvx.cpp
├── model/custom_matmul_qhpi_hvx_model.cpp
├── input/
├── scripts/generate_inputs.py
└── test_data/
```

## 构建

需要重新生成或更新 Package 骨架时执行：

```bash
qnn-op-package-generator \
  --config_path "$REPO/qnn_custom_ops/matmul_qhpi_hvx/config/MatMulQhpiHvxOpPackage.xml" \
  --output_path "$REPO/qnn_custom_ops/matmul_qhpi_hvx/htp" \
  --debug
```

构建 HTP Package：

```bash
cd "$REPO/qnn_custom_ops/matmul_qhpi_hvx/htp/MatMulQhpiHvxOpPackage"
make htp_v75
make htp_aarch64
```

预期生成的动态库名称：

```text
libQnnMatMulQhpiHvxOpPackage.so
libcustom_matmul_qhpi_hvx_model.so
```

## 验证

重新生成示例输入和预期输出：

```bash
python3 qnn_custom_ops/matmul_qhpi_hvx/scripts/generate_inputs.py
```

构建 Hexagon 动态库后，检查其中是否包含 HVX 整数乘法指令：

```bash
llvm-objdump -d build/hexagon-v75/libQnnMatMulQhpiHvxOpPackage.so | \
  grep -E "vmpy|vmpyacc|vsplat|vmem"
```

按照 FP16 结果容差，将设备输出与 `test_data/expected_float.raw` 比较。当前 Q13
HVX 路径在示例输入上的实测结果为：

```text
max_abs_error  = 0.0002441704
mean_abs_error = 0.0000352946
allclose(atol=1e-3, rtol=1e-3) = true
```

## HVX Intrinsic 说明

`matmulqhpihvxCompute64ColumnsHvx()` 是本 Package 的本地辅助函数，不是
Qualcomm API。函数内部使用的 HVX intrinsic 来自 Hexagon SDK 和 QNN HTP 头文件。

相关源文件：

```text
Hexagon SDK intrinsic 声明：
/local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0/tools/HEXAGON_Tools/8.7.06/Tools/target/hexagon/include/hvx_hexagon_protos.h

QNN HTP 封装：
/home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN/HTP/core/intrinsics.h
```

检索命令：

```bash
rg "Q6_.*vmpy|Q6_.*vadd|Q6_.*vsplat|Q6_.*vcvt|Q6_.*vmem" \
  /local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.5.0/tools/HEXAGON_Tools/8.7.06/Tools/target/hexagon/include/hvx_hexagon_protos.h

rg "vmemu|q6op_|HVX_Vector|HVX_VectorPair" \
  /home/lingbok/Qualcomm/qairt/2.47.0.260601/include/QNN/HTP/core/intrinsics.h
```

当前 kernel 使用以下 HVX 类型和 intrinsic：

```cpp
HVX_Vector
HVX_VectorPair
Q6_Vh_vsplat_R(...)
Q6_Ww_vmpyacc_WwVhVh(...)
Q6_V_lo_W(...)
Q6_V_hi_W(...)
vmemu(...)
```

使用 `-mhvx-length=128B` 编译时，`HVX_Vector` 表示一个 128 字节 HVX vector。
对于 16-bit lane，每个 vector 包含 64 个 lane；对于 32-bit lane，每个 vector
包含 32 个 lane。

`HVX_VectorPair` 由两个 HVX vector 组成。许多 widening 操作需要使用 vector
pair，因为 64 个 int16 lane 相乘会产生 64 个 int32 结果，需要
`64 * 4 = 256` 字节。

名称解析示例：

```text
Q6_Ww_vmpyacc_WwVhVh
   Q6       Hexagon intrinsic 命名空间
   Ww       结果是由 32-bit word 组成的 vector pair
   vmpyacc  vector multiply accumulate
   Ww       累加器输入是由 32-bit word 组成的 vector pair
   Vh       第一个乘数是由 16-bit halfword 组成的 vector
   Vh       第二个乘数是由 16-bit halfword 组成的 vector
```

该 intrinsic 的含义近似为：

```cpp
int32_vector_pair += int16_vector * int16_vector;
```

本 Package 在执行 HVX 乘法前将 FP16 输入转换为 Q13 定点值：

```cpp
const int16_t lhs_q13 = matmulqhpihvxFloatToQ13(...);
const HVX_Vector lhs_vec = Q6_Vh_vsplat_R(lhs_q13);
acc = Q6_Ww_vmpyacc_WwVhVh(acc, lhs_vec, vmemu(rhs_q13));
```

`Q6_Vh_vsplat_R()` 将一个 16-bit 标量广播到 `HVX_Vector` 的全部 64 个
halfword lane。

`vmemu(ptr)` 是 QNN HTP 对非对齐 HVX vector load/store 的封装。这里用它访问
栈缓冲区和输出缓冲区，从而不要求指针按 128 字节对齐：

```cpp
HVX_Vector rhs_vec = vmemu(rhs_q13);
vmemu(acc_lo_store) = Q6_V_lo_W(acc);
```

`Q6_V_lo_W()` 和 `Q6_V_hi_W()` 将 `HVX_VectorPair` 拆分为低位和高位 vector。
对于 `Q6_Ww_vmpyacc_WwVhVh`，低位 vector 包含偶数输出 lane，高位 vector
包含奇数输出 lane，因此写回时需要交错排列：

```cpp
output[col + 2 * lane]     = dequantized_low_lane;
output[col + 2 * lane + 1] = dequantized_high_lane;
```

### 如何确认使用的是 HVX 而不是 HMX

确认本 Package 使用 HVX 的依据：

- 代码使用 `HVX_Vector` 和 `HVX_VectorPair`。
- intrinsic 声明位于 `hvx_hexagon_protos.h`。
- 生成代码反汇编后包含以下 vector 指令：

```text
v5:4.w += vmpy(v0.h,v1.h)
v0.h = vsplat(r2)
vmem(...)
```

这些都是普通的 HVX vector-lane 指令。HMX 需要专用头文件、intrinsic、
matrix/tile layout 以及 accumulator/control 路径。本 Package 没有使用任何 HMX
intrinsic 或 HMX tile 数据路径。
