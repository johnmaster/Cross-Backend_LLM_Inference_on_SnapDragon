# QHPI Scalar MatMul

## 1. 这个例子是什么

本目录实现了一个基于 QHPI External Op Package API 的 FLOAT32 批量矩阵乘法：

```text
lhs    [B,H,M,K]
rhs    [B,H,K,N]
output [B,H,M,N]
```

计算公式：

```text
output[b,h,m,n] += lhs[b,h,m,k] * rhs[b,h,k,n]
```

当前实现使用普通 C++ 标量循环，没有调用 HVX 或 HMX intrinsic。kernel 必须声明为
`QHPI_RESOURCE_HVX` 才能被当前 HTP QHPI PluginOp 接受，因此它会占用一个 HVX
worker，但这不等于代码已经使用 HVX SIMD 指令。

这个例子的准确定位是：

```text
QHPI API + scalar MatMul + HVX worker scheduling
```

真正的 HVX 和 HMX 实现应分别放在独立目录中：

```text
qnn_custom_ops/matmul_qhpi_hvx/
qnn_custom_ops/matmul_qhpi_fp16_scalar/
```

## 2. QHPI 与 Legacy Op Package 的区别

Legacy 和 QHPI 都能访问 Tensor。QHPI 的重点不是“可以使用 Tensor”，而是通过
版本化 C API 明确描述 kernel 合同：

```text
输入输出的数据类型
Tensor layout 与 storage
Tensor 位于 DDR、TCM 或两者均可
kernel 需要 HVX 还是 HMX 资源
cost、predicate、tiling 和 rewrite 回调
```

本例的 Tensor signature 为：

```cpp
.element_type  = QHPI_Float32
.layout        = QHPI_Layout_Flat4
.storage       = QHPI_Storage_Direct
.mem_placement = QHPI_MemLoc_DDR_OR_TCM
```

运行时通过以下 API 访问 Tensor：

```cpp
qhpi_tensor_shape(tensor)
qhpi_tensor_raw_data(tensor)
```

## 3. `QHPI_RESOURCE_MAIN` 的实际限制

QAIRT 2.47 的 `qhpi.h` 定义了：

```cpp
QHPI_RESOURCE_MAIN = 1
QHPI_RESOURCE_HVX  = 2
QHPI_RESOURCE_HMX  = 4
```

但是在当前 SM8650 设备与 HTP 2.47 External PluginOp 路径中，把本例声明为
`QHPI_RESOURCE_MAIN` 会出现：

```text
ERROR: invalid resource flag 0x1
WARNING: Selecting disabled op
runlist = ... + 0 vec + 0 mtx
```

即使最后出现：

```text
Graph execution finished with result 0
```

也不能说明自定义 MatMul 已经执行。Graph API 成功返回与自定义 kernel 真正进入
runlist 是两个不同的验证条件。

将资源改为：

```cpp
.resources = QHPI_RESOURCE_HVX
```

之后日志变为：

```text
Matched kernel 'matmulqhpiscalar_float_32_Execute', creating PluginOp!
runlist = 14 + 1 vec + 0 mtx + 0 elt + 0 non-run
```

这证明 kernel 被调度到 HVX worker，但仍不能证明编译结果包含 HVX 指令。

## 4. 代码和生成文件

### 4.1 XML 定义

```text
config/MatMulQhpiScalarOpPackage.xml
```

关键属性：

```xml
<OpDefCollection
        PackageName="MatMulQhpiScalarOpPackage"
        UseQHPI="true">
```

XML 定义了两个 FLOAT32 输入、一个 FLOAT32 输出以及 HTP backend。

### 4.2 QHPI Op 实现

```text
htp/MatMulQhpiScalarOpPackage/src/ops/MatMulQhpiScalar.cpp
```

重要对象：

```text
QHPI_Tensor_Signature_v1  描述输入输出 Tensor
QHPI_Kernel_v1            描述执行函数和资源
QHPI_OpInfo_v1            描述完整 Op
qhpi_register_ops_v1()    向 HTP 注册 QHPI Op
```

需要开发者重点实现的函数：

```cpp
matmulqhpiscalar_float_32_Execute(...)
```

它负责：

```text
检查输入输出数量
读取并验证 [B,H,M,K]、[B,H,K,N]、[B,H,M,N]
取得三个 Tensor 的 raw data
执行五重循环 MatMul
写入 output Tensor
```

### 4.3 Package Interface

```text
htp/MatMulQhpiScalarOpPackage/src/MatMulQhpiScalarOpPackageInterface.cpp
```

它提供两类入口：

```text
MatMulQhpiScalarOpPackageInterfaceProvider  QNN Op Package ABI 入口
qhpi_init                                   QHPI 插件注册入口
```

### 4.4 Model Library 源码

```text
model/custom_matmul_qhpi_scalar_model.cpp
```

模型中的 `addNode()` 必须和 XML、Package Interface 完全一致：

```cpp
model.addNode(
    QNN_OPCONFIG_VERSION_1,
    "MatMulQhpiScalar_0",
    "MatMulQhpiScalarOpPackage",
    "MatMulQhpiScalar",
    nullptr,
    0,
    inputNames,
    2,
    outputTensors,
    1);
```

## 5. ARM64 与 Hexagon 双库

HTP External Op Package 需要编译两份同名源码：

```text
build/aarch64-android/libQnnMatMulQhpiScalarOpPackage.so
build/hexagon-v75/libQnnMatMulQhpiScalarOpPackage.so
```

职责不同：

```text
ARM64 library
  -> Android 进程加载
  -> 与 libQnnHtpPrepare.so 协作
  -> 提供 Op 定义、验证和图准备信息

Hexagon v75 library
  -> CDSP/FastRPC 加载
  -> 注册 QHPI kernel
  -> 执行 matmulqhpiscalar_float_32_Execute()
```

部署时建议明确改名：

```text
libQnnMatMulQhpiScalarOpPackage_Cpu.so
libQnnMatMulQhpiScalarOpPackage_Htp.so
```

改名只是设备端区分文件，不改变库内的 PackageName。

## 6. CDSP 动态加载路径

Android ARM64 库由 `LD_LIBRARY_PATH` 搜索，Hexagon 库由 `ADSP_LIBRARY_PATH`
搜索：

```bash
export LD_LIBRARY_PATH="$PWD/custom_matmul_qhpi_scalar/lib:$PWD/lib:$LD_LIBRARY_PATH"

export ADSP_LIBRARY_PATH="$PWD/custom_matmul_qhpi_scalar/dsp;$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
```

`--op_packages` 中 ARM64 项可以使用 Android 相对路径，但 HTP 项应只传裸文件名：

```text
custom_matmul_qhpi_scalar/lib/libQnnMatMulQhpiScalarOpPackage_Cpu.so:
MatMulQhpiScalarOpPackageInterfaceProvider:CPU

libQnnMatMulQhpiScalarOpPackage_Htp.so:
MatMulQhpiScalarOpPackageInterfaceProvider:HTP
```

错误写法：

```text
custom_matmul_qhpi_scalar/dsp/libQnnMatMulQhpiScalarOpPackage_Htp.so:...:HTP
```

这种写法可能在 context create 阶段出现：

```text
Failed to register op package on the Skel side
err = 4005
err = 4007
```

原因是 CDSP 动态加载器不会按 Android shell 当前目录解析该相对路径。

## 7. 验证标准

一次真正成功的实验必须同时满足以下条件：

1. ARM64 与 Hexagon package 均注册成功。
2. 日志出现 `Matched kernel`。
3. runlist 出现 `1 vec`。
4. 不出现 `invalid resource flag`。
5. Graph finalize 与 execute 成功。
6. 输出与 NumPy `np.matmul(lhs, rhs)` 一致。

本例已验证：

```text
Matched kernel: true
runlist: 1 vec + 0 mtx
NumPy allclose: true
```

## 8. 不应得出的结论

本例不能证明：

```text
编译器生成了 HVX SIMD 指令
Tensor 实际分配在 VTCM
HMX matrix thread 被使用
HMX intrinsic 已执行
```

真正的 HVX 实现还需要在反汇编中确认 `v*` 向量指令；真正的 HMX 实现需要：

```text
QHPI_RESOURCE_HMX
HMX 支持的数据类型和 Tensor layout
VTCM 数据搬运与 tile
Q6_* HMX intrinsic
runlist 中 mtx > 0
反汇编中出现 mx* 指令
数值结果与参考实现一致
```
