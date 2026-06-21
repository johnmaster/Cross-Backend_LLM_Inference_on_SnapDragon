# QHPI FP16 Scalar MatMul

本目录实现一个基于 QHPI External Op Package API 的 FP16 矩阵乘法参考算子。

它原名为 `matmul_qhpi_hmx`。实际矩阵乘法一直由 C++ 标量循环完成，只有一条
`mxclracc.hf` 探测指令，并没有使用 HMX activation、weight 和 accumulator
数据通路。由于外部开发者缺少 HMX tile 布局与控制字规范，项目已更名并移除
所有 HMX 声明，避免把资源声明或单条指令误认为 HMX MatMul。

## 当前实现

- QHPI Op：`MatMulQhpiFp16Scalar`
- Op Package：`MatMulQhpiFp16ScalarOpPackage`
- 内部输入与输出：`QHPI_Float16`
- Tensor layout：`QHPI_Layout_Flat4`
- 执行资源：`QHPI_RESOURCE_MAIN`
- 计算方式：标量 FP16 输入、FP32 累加、FP16 输出
- 图边界：FP32 输入和输出，通过 QNN `Cast` 与算子内部 FP16 Tensor 连接

该版本的定位是正确性基线。后续 HVX 版本应与它比较输出，而不是直接修改本目录
并失去参考实现。

## 目录结构

```text
matmul_qhpi_fp16_scalar/
├── config/MatMulQhpiFp16ScalarOpPackage.xml
├── htp/MatMulQhpiFp16ScalarOpPackage/
│   ├── Makefile
│   ├── config/MatMulQhpiFp16ScalarOpPackage.xml
│   └── src/
│       ├── MatMulQhpiFp16ScalarOpPackageInterface.cpp
│       └── ops/MatMulQhpiFp16Scalar.cpp
├── model/custom_matmul_qhpi_fp16_scalar_model.cpp
├── input/
├── scripts/generate_inputs.py
└── test_data/
```

`MatMulQhpiFp16Scalar.cpp` 中的核心计算为：

```cpp
for (uint32_t row = 0; row < m; ++row) {
  for (uint32_t col = 0; col < n; ++col) {
    float sum = 0.0f;
    for (uint32_t reduction = 0; reduction < k; ++reduction) {
      sum += static_cast<float>(lhs[lhs_index]) *
             static_cast<float>(rhs[rhs_index]);
    }
    output[output_index] = static_cast<__fp16>(sum);
  }
}
```

## Graph 流程

```text
lhs FP32  -> Cast(FP16) --+
                            -> MatMulQhpiFp16Scalar -> Cast(FP32) -> output
rhs FP32  -> Cast(FP16) --+
```

模型中的 package name、op type 与动态库注册信息必须完全一致：

```text
PackageName = MatMulQhpiFp16ScalarOpPackage
OpType      = MatMulQhpiFp16Scalar
Provider    = MatMulQhpiFp16ScalarOpPackageInterfaceProvider
```

## 与其他版本的关系

| 目录 | API | 数据类型 | 实际计算资源 | 用途 |
|---|---|---|---|---|
| `matmul/` | Legacy HTP Op Package | FP32 | 标量 | Legacy API 入门 |
| `matmul_qhpi_scalar/` | QHPI | FP32 | 标量实现 | QHPI 流程入门 |
| `matmul_qhpi_fp16_scalar/` | QHPI | FP16 | 标量主线程 | FP16 正确性基线 |
| 后续 `matmul_qhpi_hvx/` | QHPI | FP16/FP32 | HVX | 向量化优化目标 |

## 重新生成与编译

改名后必须重新生成或重新编译 Op Package 和 model library。旧目录中的 `.so`、
`.o` 和 `.d` 是旧名称构建产物，不能作为新接口名的有效产物使用。

生成 Op Package 时使用：

```bash
qnn-op-package-generator \
  --config_path "$REPO/qnn_custom_ops/matmul_qhpi_fp16_scalar/config/MatMulQhpiFp16ScalarOpPackage.xml" \
  --output_path "$REPO/qnn_custom_ops/matmul_qhpi_fp16_scalar/htp" \
  --debug
```

后续实际编译前，应先清理旧 build 目录，再按现有 HTP/QHPI 编译流程生成新的
`libQnnMatMulQhpiFp16ScalarOpPackage.so` 和 `libcustom_matmul_qhpi_fp16_scalar_model.so`。

## 下一步

新建独立的 `matmul_qhpi_hvx/`，保持相同 shape、数据与期望输出，然后逐步加入：

1. HVX 128-byte 对齐和尾部处理；
2. FP16 向量加载与乘法；
3. reduction 累加；
4. 分块与多 slice；
5. 可选 VTCM 数据复用；
6. 数值、反汇编和性能三类验证。
