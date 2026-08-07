# QNN custom-op optimization sequence

一级目录按优化依赖关系编号。后一个目录不一定完整继承前一个目录；RHS/LHS
prepack、多线程和 tile-cache 是从共同基线展开的并行分支。

| 编号 | 目录 | 定位或主要变化 |
|---:|---|---|
| 01 | `01_matmul` | Legacy HTP custom-op 入门实现 |
| 02 | `02_matmul_qhpi_scalar` | QHPI FP32 标量正确性基线 |
| 03 | `03_matmul_qhpi_fp16_scalar` | FP16 输入、FP32 累加的标量基线 |
| 04 | `04_matmul_qhpi_hvx` | Q13 HVX `1x64` MatMul 基线 |
| 05 | `05_matmul_qhpi_hvx_multi_row` | 扩展为 `4x64`，跨行复用 RHS |
| 06 | `06_matmul_qhpi_hvx_multi_row_vector_convert` | RHS FP16→Q13 向量化 |
| 07 | `07_matmul_qhpi_hvx_8row_vector_convert` | 扩展为 `8x64`，进一步复用 RHS |
| 08 | `08_matmul_qhpi_hvx_8row_rhs_prepack` | 独立 PackRhs 实验 |
| 09 | `09_matmul_qhpi_hvx_8row_fp32_store` | accumulator 向量转换并直接写 FP32 |
| 10 | `10_matmul_qhpi_hvx_8row_lhs_prepack_fp32_store` | 独立 PackLhs 实验 |
| 11 | `11_matmul_qhpi_hvx_8row_both_prepack_fp32_store` | 同时预转换 LHS 和 RHS |
| 12 | `12_matmul_qhpi_hvx_8row_lhs_tile_cache_fp32_store` | 将 PackLhs 融合为算子内局部缓存 |
| 13 | `13_matmul_qhpi_hvx_8row_fp32_store_multithread` | QHPI self-slicing 与局部 LHS tile-cache |
| 14 | `14_matmul_qhpi_hvx_8row_lhs_prepack_fp32_store_multithread` | LHS prepack 与 self-slicing 组合 |
| 15 | `15_matmul_qnn_builtin` | QNN builtin MatMul 性能对照 |
| 16 | `16_qhpi_precompute_probe` | QHPI precompute 生命周期探测 |
| 17 | `17_qwen_flash_attention_hvx` | 从 MatMul 扩展到 Qwen fused attention |
| 18 | `18_tools` | 公共 profiling 和宿主侧辅助工具，不是 kernel 优化阶段 |

MatMul 的主要演进主线为：

```text
01 Legacy custom op
  -> 02/03 scalar baselines
  -> 04 HVX 1x64
  -> 05 HVX 4x64
  -> 06 vector conversion
  -> 07 HVX 8x64
  -> 09 direct FP32 store
  -> 12 fused LHS tile-cache
  -> 13 multithread + fused LHS tile-cache
```

从 `07` 和 `09` 展开的预打包分支为：

```text
07 -> 08 RHS prepack
09 -> 10 LHS prepack -> 11 both prepack
                    -> 14 LHS prepack + multithread
09 -> 12 fused LHS tile-cache -> 13 multithread + fused LHS tile-cache
```
