# MatMul QHPI HVX 8-row LHS tile-cache FP32 store

本目录用于验证 `PackLhs + MatMul` 融合方案。custom op 的输入仍然是
FP16 LHS/RHS，输出为 FP32；区别是 op 内部会按 8 行 tile 把 LHS 转成 Q13
小缓存，然后 HVX MatMul 内层循环直接读取缓存中的 Q13 LHS。

## 优化思路

早期 `LHS-prepack` 版本在 QNN graph 中新增了独立 `PackLhs` 节点：

```text
FP16 LHS -> PackLhs(Q13) -> MatMul
```

这个版本改成：

```text
FP16 LHS/RHS -> MatMulQhpiHvx8RowLhsTileCacheFp32Store
```

当前 op 也支持可选第三输入：

```text
FP16 LHS/RHS + optional FP32 bias -> MatMulQhpiHvx8RowLhsTileCacheFp32Store
```

其中 bias shape 可以是 `[N]` 或 `[1,1,1,N]`。这个 fused-bias 路径用于
`qwen_block_custom_qnn` 中的实验，但目前数值误差比外部 QNN Add 版本更大，
因此还不能作为默认推荐路径。

op 内部执行：

1. 每个 8 行 tile 只把 LHS 从 FP16 转 Q13 一次；
2. 每个 `8x64` 输出 tile 复用这份 Q13 LHS；
3. RHS 仍在 MatMul 内按 64 列向量转成 Q13；
4. accumulator 使用 Q26，最后转 FP32 写出。

这样可以减少图中的中间 tensor，并避免原始 kernel 在每个输出列块中重复转换同一份
LHS。

## 编译

```bash
PKG=qnn_custom_ops/matmul_qhpi_hvx_8row_lhs_tile_cache_fp32_store/htp/MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage

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
```

输出文件：

```text
htp/MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage/build/hexagon-v75/
  libQnnMatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage.so

htp/MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage/build/aarch64-android/
  libQnnMatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage.so
```

## tiny_llm_block_custom_matmul 中的用法

生成替换 `q_proj` 的 prefill model source：

```bash
python tiny_llm_block_custom_matmul/tools/patch_prefill_q_proj_lhs_tile_cache.py
```

编译 model lib：

```bash
PATH=/home/lingbok/android/android-ndk-r28:$PATH \
/home/lingbok/Qualcomm/qairt/2.47.0.260601/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c tiny_llm_block_custom_matmul/generated/tiny_block_prefill_q_proj_lhs_tile_cache.cpp \
  -b tiny_llm_block/generated/tiny_block_prefill.bin \
  -t aarch64-android \
  -l tiny_block_prefill_q_proj_lhs_tile_cache \
  -o tiny_llm_block_custom_matmul/model_libs
```

详细的设备运行、拉回 profile、CSV 解释和横向结果记录在：

```text
tiny_llm_block_custom_matmul/README.md
```

当前 prefill graph 实测摘要：

```text
custom_lhs_tile_cache root_cycles=480235 qnn_us=4821 netrun_us=6238 q_proj_cycles=146244
```

相比独立 `PackLhs` 版本，q_proj custom op cycles 从百万级下降到约 `146k`。
端到端仍慢于 QNN builtin，后续优化重点应该转向减少 Cast/Reshape、外部 OpPackage
调度开销，以及静态 RHS weight 的真正离线 prepack。
