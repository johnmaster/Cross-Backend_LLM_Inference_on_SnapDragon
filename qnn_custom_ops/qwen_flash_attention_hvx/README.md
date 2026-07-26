# Qwen GQA FlashAttention QHPI/HTP

本目录从真实 Qwen2.5-0.5B layer0 decode graph 抽离 attention，并实现独立
QHPI fused operator：

```text
QKᵀ -> scale -> causal mask -> online Softmax -> probability × V
```

## 接口

```text
Q      FP32 [B, 14, Q, 64]
K      FP32 [B, KV, 64, 2]   # 直接匹配当前 QNN NHWC graph boundary
V      FP32 [B, 2, KV, 64]
output FP32 [B, 14, Q, 64]
```

`2` 是 KV head 数，`7` 是每个 KV head 对应的 query head 数。kernel 支持动态
`KV` 和 `Q`，并使用 `KV_BLOCK=32`。decode query 被视为 K/V 尾部的连续位置，
所以 query `i` 的可见长度为 `KV - Q + i + 1`。

## 实现状态

| 阶段 | 状态 |
|---|---|
| 独立 NumPy fixture | 已完成 |
| GQA-aware blockwise online softmax | 已完成 |
| QHPI 多 worker 分工 | 已完成 |
| Hexagon v75 与 ARM64 OpPackage 编译 | 已完成 |
| 真实 past128 graph patch | 已完成 |
| 真机正确性/profile | 已完成，当前版本未采用 |
| HVX QK dot 与 AV 累加 | 待实现 |

当前版本是 FlashAttention 算法基线：不物化 `[query_heads,Q,KV]`
score/probability，额外工作内存固定为一个 32-token score block、64 维 FP32
accumulator、running max 和 running sum。它注册为 `QHPI_RESOURCE_HVX` 以进入
HTP PluginOp worker，但核心 dot/exp/AV 循环当前仍是标量 FP32，不能把本阶段的
编译结果描述为“HVX SIMD 已完成”。

## 主机正确性

```bash
python3 qnn_custom_ops/qwen_flash_attention_hvx/tools/online_softmax_reference.py \
  --kv-length 129 \
  --block-size 32

python3 qnn_custom_ops/qwen_flash_attention_hvx/tools/generate_fixture.py
```

固定 seed 的 past128 fixture 中，blockwise online-softmax 与物化 attention：

```text
max_abs_error = 2.60770321e-08
mean_abs_error = 4.09351797e-09
cosine = 1.00000012
```

## 构建

```bash
bash qnn_custom_ops/qwen_flash_attention_hvx/scripts/build.sh
```

输出：

```text
htp/QwenFlashAttentionHvxOpPackage/build/hexagon-v75/
  libQnnQwenFlashAttentionHvxOpPackage.so
htp/QwenFlashAttentionHvxOpPackage/build/aarch64-android/
  libQnnQwenFlashAttentionHvxOpPackage.so
```

## 接入真实 Qwen decode graph

补丁以正式采用的 grouped-GQA delta-KV graph 为输入，删除独立
QK/scale/mask/Softmax/AV 节点，保留 projection、RoPE、KV Concat、o_proj、
MLP 和 delta-KV 输出：

```bash
bash qnn_custom_ops/qwen_flash_attention_hvx/scripts/build_qwen_graph.sh
```

生成：

```text
qwen_block_custom_qnn/generated/
  qwen2_0_5b_layer0_decode_past128_flash_attention.cpp
qnn_custom_ops/qwen_flash_attention_hvx/model_libs/aarch64-android/
  libqwen2_0_5b_layer0_decode_past128_flash_attention.so
```

运行时必须同时注册 ARM prepare package 和 Hexagon package：

```text
QwenFlashAttentionHvxOpPackage:CPU
QwenFlashAttentionHvxOpPackage:HTP
```

## 验证标准

真机结果必须同时满足：

1. `hidden_out` 与 grouped-GQA delta-KV baseline 比较误差和 cosine。
2. `current_key/current_value` 逐 bit 相同。
3. profile 中存在 `_QwenGqaFlashAttention`，且原 QK/Softmax/AV 不再执行。
4. 同时报告 fused-op cycles、root cycles、QNN us 和 NetRun us。
5. 与正式 baseline `10907 us` 及旧固定 past128 fused 实验 `19432 us` 比较。

## 第一次真机结果

最初保留 rank-5 Q 接口时，HTP graph finalize 返回错误 `1002`。QHPI
`Flat4` 改用物理等价的 `[1,14,1,64]` rank-4 接口后，graph 成功 finalize，
使用 4 个 HVX workers 完成 10 次执行。

与正式 grouped-GQA delta-KV baseline 比较：

```text
hidden_out   max_abs=0.0625  mean_abs=0.00815392  cosine=0.999947424
current_key  逐 bit 相同
current_value逐 bit 相同
```

```bash
python3 qnn_custom_ops/qwen_flash_attention_hvx/tools/compare_device_output.py
```

10 次 detailed profiling、去 warmup 中位数：

```text
root_cycles:        19,508,660
QNN accelerator:       21,213 us
QNN execute:           23,283 us
NetRun:                23,338 us
fused attention:   约 17.5M cycles
```

正式 builtin grouped-GQA delta-KV 为 `10907 us`，因此当前版本慢约
`12431 us / 113.97%`，也未达到现有 hidden 输出误差门槛。profile 中只有
`_QwenGqaFlashAttention`，原 QK/Softmax/AV 没有进入执行序列，说明负结果来自
custom kernel，而不是 graph 没有真正融合。

当前结论：工程抽离、动态 KV、causal blockwise online softmax 和真实 graph
接入已经完成；标量 FP32 版本只作为 correctness/performance baseline，不采用。
下一阶段必须先完成 HVX QK dot/reduction 和准确的向量化 AV 累加，再讨论替换
QNN builtin。

## 第二次真机结果：HVX IEEE-FP AV

先反汇编标量版本，确认 QK 和 AV 主循环均为标量 `sfmpy`，编译器没有自动生成
HVX FP32 指令。随后仅向量化 online-softmax 的 64 维 accumulator：

```text
old accumulator *= old_scale
accumulator += softmax_weight * V[token]
output = accumulator / running_sum
```

每个 64 维向量由两个 128-byte HVX vector 处理。v75 首次编译报错：

```text
Attempting to emit V6_vmpy_sf_sf instruction but
Feature_UseHVXIEEEFP predicate(s) are not met
```

原因不是 intrinsic 名称错误，而是 v75 FP32 HVX 指令需要显式启用
`-mhvx-ieee-fp`。该开关已记录在 OpPackage Makefile。重新构建 ARM prepare
package 和 Hexagon v75 package 后，真实 past128 graph 成功 finalize 并执行
10 次。

10 次 detailed profiling、去 warmup 中位数：

```text
                         标量 baseline       HVX AV          变化
fused attention cycles    17,618,383       11,371,942      -35.45%
root cycles               19,508,660       13,165,785      -32.51%
QNN accelerator               21,213 us        16,672 us
QNN execute                   23,283 us        19,208 us
NetRun                        23,338 us        19,234 us     -17.59%
```

数值结果：

```text
hidden_out   max_abs=0.22265625  mean_abs=0.00644305  cosine=0.999941244
current_key  逐 bit 相同
current_value逐 bit 相同
```

AV 向量化确实消除了约 624 万 fused-op cycles，但 v75 没有可直接替代标量
FP32 fused multiply-add 的 `Vsf × Vsf + Vsf` intrinsic；当前实现使用独立的
HVX multiply 和 add，改变了 129 个 token 的累加舍入顺序。没有发现索引或越界
问题，误差特征也与此前固定 past128 的 HVX-AV 实验一致。

结论：该实现保留在独立实验 OpPackage 中，便于复现和继续优化，但不替换正式
builtin attention。它仍比正式 `10907 us` baseline 慢 `76.34%`，且最大误差未
通过当前门槛。完整结果保存在：

```text
device_output/past128_hvx_av/
```

下一步不再继续堆叠 AV 近似，而是先把 K cache/graph boundary 调整为
`[B, KV_HEAD, KV, HEAD_DIM]` 连续布局，再实现两个 HVX vector 的 QK dot 和
reduction；当前 `[B, KV, HEAD_DIM, KV_HEAD]` 的 head-interleaved K 会使 gather
成本抵消 QK 向量乘法收益。

## 第三次真机结果：head-contiguous K + HVX QK

本轮新增独立 graph variant：

```text
past K:    [1,128,64,2] -> [1,2,128,64]
current K: 删除 NCHW -> NHWC Transpose
K Concat:  axis 1 -> axis 2
fused K:   [1,2,129,64]
```

对应文件：

```text
tools/patch_qwen_decode_head_contiguous.py
scripts/build_qwen_graph_head_contiguous.sh
qwen_block_custom_qnn/generated/
  qwen2_0_5b_layer0_decode_past128_flash_attention_head_contiguous.cpp
qwen_block_custom_qnn/tools/
  device_input_list_layer0_decode_past128_flash_head_contiguous.txt
```

QK 每个 64 维 dot 使用两个 128-byte HVX FP32 multiply。v75 没有直接 FP32
horizontal-reduction intrinsic，因此先把两个乘积 vector 写入 128-byte 对齐的
局部数组，再按标量顺序求和。与旧 HVX AV 版本相比，hidden/K/V 输出逐 bit
相同，说明 K layout 变换和 QK 路径没有引入额外输出变化。

同时将两个 64 维 AV accumulator vector 保持在 HVX 寄存器中，避免每个 token
都从局部数组读取并写回。10 次 detailed profiling、去 warmup 中位数：

```text
                              HVX AV      head-contiguous QK+AV    builtin
fused attention cycles     11,371,942             891,176           -
root cycles                13,165,785           2,768,200     2,231,170
QNN accelerator                16,672 us             8,910 us       9,493 us
QNN execute                    19,208 us            11,376 us      10,880 us
NetRun                         19,234 us            11,403 us      10,907 us
```

相对上一版：

```text
fused cycles: -92.16%
NetRun:       -40.72%
```

相对 builtin：

```text
QNN accelerator: 快 583 us / 6.14%
NetRun:          慢 496 us / 4.55%
```

正确性仍继承 FP32 HVX AV 的限制：

```text
hidden_out   max_abs=0.22265625  mean_abs=0.00644305  cosine=0.999941244
current_key  逐 bit 相同
current_value逐 bit 相同
```

这说明 head-contiguous 的收益已经得到真机证实，custom attention 在纯 accelerator
口径也首次快于 builtin；但端到端仍被 QNN/RPC 固定开销反超，而且 AV 最大误差未
通过门槛。因此当前仍是“实验采用、正式 graph 不替换”。

完整结果：

```text
device_output/past128_head_contiguous_qk/
device_output/past128_head_contiguous_reg_av/
```

### QFloat32 AV 对照实验（未采用）

曾把 AV accumulator 改为 QFloat32 扩展精度并保持在寄存器中：

```text
fused cycles=977,355
root cycles=2,800,244
NetRun=11,504 us
hidden max_abs=0.198242188
hidden mean_abs=0.00703363
hidden cosine=0.999941745
```

最大误差略有改善，但平均误差和性能退化，因此默认仍使用 FP32。该路径没有丢失，
可用下面的可选编译参数复现：

```bash
make ... QWEN_FLASH_EXTRA_FLAGS=-DQWEN_FLASH_AV_QFLOAT=1
```

结果保存在：

```text
device_output/past128_head_contiguous_qfloat/
```

真机注册 QHPI package 时必须写完整三段式语法：

```text
arm_prepare.so:QwenFlashAttentionHvxOpPackageInterfaceProvider:CPU,
libQnnQwenFlashAttentionHvxOpPackage.so:
QwenFlashAttentionHvxOpPackageInterfaceProvider:HTP
```

省略 InterfaceProvider 会被 QNN 错误解释，Context Creation 返回 package
registration `4005/4007`；该失败与 graph layout 或 HVX 内核无关。
