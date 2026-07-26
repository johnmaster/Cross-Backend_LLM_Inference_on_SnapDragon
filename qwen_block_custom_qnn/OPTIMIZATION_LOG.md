# Qwen q_proj custom HTP 优化记录

本文件用于固定优化顺序、对应代码、真机结果和是否采用，避免后续复现时只看到最终源码。
测试环境为 Snapdragon 8 Gen 3 / OnePlus 12、QAIRT 2.47、HTP v75，模型为
Qwen2.5-0.5B-Instruct layer0 prefill、`B=1`、`S=16`。

| 顺序 | 版本 | 核心变化 | NetRun 中位数 | 状态 |
|---:|---|---|---:|---|
| 0 | QNN builtin | converter 原始 graph | 14,826 us | 基线 |
| 1 | old custom | 8-row、单线程、LHS tile cache | 16,310 us | custom 基线 |
| 2 | multithread 8-row | QHPI self-slicing + worker-private LHS Q13 cache | 15,861 us | 已验证 |
| 3 | multithread 4-row | `M=16` 时四个 worker 各处理 4-row tile | 15,884 us | kernel wall time 更好，但端到端持平 |
| 4 | 最小 precompute probe | 16-byte graph-load 数据、MAIN thread | 可执行 | API 已验证；先前失败来自错误 provider 参数 |
| 5 | 完整 static RHS precompute | graph load 读取 `[896,896]` FP32 权重并写 Q13 | execution failure | tensor metadata 可用，但当前 runtime 不支持可靠遍历完整 input payload |
| 6 | 离线数学 Q13 RHS | 删除 q_proj RHS Cast，BIN 直接保存 INT16 Q13 | 16,745 us | 不采用：数值精度退化，且不能复现 HTP 的未受支持 `<13>` 转换 |
| 7 | 设备导出 Q13 RHS | 在 HTP 上生成与旧 kernel 逐 bit 相同的 INT16 RHS，再嵌入 BIN | 13,152 us | **当前采用：输出逐 bit 相同，快于 custom 和 builtin** |
| 8 | device-Q13 4x128 | 每次 reduction 复用 LHS splat 到两个 64-column RHS vector | 13,105 us | **当前采用：q_proj cycles 降低 29.52%，输出逐 bit 相同** |
| 9 | 4x128 reduction unroll2 | 每轮显式处理两个连续 K 元素 | 13,091 us | 不采用：q_proj cycles 增加 15.53%，NetRun 差异属于整图波动 |
| 10 | 精简 device-Q13 BIN | 删除已不再引用的 FP32 q_proj payload | 13,077 us | **采用：model lib 减少 3,211,536 bytes，输出逐 bit 相同** |

采用版本的主要文件：

```text
qnn_custom_ops/matmul_qhpi_hvx_8row_fp32_store_multithread/
qwen_block_custom_qnn/tools/patch_qwen_prefill_q_proj_multithread_tile_cache.py
qwen_block_custom_qnn/tools/run_multithread_tile_cache_repro.sh
qwen_block_custom_qnn/tools/device_input_list_layer0_prefill_seq16.txt
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_custom_multithread_tile_cache.cpp
```

历史 profile：

```text
qwen_block_custom_qnn/device_output/q_proj_custom_multithread_tile_cache_layer0_prefill_seq16/
qwen_block_custom_qnn/device_output/q_proj_custom_multithread_4row_layer0_prefill_seq16/
```

一键复现：

```bash
bash qwen_block_custom_qnn/tools/run_multithread_tile_cache_repro.sh
```

脚本会保存 raw 输出、原始 profiling log、解码后的 CSV、正确性摘要和性能摘要。
环境变量和结果字段说明见本目录 `README.md` 的 M6。

独立 probe 位于 `qnn_custom_ops/qhpi_precompute_probe/`。它证明 QAIRT 2.47 /
HTP v75 能够执行 `do_precomputation_function` 和
`function_with_precomputed_data`。此前的 `Context Creation failure` 根因是
`--op_packages` 错写成不存在的 `interfaceProvider` 符号，并遗漏 CPU/HTP 双端配置，
不是 precompute API 或 `PREPARE_DISABLED` 不兼容。

进一步验证了 `128 KiB`、`1 MiB` 和约 `1.53 MiB` 的 header-only buffer，三者均可
完成 inference，因此失败不是 `precomputed_data_size` 容量上限。只有在 graph-load
阶段遍历 static RHS payload 并写入 Q13 cache 后才出现 execution failure；即使只缓存
前 512 列（约 896 KiB）也失败。当前结论是 QHPI precompute 可用于 shape、量化参数和
少量派生 metadata，但不能依赖输入 tensor 的完整 payload 在该阶段可读。完整 static
RHS op 未进入默认注册表。

## 离线 Q13 RHS 负实验

为绕过 graph-load precompute 对 static tensor payload 的限制，增加了纯离线方案：

```text
qwen_block_custom_qnn/tools/generate_q_proj_q13_model_bin.py
qwen_block_custom_qnn/tools/patch_qwen_prefill_q_proj_offline_q13.py
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_q13.bin
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_q13.json
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_offline_q13.cpp
qwen_block_custom_qnn/device_output/q_proj_offline_q13_layer0_prefill_seq16/
```

该版本把原始 FP32 q_proj 权重先舍入为 IEEE FP16，再按 Q13 转为 INT16，删除
`_MatMul_rhs_cast_fp16`，由新增的
`MatMulQhpiHvxOfflineQ13RhsFp32StoreMultithread` 直接读取 INT16 RHS。模型可以正常
finalize 和执行，证明 INT16 static tensor 直达 QHPI kernel 的路径可用。

但它没有通过正确性门槛：

| 版本 | vs FP32 reference `hidden_out` max abs | mean abs | cosine |
|---|---:|---:|---:|
| 旧 4-row custom | 0.0183071 | 0.00119501 | 0.999998629 |
| 离线数学 Q13 | 0.0990594 | 0.00797660 | 0.999942243 |

离线版本相对旧 custom 的 `hidden_out` 最大差值为 `0.0971680`，而另外两个 graph
输出逐 bit 相同，误差已定位到 q_proj 路径。性能也没有获益：10 次 detailed
profiling 的 NetRun 中位数为 `16,745 us`，慢于采用版本的 `15,884 us`。

根因线索来自 QAIRT 2.47 的 `HTP/core/hvx_mathops.h`：

- `s16_from_hf_rnd_sat` 的 half-way 规则是 away-from-zero；
- 实现只保证 `FBITS=-2..9`；
- 当前 custom kernel 使用 `s16_from_hf_rnd_sat<13>`，超出保证范围，头文件明确
  提醒 `FBITS>=10` 会出现内部舍入误差。

因此不能用 NumPy 的标准数学 Q13 值替代旧 kernel 的设备转换值。曾额外测试
ties-to-even，结果同样不通过，说明问题不是简单的 tie-breaking。当前默认注册仍保留
原 FP16 RHS op；offline-Q13 op 只作为实验接口存在，不应替换复现脚本中的采用版本。

下一步若继续离线化，必须先构造独立 HTP conversion probe，把所有 FP16 权重通过
设备上的 `s16_from_hf_rnd_sat<13>` 转换并以 native INT16 导出，再将这份逐 bit
结果写回 model BIN。只有它与旧 custom 三个 graph outputs 逐 bit 一致后，才进入
性能比较。另一条更干净但会改变数值格式的路线，是把 LHS/RHS 同时改为受支持的 Q9，
并重新评估整个 block 相对 FP32 reference 的误差；不能直接假设 Q9 可接受。

## 设备导出 Q13 RHS：采用版本

上述 conversion probe 已经实现并完成真机验证：

```text
qwen_block_custom_qnn/tools/device_q13_probe_model.cpp
qwen_block_custom_qnn/tools/device_q13_probe_input_list.txt
qwen_block_custom_qnn/device_output/device_q13_conversion_probe/Result_0/weight_q13_native.raw
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_device_q13.bin
qwen_block_custom_qnn/generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_device_q13.json
qwen_block_custom_qnn/model_libs/aarch64-android/libqwen2_0_5b_layer0_prefill_seq16_q_proj_device_q13.so
qwen_block_custom_qnn/device_output/q_proj_device_q13_layer0_prefill_seq16/
```

probe graph 的数据路径与原 Qwen q_proj RHS 完全一致：

```text
FP32 q_proj weight
  -> QNN builtin Cast(FP16)
  -> HTP hnnx::s16_from_hf_rnd_sat<13>
  -> APP_READ QNN_DATATYPE_INT_16
  -> qnn-net-run --output_data_type native_only
```

设备输出揭示了 `<13>` 的具体异常行为。对于当前全部 `802816` 个权重：

```text
device_int16 == mathematical_q13 XOR 0x8000
different elements: 802816 / 802816
delta:              exactly -32768 or +32768
device SHA256:      b838c624526c30de48b0a8a3fe3c853b8a570e6d3a47e167302498f3868fc38e
```

因此不能在主机上按普通 Q13 生成采用权重；采用 BIN 必须来自保存的 native probe
输出，或在目标 QAIRT/HTP 组合上重新执行 probe。将该 payload 嵌回模型并删除
`_MatMul_rhs_cast_fp16` 后，三个 graph outputs 相对旧 4-row custom 均逐 bit 相同。

10 次 detailed profiling，丢弃 warm-up 后取中位数：

| 版本 | NetRun | QNN | QNN accelerator | root cycles |
|---|---:|---:|---:|---:|
| QNN builtin | 14,826 us | 14,777 us | 12,141 us | 2,276,136 |
| 4-row custom | 15,884 us | — | 13,058 us | — |
| **device-Q13 custom** | **13,152 us** | **13,125 us** | **10,347 us** | 3,131,207 |

端到端结果：

```text
vs 4-row custom: (15884 - 13152) / 15884 = 17.20% faster
vs QNN builtin:  (14826 - 13152) / 14826 = 11.29% faster
```

这里 custom op 的 `_MatMul` 中位数为 `826368 cycles`，不能直接和 builtin 的
内部 fused subevent cycles 一一比较；采用判断以相同输入、相同 10 次运行策略下的
NetRun/QNN wall time为准。profiling 原始日志、CSV、输出、正确性和摘要均保存在
`device_output/q_proj_device_q13_layer0_prefill_seq16/`。

注意：该采用版本复现的是当前 QAIRT 2.47 / HTP v75 kernel 的实际数值行为，而不是
数学意义上的标准 Q13。升级 QAIRT、Hexagon compiler 或目标 SoC 后，必须重新运行
probe 并核对 payload SHA256 与 graph outputs，不能盲目复用上述二进制。

## device-Q13 `4x128` micro-kernel

device-Q13 消除 RHS 转换后，q_proj 仍需对 14 个 64-column tiles 分别遍历
`K=896`。`4x128` kernel 在每个 reduction step 同时读取两个相邻 RHS vectors，
将同一组四个 LHS splat 复用到 128 个输出列：

```text
4x64:  14 column tiles * 896 reduction iterations
4x128:  7 column tiles * 896 reduction iterations
```

每个输出 accumulator 的 reduction 顺序没有改变。真机验证三个 graph outputs
相对 `4x64 device-Q13` 均逐 bit 相同。

10 次 detailed profiling 的中位数：

| device-Q13 kernel | q_proj cycles | QNN accelerator | QNN | NetRun |
|---|---:|---:|---:|---:|
| 4x64 | 826,368 | 10,347 us | 13,125 us | 13,152 us |
| **4x128** | **582,392** | **10,325 us** | **13,080 us** | **13,105 us** |

```text
q_proj cycles reduction: (826368 - 582392) / 826368 = 29.52%
NetRun improvement:      (13152 - 13105) / 13152 = 0.36%
vs builtin NetRun:       (14826 - 13105) / 14826 = 11.61% faster
```

micro-kernel 本身收益清晰，但端到端只改善 `47 us`，说明完整 block 的主要波动和剩余
时间已经在其他 projection、bias Add、runtime wait 和电源/调度路径。结果保存在：

```text
qwen_block_custom_qnn/device_output/q_proj_device_q13_4x128_layer0_prefill_seq16/
```

没有直接实现 `4x256` 默认路径：4 rows × 4 RHS vectors 需要 16 个
`HVX_VectorPair`，仅 accumulator 就占用 32 个 vector registers，尚未计入 RHS、
LHS 和地址临时值，极易产生 register spill。后续若测试 256-column tile，应使用
分阶段 accumulator 或汇编级寄存器规划，并将 spill 检查列为采用门槛。

### 未采用：reduction 2-way unroll

在 `4x128` reduction loop 中显式展开两个连续 K 元素，保持 accumulator 更新顺序，
三个 graph outputs 仍逐 bit 相同。但 profiling 显示 kernel 明确退化：

| 版本 | q_proj cycles | QNN accelerator | QNN | NetRun |
|---|---:|---:|---:|---:|
| 4x128 | 582,392 | 10,325 us | 13,080 us | 13,105 us |
| 4x128 unroll2 | 672,836 | 10,389 us | 13,067 us | 13,091 us |

q_proj cycles 增加 `15.53%`，说明额外同时存活的 RHS/LHS temporaries 加重了寄存器
压力或降低了编译器调度质量。NetRun 的 `14 us` 表面改善小于整图运行波动，且与
kernel cycles 方向相反，不能作为采用依据。源码已恢复非展开 `4x128`，负实验结果
保存在：

```text
qwen_block_custom_qnn/device_output/q_proj_device_q13_4x128_unroll2_layer0_prefill_seq16/
```

## 精简采用模型中的 q_proj 权重

device-Q13 graph 的 q_proj static tensor 已经改为引用
`onnx__MatMul_227_q13.raw`，原始 `onnx__MatMul_227.raw` 不再被生成的 C++ 引用。
生成器增加 `--drop-source-member`，仅在最终 device-Q13 BIN 中删除该冗余成员；
用于 probe 的原始 BIN 和主机数学 Q13 诊断 BIN不受影响。

体积变化：

```text
original QNN BIN:               59,678,720 bytes
slim device-Q13 BIN:            58,071,040 bytes
old non-slim device-Q13 model:  61,713,616 bytes
slim device-Q13 model lib:      58,502,080 bytes
model lib reduction:             3,211,536 bytes (5.20%)
```

删除量恰好等于 `896 * 896 * sizeof(float)`。瘦身模型能够正常 compose、finalize 和
execute，三个 graph outputs 相对非瘦身 `4x128` 逐 bit 相同。

10 次 detailed profiling 中位数为 `13,077 us`，q_proj 为 `590,453 cycles`。
它与非瘦身版本 `13,105 us` / `582,392 cycles` 的小幅差异属于运行波动；该优化的
采用依据是体积和加载材料减少，而不是宣称 execute kernel 加速。最终结果保存在：

```text
qwen_block_custom_qnn/device_output/q_proj_device_q13_4x128_slim_layer0_prefill_seq16/
```

## KV-cache decode：M=1 专用 custom q_proj 对比

测试 shape 为当前 hidden `[1,1,896]`、past K/V `[1,2,16,64]`、present K/V
`[1,2,17,64]`。当前 token 使用 position 16 的 RoPE，并真实执行 past/current
K/V concat。PyTorch 与 ONNX Runtime 三路输出最大绝对误差为
`9.54e-7 / 9.54e-7 / 3.73e-8`。

为 custom 路径增加了单行 `1x128` HVX micro-kernel，只保留两个
`HVX_VectorPair` accumulator，避免计算不存在的三行。decode 与 prefill 的原始
q_proj 权重 SHA256 完全一致，因此复用了 device conversion probe 的精确 Q13
payload。主要产物：

```text
tools/patch_qwen_decode_q_proj_offline_q13.py
generated/qwen2_0_5b_layer0_decode_past16_q_proj_offline_q13.cpp
generated/qwen2_0_5b_layer0_decode_past16_q_proj_q13.bin
device_output/builtin_layer0_decode_past16/
device_output/custom_q13_layer0_decode_past16/
```

同一设备、burst、detailed、10 次运行去首轮后的中位数：

| decode past16 | root cycles | QNN accelerator | QNN | NetRun |
|---|---:|---:|---:|---:|
| builtin | 1,712,987 | 8,738 us | 10,954 us | **11,007 us** |
| custom Q13 M=1 | 1,867,003 | 9,547 us | 11,010 us | 11,041 us |

```text
builtin fused q_proj:       26,065 cycles
custom q_proj 1x128:        72,112 cycles
custom separate bias Add:   49,859 cycles
custom q_proj + bias:       121,971 cycles
NetRun regression:              34 us (0.31%)
```

custom hidden 相对 FP32 的 cosine 为 `0.999997854`，present key/value 与 builtin
逐 bit 相同。结论是 KV-cache decode 已实现成功，但这版 custom q_proj 是性能负
结果：`M=1` 属于小 GEMV，builtin 融合路径只有约 2.6 万 cycles，而 custom 还承担
QHPI 和独立 bias 节点开销。当前 decode 应保留 builtin；只有融合 bias、减少调度
成本或设计专用 GEMV 权重布局后，才值得重新测试。

## KV-cache context sweep 与边界布局失败

新增 past16/32/64/128 builtin decode graph。第一次 sweep 直接把 ONNX NCHW
`past_key/value.raw` 送入所有 QNN graph，graph 全部能够运行，但 past32/64/128
的 present cache 严重错误。这是一次危险的“执行成功但数据错误”实验。

原因是 converter 的边界 layout rewrite 随 shape 改变：

```text
past16:      API K/V 均为 [1,2,16,64]
past32+:     API K/V 均为 [1,S,64,2]
past32+ out: present K 为 NHWC，present V 为 NCHW
```

增加 `prepare_decode_qnn_io.py` 和 `compare_qnn_output.py
--qnn-decode-layout` 后重新运行。修正后的 past32/64/128 present K/V cosine
均约为 1。错误结果保留在 `device_output/*layout_wrong/`，不得作为性能采用依据。

正式 context sweep 中位数：

| past | root cycles | QNN accelerator | QNN | NetRun |
|---:|---:|---:|---:|---:|
| 16 | 1,712,987 | 8,738 us | 10,954 us | 11,007 us |
| 32 | 2,070,466 | 9,077 us | 11,828 us | 11,852 us |
| 64 | 2,140,213 | 9,193 us | 11,816 us | 11,844 us |
| 128 | 2,180,782 | 9,433 us | 12,007 us | 12,032 us |

## 采用：grouped-GQA 去除显式 K/V Tile

假设：`repeat_interleave(7)` 物化 14-head K/V 会增加临时内存和 attention 工作。
在导出器增加 `--grouped-gqa`，把 Q 表达为 `[B,2,7,1,64]`，让两个 KV heads
通过广播分别服务七个 query heads。QNN C++ 的 Tile 节点由 2 个变为 0。

past128 A/B：

| 版本 | root cycles | QNN accelerator | QNN | NetRun |
|---|---:|---:|---:|---:|
| explicit Tile | 2,180,782 | 9,433 us | 12,007 us | 12,032 us |
| grouped-GQA | 2,054,078 | 9,540 us | 11,806 us | 11,862 us |

```text
QK+Softmax+AV: 283430 -> 109733 cycles，下降 61.28%
root cycles:   下降 5.81%
NetRun:        改善 170 us / 1.41%
三个输出:      相对原 past128 builtin 逐 bit 相同
```

结论：采用。虽然 QNN accelerator wall time单次指标有反向波动，但 QNN、NetRun、
root cycles 和逐 bit 正确性共同支持采用。结果目录：

```text
device_output/builtin_layer0_decode_past128_grouped_gqa/
```

## 采用：current-token KV delta output

假设：完整 present-cache 输出每个 token 都把 `[1,2,S+1,64]` 的 K/V 通过 graph
边界返回；真实生成循环可以由宿主维护持久 cache，只接收当前 token delta。

在导出器增加 `--delta-kv-output`。Attention 内部 concat 保持不变，输出从
完整 K/V 改为 `current_key/value [1,2,1,64]`。past128 的 K/V 输出 payload
从 `132,096 bytes` 降到 `1,024 bytes`，缩小 129 倍。

past128 grouped-GQA A/B：

| 输出策略 | root cycles | QNN accelerator | QNN | NetRun | Output cycles |
|---|---:|---:|---:|---:|---:|
| full present K/V | 2,054,078 | 9,540 us | 11,806 us | 11,862 us | 41,007 |
| current K/V delta | 2,231,170 | 9,493 us | 10,880 us | 10,907 us | 8,798 |

```text
QNN:    改善 926 us / 7.84%
NetRun: 改善 955 us / 8.05%
Output: cycles 下降 78.54%
```

完整-cache graph 的最后一个 K/V token 与 delta graph 的 current K/V 对比，
加上 hidden 输出，三者均逐 bit 相同。

root cycles 与 wall time方向相反，说明不同输出 graph rewrite 下该计数不能单独作为
采用标准。本实验同时具有输出 payload 缩小 129 倍、Output cycles 下降以及 QNN/
NetRun wall time一致改善，因此采用。

限制：这还不是设备原地 cache。内部 concat 和下一步完整 past-cache 输入仍存在；
宿主必须把 delta 写入持久 buffer。后续研究 shared/persistent tensor 或 custom
cache-update op 时应以本版本为基线。

结果目录：

```text
device_output/builtin_layer0_decode_past128_grouped_gqa_delta_kv/
```

## 未采用：FP16 KV-cache graph boundary

假设：past128 K/V 输入从 `131,072 bytes` 减为 `65,536 bytes`，current delta
输出从 `1,024 bytes` 减为 `512 bytes`，可以降低 RPC 和内存带宽。

实现过程暴露两个容易造成静默错误的问题：

1. QAIRT 2.47 converter 把 ONNX FP16 Cast 在转换期解释，生成的 APP tensor 仍为
   FLOAT32；必须用 `patch_qwen_decode_fp16_kv_boundary.py` 恢复 FP16 API，并插入
   attention 侧 FP16→FP32 Cast。
2. qnn-net-run 默认按 FP32 文件读取输入；必须使用
   `--native_input_tensor_names graph:past_key,past_value`，输出使用
   `--output_data_type native_only`。否则一个 FP16 文件会被误报成两个 batch。

真机 past128 A/B：

| KV boundary | root cycles | QNN accelerator | QNN | NetRun | Output cycles |
|---|---:|---:|---:|---:|---:|
| FP32 | 2,231,170 | 9,493 us | 10,880 us | 10,907 us | 8,798 |
| FP16 | 1,890,236 | 9,184 us | 11,741 us | 11,767 us | 3,214 |

FP16 root cycles 下降 `15.28%`、accelerator time 下降 `3.26%`，但 QNN/NetRun
分别退化 `7.91%/7.88%`。边界 Cast、native I/O 或 runtime 调度成本超过了
单层 past128 的带宽收益。

正确性：

```text
FP16 vs FP32 QNN hidden: bit-exact
current key max_abs:     1.53e-5
current value max_abs:   7.45e-9
K/V cosine:              approximately 1
```

结论：不采用，保留 FP32 grouped-GQA delta-KV 为正式 decode 基线。FP16 结果：

```text
device_output/builtin_layer0_decode_past128_grouped_gqa_delta_kv_fp16/
```

## 未采用：shared buffer 与 input tensor-set cache

QAIRT 2.47 的 qnn-net-run 支持 `--shared_buffer`，SDK 也提供基于 `memRegister`
的 HTP RPC shared-buffer sample。但该选项只注册 graph I/O 内存，不会把当前
K/V delta 自动反馈为下一次 execute 的 past cache。

第一次 10 次实验中 shared buffer 明显退化。为排除时序波动，在同一设备连续做
30 次普通模式和 30 次 shared-buffer 模式，并测试
`--max_input_cache_tensor_sets 1`：

| mode | root cycles | QNN accelerator | QNN | NetRun |
|---|---:|---:|---:|---:|
| normal | 2,350,585 | 9,654 us | 11,736 us | 11,758 us |
| shared buffer | 2,331,778 | 9,713 us | 12,028 us | 12,052 us |
| input tensor cache | 2,295,054 | 9,550 us | 12,145 us | 12,175 us |

```text
shared buffer regression:      294 us / 2.50%
input tensor cache regression: 417 us / 3.55%
```

accelerator cycles 持平或略有改善，但 QNN/NetRun 退化，推测注册内存访问、同步或
宿主 bookkeeping 成本超过了 past128 单层输入复制收益。两者均不采用。

机器可读汇总：

```text
device_output/decode_io_mode_summary.csv
```

## 2026-07-26：past128 decode attention 融合（未采用）

目标：在正式的 FP32 grouped-GQA delta-KV graph 上，把 QK MatMul、scale/mask、
Softmax 和 AV MatMul 合成一个固定 shape HTP custom op。

实现：

- 新增 `Qwen2DecodeAttentionPast128Fp32`，输入 Q/K/V 分别为
  `[1,14,1,64]`、`[1,129,64,2]`、`[1,2,129,64]`。
- 按 14 个 query head 自切片到 4 个 QHPI worker。
- online softmax 不物化 `[14,129]` score/probability。
- `patch_qwen_decode_fused_attention.py` 显式删除被融合节点并让 custom op 直接
  输出 `[1,14,1,64]`。

三轮端侧结果：

| 版本 | fused cycles | NetRun | hidden max_abs | cosine |
|---|---:|---:|---:|---:|
| scalar + `std::exp` | ~14.3M | 22423 us | 1.953e-3 | 0.999999881 |
| fast-exp polynomial | ~17.9M | 24158 us | 1.953e-3 | 0.999999881 |
| fast-exp + HVX value | ~11.25M | 19432 us | 0.222656 | 0.999941289 |
| adopted builtin delta-KV | QK+Softmax+AV ~101k | 10907 us | reference | reference |

结论：失败原因不是 fusion boundary，而是 custom 实现中的标量 FP32 dot、softmax
和累加远慢于 HTP builtin vector kernel。HVX value 版本虽改善 cycles，却引入不可
接受的数值偏差。三版均不采用，源码恢复到数值可靠、便于后续研究的 scalar
reference；正式 decode 继续使用 builtin attention。

保存结果：

```text
device_output/fused_attention_layer0_decode_past128/
device_output/fused_attention_fast_exp_layer0_decode_past128/
device_output/fused_attention_hvx_value_layer0_decode_past128/
generated/qwen2_0_5b_layer0_decode_past128_fused_attention.cpp
tools/patch_qwen_decode_fused_attention.py
```

## 2026-07-26：persistent past128 KV-cache runner（采用为 runner 基线）

目标：不再依赖 qnn-net-run 每轮 tensor/input/output bookkeeping，在同一进程中只
创建一次 QNN backend/context/graph 和 tensor，并把 current K/V 写回固定窗口 cache。

代码：

```text
qnn_custom_ops/tools/qnn_sample_app_profile/src/QnnSampleApp.cpp
qnn_custom_ops/tools/qnn_sample_app_profile/src/QnnSampleApp.hpp
qnn_custom_ops/tools/qnn_sample_app_profile/src/main.cpp
tools/run_persistent_kv_runner_repro.sh
tools/summarize_persistent_runner.py
```

新增 CLI：`--persistent_decode_past128`。每步把 `[1,2,1,64]` current K/V 转成
head-interleaved `[64,2]`，past NHWC buffer 左移一个 token 后追加，下一步复用
完全相同的输入 tensor 地址。

正确性：100 步运行的 step 0 与正式 delta-KV baseline 比较，hidden/current-K/
current-V 三路逐 bit 相同。

100 步、去首步统计：

| timer | median | mean | P90 | P95 |
|---|---:|---:|---:|---:|
| graphExecute | 4988 us | 4982.68 us | 5609 us | 6155 us |
| CPU cache update | 34 us | 64.65 us | 40 us | 51 us |
| step total | 5037 us | 5047.47 us | 5692 us | 6404 us |

关闭 profiling 的相同 100 次进程 wall time：

```text
qnn-net-run:       1.573442 s
persistent runner: 1.316746 s
改善:              0.256696 s / 16.32%
```

采用范围：作为下一阶段 runner 基线。不能宣称 HTP kernel 本身加速，因为应用内
timer 不含初始化/文件 I/O，且 cache 更新仍在 CPU 普通内存。当前仍是单层固定
past128 sliding window，hidden input 不随真实 token 更新，graph 内仍有 Concat。

结果：

```text
device_output/persistent_runner_layer0_decode_past128_final100/
device_output/persistent_runner_layer0_decode_past128_repeat100/
```

## 2026-07-26：persistent shared buffer（采用为可选模式）

在 persistent runner 的一次性 tensor allocation 上加入：

```text
rpcmem_alloc -> rpcmem_to_fd -> QNN_MEM_TYPE_ION
-> context memRegister -> QNN_TENSORMEMTYPE_MEMHANDLE
```

新增 CLI `--persistent_shared_buffer`。host pointer 通过 tensor-ID map 保存，CPU
直接更新同一 rpcmem cache；退出时 `memDeRegister` 后 `rpcmem_free`。

正确性：3 步 smoke test 的 step 0 三路输出与 builtin delta-KV baseline 逐 bit
相同。

为了控制时序漂移，执行 normal/shared/normal 和 shared/normal/shared 两组夹心
实验，每次 100 步、去首步：

| order | mode | graph median | update median | total median |
|---:|---|---:|---:|---:|
| 1 | normal | 4938 | 36 | 4974 |
| 2 | shared | 4891 | 39 | 4934 |
| 3 | normal | 4920 | 37 | 4968 |
| 4 | shared | 4896 | 40 | 4936 |
| 5 | normal | 4954 | 36 | 4991 |
| 6 | shared | 4919 | 39 | 4955 |

mean of medians：

```text
normal graph/total: 4937.33 / 4977.67 us
shared graph/total: 4902.00 / 4941.67 us
shared total 改善:  36.00 us / 0.72%
```

shared cache 的 CPU 更新约慢 3 us，但 graphExecute 快约 35 us，获得小幅净收益。
六次方向一致，因此采用为长序列 runner 的可选推荐模式；收益不足 1%，普通 buffer
仍保留为兼容 fallback。它不是 graph 内原地 cache，也没有移除 Concat。

结果：

```text
device_output/persistent_shared_ab100/
device_output/persistent_shared_ab100/summary.csv
```

## 2026-07-26：NCHW value-cache boundary（未采用）

动机：正式 profile 中 K/V Concat 合计只有 `5472 cycles`（root 约 `0.25%`），
不值得实现高风险 destructive alias；`past_value_nchw` Transpose 则约
`25203 cycles`，所以先尝试把 past-value API 直接改为 `[1,2,128,64]` NCHW。

改动：

```text
tools/patch_qwen_decode_nchw_value_cache.py
tools/device_input_list_layer0_decode_past128_nchw_value_cache.txt
generated/qwen2_0_5b_layer0_decode_past128_nchw_value_cache.cpp
```

runner 根据 past-value shape 选择 NHWC interleaved 更新或按两个 NCHW head 分别
滑动。smoke test 三路输出逐 bit 相同。

两次 shared-persistent 交替 A/B 的中位数均值：

```text
old NHWC: graph 4900.0 us, update 39.0 us, total 4939.5 us
NCHW:     graph 4932.0 us, update 40.5 us, total 4988.5 us
退化:                                         49.0 us / 0.99%
```

同窗口 detailed profile：

```text
past_value_nchw: 26662 -> 0 cycles
V Concat:         3592 -> 16323 cycles
root:          2246194 -> 2343297 cycles
QNN:             11112 -> 11648 us
NetRun:           11133 -> 11685 us
```

结论：节点删除成功，但 boundary layout 改变导致下游 Concat/graph rewrite 选择更差，
端到端负优化，不采用。保留普通 NHWC value boundary。

结果：

```text
device_output/nchw_value_cache_ab100/
device_output/nchw_value_cache_profile/
device_output/old_value_cache_profile_same_window/
```
## 独立 Qwen GQA FlashAttention OpPackage（当前未采用）

新增独立目录：

```text
qnn_custom_ops/qwen_flash_attention_hvx/
```

与旧固定 past128 fused experiment 不同，新 op 支持动态 KV/Q length、Qwen
`14Q/2KV` grouped-query 映射、32-token blockwise online softmax 和 causal
query boundary，并直接接受正式 graph 的 K-NHWC/V-NCHW 边界。

主机 NumPy blockwise reference 对物化 attention：

```text
max_abs_error=2.60770321e-08
cosine=1.00000012
```

第一次 rank-5 QHPI `Flat4` graph 在 finalize 返回 `1002`；将连续等价的
`[1,2,7,1,64]` 改成 `[1,14,1,64]` rank 4 后成功执行。

真机 past128、10 次 detailed profiling 去 warmup 中位数：

```text
root_cycles=19,508,660
qnn_accel_us=21,213
qnn_us=23,283
netrun_us=23,338
_QwenGqaFlashAttention≈17.5M cycles
```

与正式 grouped-GQA delta-KV 输出比较：

```text
hidden max_abs=0.0625
hidden mean_abs=0.00815392
hidden cosine=0.999947424
current K/V=逐 bit相同
```

结论：graph 中原 QK/Softmax/AV 已被真正删除，但标量 FP32 kernel 仍比正式
`10907 us` baseline 慢 `113.97%`，且误差未达门槛，因此不采用。后续只在完成
HVX QK dot/reduction 与准确的 HVX AV 累加后复测。

## 2026-07-26：独立 FlashAttention 的 HVX IEEE-FP AV（未采用）

反汇编确认独立动态 KV kernel 的 QK/AV 均为标量 `sfmpy`。本轮将 64 维
value accumulator 的缩放、加权累加和最终归一化改为两个 128-byte HVX
FP32 vector。v75 首次构建失败：

```text
Attempting to emit V6_vmpy_sf_sf instruction but
Feature_UseHVXIEEEFP predicate(s) are not met
```

在 OpPackage v75 flags 增加 `-mhvx-ieee-fp` 后，ARM/HTP package 构建成功，
真实 past128 graph 完成 10 次执行。

去 warmup 中位数：

```text
fused cycles: 17,618,383 -> 11,371,942 (-35.45%)
root cycles:  19,508,660 -> 13,165,785 (-32.51%)
NetRun:           23,338 ->     19,234 us (-17.59%)
```

正确性：

```text
hidden max_abs=0.22265625
hidden mean_abs=0.00644305
hidden cosine=0.999941244
current K/V=逐 bit相同
```

v75 此路径使用分离的 vector multiply/add，不能保持原标量 fused multiply-add
的逐 token 舍入行为。性能改善真实存在，但仍比正式 builtin `10907 us` 慢
`76.34%`，最大误差也未通过门槛，因此不采用为正式 graph。源码和结果保留在：

```text
qnn_custom_ops/qwen_flash_attention_hvx/
qnn_custom_ops/qwen_flash_attention_hvx/device_output/past128_hvx_av/
```

下一步先改 K cache 为 head-contiguous layout，再做 HVX QK dot/reduction；
不在当前 head-interleaved K 上引入高成本 gather。

## 2026-07-26：head-contiguous K + HVX QK（实验保留）

将 past K boundary 从 `[1,128,64,2]` 改为 `[1,2,128,64]`，删除 current K
到 NHWC 的 Transpose，并沿 token axis 2 拼接。QK 的 64 次 FP32 multiply 使用
两个 128-byte HVX vector；v75 缺少直接 FP32 horizontal reduction，乘积写入
对齐局部数组后标量求和。AV 的两个 accumulator vector 改为跨 token 寄存器驻留。

保存文件：

```text
qnn_custom_ops/qwen_flash_attention_hvx/tools/
  patch_qwen_decode_head_contiguous.py
qnn_custom_ops/qwen_flash_attention_hvx/scripts/
  build_qwen_graph_head_contiguous.sh
generated/qwen2_0_5b_layer0_decode_past128_flash_attention_head_contiguous.cpp
tools/device_input_list_layer0_decode_past128_flash_head_contiguous.txt
```

10 次 detailed profiling、去 warmup 中位数：

```text
fused cycles: 11,371,942 ->   891,176 (-92.16%)
root cycles:  13,165,785 -> 2,768,200
QNN accel:        16,672 ->     8,910 us
NetRun:           19,234 ->    11,403 us (-40.72%)
```

与正式 builtin：

```text
QNN accelerator: 8,910 vs 9,493 us，custom 快 6.14%
NetRun:          11,403 vs 10,907 us，custom 慢 4.55%
```

输出与旧 HVX AV 版本逐 bit 相同：

```text
hidden max_abs=0.22265625
hidden mean_abs=0.00644305
hidden cosine=0.999941244
current K/V=逐 bit相同
```

结论：head-contiguous K 是决定性优化，纯 accelerator 已超过 builtin；但端到端
仍慢 `496 us`，AV 精度也未过门槛，所以只保留为实验 graph，正式 decode 仍使用
builtin。

QFloat32 AV 对照为 `11504 us`、max_abs `0.198242188`、mean_abs
`0.00703363`，性能和平均误差退化，不采用。源码以
`QWEN_FLASH_AV_QFLOAT` 开关保留，结果分别位于：

```text
qnn_custom_ops/qwen_flash_attention_hvx/device_output/
  past128_head_contiguous_qk/
  past128_head_contiguous_qfloat/
  past128_head_contiguous_reg_av/
```

部署时还确认 QHPI `--op_packages` 必须使用
`path:InterfaceProvider:CPU/HTP` 完整语法；省略 provider 会在 Context Creation
阶段触发 package registration `4005/4007`。
