# QNN 学习路线

本文记录接下来学习 Qualcomm QNN 的主线。当前重点不是继续折腾 LLM Genie，而是先把 QNN 的基础执行链路学清楚，并最终写出自己的最小 runner。

## 当前基础

已经完成的前置工作：

- QAIRT / QNN SDK 已安装到本机。
- OnePlus 12 手机端已部署 QNN runtime 文件。
- `qnn-platform-validator` 已验证 GPU 和 DSP / HTP 后端可用。
- AI Hub 导出的 MobileNet V2 `qnn_context_binary` 已能在手机上运行。
- `qnn-net-run` 已能用 HTP backend 跑通一次推理。
- `qnn-throughput-net-run` 已能拿到 HTP throughput，但退出时曾出现 segmentation fault。
- Genie BGE embedding 已跑通，说明 Genie 基础环境也可用。

因此下一步应该学习 QNN runner，而不是继续先攻 LLM。

## 为什么先从 `qnn-net-run` 入手

`qnn-net-run` 是官方工具，适合用来理解 QNN 推理的基本流程：

```text
加载 backend
创建或恢复 context
读取 graph
准备 input tensor
执行 graph
读取 output tensor
保存 output raw
释放资源
```

当前已经跑通的命令：

```bash
cd /data/local/tmp/qnn

export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp

./bin/qnn-net-run \
  --backend lib/libQnnHtp.so \
  --retrieve_context mobilenet_v2/mobilenet_v2.bin \
  --input_list mobilenet_v2/input/input_list.txt \
  --output_dir mobilenet_v2/output_htp
```

这条命令的含义：

| 参数 | 含义 |
| --- | --- |
| `--backend lib/libQnnHtp.so` | 使用 HTP backend |
| `--retrieve_context mobilenet_v2/mobilenet_v2.bin` | 从 context binary 恢复 QNN context |
| `--input_list mobilenet_v2/input/input_list.txt` | 指定输入 raw 文件列表 |
| `--output_dir mobilenet_v2/output_htp` | 保存输出 tensor |

当前阶段只研究 `--retrieve_context` 模式。也就是先学习如何加载已经编译好的 context binary，不急着研究模型转换、离线 prepare 或自定义 op。

## 推荐源码入口

SDK 中不一定提供完整 `qnn-net-run` 工具源码。更适合学习的是官方 SampleApp：

```text
$QAIRT_SDK_ROOT/examples/QNN/SampleApp/SampleAppMultiGraph/
```

重点文件：

```text
src/main.cpp
src/QnnSampleApp.cpp
src/Utils/IOTensor.cpp
src/Utils/DynamicLoadUtil.cpp
src/Utils/DataUtil.cpp
src/WrapperUtils/QnnWrapperUtils.cpp
```

建议先看：

```bash
sed -n '1,220p' "$QAIRT_SDK_ROOT/examples/QNN/SampleApp/SampleAppMultiGraph/src/main.cpp"
sed -n '1,260p' "$QAIRT_SDK_ROOT/examples/QNN/SampleApp/SampleAppMultiGraph/src/QnnSampleApp.cpp"
```

先不要从 `SampleAppSharedBuffer` 开始。SharedBuffer 涉及共享内存和零拷贝，适合第二阶段学习。

## 学习顺序

### 1. 跑熟 `qnn-net-run`

目标：

- 知道 backend、context binary、input list、output dir 分别是什么。
- 知道输出目录里 `Result_0/class_logits.raw` 是模型输出。
- 能独立复现 MobileNet V2 HTP 单次推理。

需要特别记住：当前 `mobilenet_v2.bin` 是 HTP-specific context binary。它不能直接拿去 CPU/GPU backend 跑。之前 CPU/GPU throughput 失败不是后端不可用，而是 context binary 和 backend 不匹配。

### 2. 阅读 `SampleAppMultiGraph`

按调用链阅读：

```text
main.cpp
  -> 参数解析
  -> 创建 QnnSampleApp
  -> 初始化 backend
  -> 创建或恢复 context
  -> 读取 graph 信息
  -> 准备输入输出 tensor
  -> 执行 graph
  -> 保存输出
  -> 释放资源
```

阅读时重点关注这些问题：

- backend `.so` 是如何动态加载的？
- QNN API function pointers 是怎么取出来的？
- context binary 是如何读入并恢复的？
- graph 名称和 tensor 信息是如何拿到的？
- input raw 文件如何映射到 input tensor？
- output tensor 的大小、类型、名字如何确定？
- `graphExecute` 前后做了哪些准备和清理？

### 3. 写最小 HTP runner

第一版 runner 不追求通用，只支持当前已经跑通的 MobileNet V2：

```text
backend: libQnnHtp.so
context: mobilenet_v2.bin
input: image_tensor.raw
output: class_logits.raw
thread: single thread
```

第一版目标：

- 能加载 `libQnnHtp.so`。
- 能恢复 `mobilenet_v2.bin`。
- 能读取输入 raw。
- 能执行一次 graph。
- 能保存输出 raw。
- 输出文件和 `qnn-net-run` 结果一致或接近。

暂时不要做：

- 多线程 throughput。
- CPU/GPU backend 自动切换。
- shared buffer。
- 自定义 op。
- LLM / Genie。

### 4. 加 benchmark

单次推理稳定后，再加简单 benchmark：

```text
warmup N 次
repeat N 次
记录每次耗时
计算 avg / min / max
写入 log
```

这一步可以对齐之前 `qnn-throughput-net-run` 的结果。例如 HTP MobileNet V2 曾测到：

```text
graphExecute repeat: 4480
avg: 2229 us
min: 1905 us
max: 3315 us
throughput: about 448 inf/s
```

注意：`inf/s` 是每秒推理次数，不是 LLM 的 token/s。

### 5. 再看 SharedBuffer

当普通 input/output buffer 版本跑通后，再学习：

```text
SampleAppSharedBuffer
```

SharedBuffer 的重点是减少数据拷贝。它更接近高性能部署，但不适合作为第一阶段入口。

## 当前已知坑

### `ADSP_LIBRARY_PATH`

HTP / DSP 后端需要正确设置：

```bash
export ADSP_LIBRARY_PATH=$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp
```

这里在 Android shell 中使用分号 `;` 分隔。

如果设置不正确，可能出现：

```text
Please use testsig if using unsigned images.
Also make sure ADSP_LIBRARY_PATH points to directory containing skels.
```

### context binary 和 backend 绑定

AI Hub 导出的 `qnn_context_binary` 通常和 backend / chipset / runtime 相关。

例如当前 `mobilenet_v2.bin` 是 HTP context binary：

- HTP backend 可以加载。
- CPU backend 不能直接加载。
- GPU backend 不能直接加载。

如果要测 CPU/GPU，需要分别导出对应 backend 兼容的 QNN artifact。

### throughput 工具退出 segfault

`qnn-throughput-net-run` 曾出现：

```text
Segmentation fault
return_code: 139
```

但在崩溃前已经输出了有效的 graphExecute 统计。logcat 显示更像是线程退出和资源清理阶段的问题，而不是 graph 执行失败。

自己写 runner 时第一版应保持：

- 单线程。
- 不提前 `dlclose` backend。
- graph/context/backend 按顺序释放。
- 先确保一次推理稳定，再做循环和多线程。

## 下一步行动

建议下一步从本机阅读源码开始：

```bash
export QAIRT_SDK_ROOT=$HOME/Qualcomm/qairt/2.47.0.260601
export QNN_SDK_ROOT=$QAIRT_SDK_ROOT

find "$QAIRT_SDK_ROOT/examples/QNN/SampleApp/SampleAppMultiGraph/src" -maxdepth 3 -type f | sort
```

然后按顺序阅读：

```bash
sed -n '1,220p' "$QAIRT_SDK_ROOT/examples/QNN/SampleApp/SampleAppMultiGraph/src/main.cpp"
sed -n '1,260p' "$QAIRT_SDK_ROOT/examples/QNN/SampleApp/SampleAppMultiGraph/src/QnnSampleApp.cpp"
```

学习目标不是一下子看懂所有 QNN API，而是先把这条线吃透：

```text
context binary -> graph execute -> output raw
```

等这条线清楚后，再开始写自己的 `mobilenet_v2_runner`。
