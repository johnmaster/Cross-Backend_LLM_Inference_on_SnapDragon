# 定制 QNN SampleApp

本目录基于 QAIRT 2.47 的 `qnn-sample-app` 源码，提供本仓库性能分析和 Qwen
decode 实验需要的扩展。它不是自定义 Op Package，也不参与算子计算；它负责加载
backend、model library、输入数据和 Op Package，然后调用 QNN graph。

上游的简要说明保留在 `README.txt`，本文件记录仓库版本的实际用途和命令。

## 仓库改动

### Profiling 事件输出

原始 SampleApp 使用 DEBUG 日志打印 backend profiling event，而设备上的 release
构建不会显示这些信息。本版本将事件输出为带固定前缀的 INFO 日志：

```text
QNN_PROFILE_EVENTS ...
QNN_PROFILE_EVENT ...
```

因此使用 `--profiling_level detailed` 时，可以直接从设备日志中取得 graph、Op
以及 sub-event 的时间或 cycle 数据。

### 持久化 past128 decode

参数 `--persistent_decode_past128` 会让 graph、context 和 tensor 只创建一次，
然后连续调用 `graphExecute()`。每一步执行后，程序将 `current_key/current_value`
追加到 `past_key/past_value` 的滑动窗口，并生成：

```text
<output_dir>/persistent_decode_timings.csv
```

该模式是当前 Qwen 实验的固定执行路径，不是通用 KV cache runner。Graph 必须满足：

- 只有一个 graph 和一组输入；
- 输入 tensor 名为 `past_key`、`past_value`；
- 输出 tensor 名为 `current_key`、`current_value`；
- KV head 数为 2，head dimension 为 64，窗口长度为 128；
- cache 和 delta 使用 FP32；
- key cache 使用 `[1,128,64,2]` 的 NHWC 物理布局；
- value cache 支持 `[1,128,64,2]`，也识别 `[1,2,128,64]`。

不满足这些约束时不能使用该参数。

### Shared Buffer

参数 `--persistent_shared_buffer` 使用 `rpcmem_alloc()` 分配 tensor buffer，并通过
QNN `memRegister()` 注册为 `QNN_TENSORMEMTYPE_MEMHANDLE`，用于减少持久化执行中的
宿主与 DSP 数据搬运。该参数应和 `--persistent_decode_past128` 一起使用，并要求
设备提供可用的 `libcdsprpc.so` 和 backend memory registration。

## 构建

在仓库根目录执行：

```bash
QAIRT_ROOT=/home/lingbok/Qualcomm/qairt/2.47.0.260601
ANDROID_NDK_ROOT=/home/lingbok/android/android-ndk-r28
APP_DIR="$PWD/qnn_custom_ops/tools/qnn_sample_app_profile"

"$ANDROID_NDK_ROOT/ndk-build" \
  APP_ALLOW_MISSING_DEPS=true \
  APP_ABI=arm64-v8a \
  NDK_PROJECT_PATH="$APP_DIR" \
  NDK_APPLICATION_MK="$APP_DIR/make/Application.mk" \
  APP_BUILD_SCRIPT="$APP_DIR/make/Android.mk" \
  QNN_SDK_ROOT="$QAIRT_ROOT"
```

主要产物：

```text
libs/arm64-v8a/qnn-sample-app
```

`obj/` 是编译中间目录，`bin/` 和 `libs/` 中的可执行文件是生成产物。

## 部署

建议使用独立名称，避免覆盖 QAIRT 自带的 `qnn-sample-app`：

```bash
adb shell 'mkdir -p /data/local/tmp/qnn/bin'
adb push \
  qnn_custom_ops/tools/qnn_sample_app_profile/libs/arm64-v8a/qnn-sample-app \
  /data/local/tmp/qnn/bin/qnn-sample-app-profile
adb shell 'chmod 755 /data/local/tmp/qnn/bin/qnn-sample-app-profile'
```

## 普通 Graph 与 Profiling

普通模式兼容 QNN SampleApp 的 model、input list 和 Op Package 参数。下面的路径
需要替换为实际实验目录：

```bash
adb shell 'cd /data/local/tmp/qnn && \
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH" && \
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp" && \
./bin/qnn-sample-app-profile \
  --backend lib/libQnnHtp.so \
  --model <model-library.so> \
  --op_packages <package.so>:<provider>:HTP \
  --input_list <input-list.txt> \
  --output_dir <output-directory> \
  --profiling_level detailed \
  --num_inferences 20 \
  --log_level info'
```

内置 QNN 算子不需要 `--op_packages`。分析性能时通常排除第一轮，再对剩余轮次
取中位数，以降低初始化和 DSP 唤醒的影响。

## 持久化 Qwen Decode

仓库已经提供完整的构建、部署、执行和结果检查脚本：

```bash
STEPS=100 SHARED_BUFFER=1 \
bash qwen_block_custom_qnn/tools/run_persistent_kv_runner_repro.sh
```

其中：

- `STEPS` 控制连续 decode 次数；
- `SHARED_BUFFER=1` 启用 rpcmem shared buffer；
- `SHARED_BUFFER=0` 使用普通 raw client buffer；
- 首轮输出保存到 `Result_0/`；
- 每步耗时保存到 `persistent_decode_timings.csv`。

直接调用可执行文件时，核心参数为：

```text
--num_inferences <steps>
--persistent_decode_past128
--persistent_shared_buffer      # 可选
```

## 相关实现

```text
src/main.cpp                    命令行参数解析
src/QnnSampleApp.cpp            profiling 与持久化执行循环
src/QnnSampleApp.hpp            扩展状态和函数声明
src/Utils/IOTensor.cpp          raw buffer、rpcmem 和 memRegister
qnn-learning.md                 QNN SampleApp API 调用时序
```
