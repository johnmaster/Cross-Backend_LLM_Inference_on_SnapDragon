# QNN SDK 本地安装与验证记录

本文记录在 Linux 主机和 OnePlus 12 手机上安装 Qualcomm QAIRT / QNN SDK 的过程，以及当前遇到的问题和解决方式。

## 1. 当前目标

当前阶段的目标不是直接跑大语言模型，而是先确认手机本地 QNN 运行环境是否可用：

1. 在 Linux 主机上解压 QAIRT / QNN SDK。
2. 把 Android 端工具和运行库推到手机。
3. 用 `qnn-platform-validator` 验证 GPU 后端。
4. 用 `qnn-platform-validator` 验证 DSP / HTP 后端。
5. 为后续 `qnn-net-run`、`qnn-throughput-net-run`、Genie LLM 部署做准备。

当前设备：

```text
Phone   : OnePlus 12
SoC     : Snapdragon 8 Gen 3
GPU     : Adreno 750
DSP/HTP : Hexagon V75
Android : Android 15
```

## 2. SDK 文件

已经下载的 SDK 压缩包：

```bash
/home/lingbok/Downloads/v2.47.0.260601.zip
```

解压后建议设置环境变量：

```bash
export QAIRT_SDK_ROOT=$HOME/Qualcomm/qairt/2.47.0.260601
```

常用目录：

```text
$QAIRT_SDK_ROOT/bin/aarch64-android
$QAIRT_SDK_ROOT/lib/aarch64-android
$QAIRT_SDK_ROOT/lib/hexagon-v75/unsigned
```

其中：

- `bin/aarch64-android`：Android 端可执行工具，例如 `qnn-platform-validator`、`qnn-net-run`。
- `lib/aarch64-android`：Android 端 QNN runtime library。
- `lib/hexagon-v75/unsigned`：Hexagon V75 DSP/HTP 侧 skel library。

## 3. 推送到手机

手机端建议统一放在：

```bash
/data/local/tmp/qnn
```

Linux 主机执行：

```bash
export QAIRT_SDK_ROOT=$HOME/Qualcomm/qairt/2.47.0.260601
export QNN_PHONE_DIR=/data/local/tmp/qnn

adb shell "mkdir -p $QNN_PHONE_DIR/bin $QNN_PHONE_DIR/lib $QNN_PHONE_DIR/dsp"
adb push $QAIRT_SDK_ROOT/bin/aarch64-android/qnn-platform-validator $QNN_PHONE_DIR/bin/
adb push $QAIRT_SDK_ROOT/bin/aarch64-android/qnn-net-run $QNN_PHONE_DIR/bin/
adb push $QAIRT_SDK_ROOT/bin/aarch64-android/qnn-throughput-net-run $QNN_PHONE_DIR/bin/
adb push $QAIRT_SDK_ROOT/lib/aarch64-android/*.so $QNN_PHONE_DIR/lib/
adb push $QAIRT_SDK_ROOT/lib/hexagon-v75/unsigned/* $QNN_PHONE_DIR/dsp/
adb shell "chmod +x $QNN_PHONE_DIR/bin/*"
```

## 4. 手机端环境变量

进入手机 shell：

```bash
adb shell
```

然后：

```sh
cd /data/local/tmp/qnn
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
```

这里最容易踩坑的是 `ADSP_LIBRARY_PATH`。

`LD_LIBRARY_PATH` 使用冒号 `:` 分隔：

```sh
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
```

但 `ADSP_LIBRARY_PATH` 推荐使用分号 `;` 分隔：

```sh
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
```

如果 DSP skel 路径没有正确被 FastRPC 找到，DSP 测试可能会失败。

## 5. qnn-platform-validator 参数

直接运行：

```sh
./bin/qnn-platform-validator
```

会报：

```text
ERROR: Missing mandatory argument --backend
```

需要指定 backend：

```sh
./bin/qnn-platform-validator --backend gpu --coreVersion
./bin/qnn-platform-validator --backend dsp --coreVersion
./bin/qnn-platform-validator --backend all --coreVersion
```

常用参数：

```text
--backend gpu|dsp|all
--libVersion
--coreVersion
--testBackend
--debug
```

## 6. GPU 后端验证

执行：

```sh
./bin/qnn-platform-validator --backend all --libVersion
./bin/qnn-platform-validator --backend all --coreVersion
./bin/qnn-platform-validator --backend all --testBackend
```

已经确认 GPU 后端可用：

```text
Backend GPU Prerequisites: Present.
Library Version of the backend GPU: OpenCL 3.0 Adreno(TM) 750
Core Version of the backend GPU: Adreno(TM) 750
Unit Test on the backend GPU: Passed.
QNN is supported for backend GPU on the device.
```

说明：

- QNN GPU 后端能找到 Adreno 750。
- OpenCL runtime 可用。
- 简单 GPU unit test 已通过。

## 7. DSP / HTP 后端验证

查询 DSP core：

```sh
./bin/qnn-platform-validator --backend dsp --coreVersion
```

结果：

```text
Backend DSP Prerequisites: Present.
Core Version of the backend DSP: Hexagon Architecture V75
```

说明 OnePlus 12 上的 DSP / HTP 架构是 Hexagon V75。

DSP unit test：

```sh
./bin/qnn-platform-validator --backend dsp --testBackend
```

最终成功结果：

```text
PF_VALIDATOR: DEBUG: Starting calculator test
PF_VALIDATOR: DEBUG: Loading sample stub: libQnnHtpV75CalculatorStub.so
PF_VALIDATOR: DEBUG: Successfully loaded DSP library - 'libQnnHtpV75CalculatorStub.so'.  Setting up pointers.
PF_VALIDATOR: DEBUG: Success in executing the sum function
Unit Test on the backend DSP: Passed.
QNN is supported for backend DSP on the device.
```

最终结论：

```text
Backend = DSP
{
  Backend Hardware  : Supported
  Backend Libraries : Found
  Core Version      : Hexagon Architecture V75
  Unit Test         : Passed
}
```

## 8. DSP 曾遇到的问题

一开始 DSP `--coreVersion` 可以通过，但 `--testBackend` 失败：

```text
Unable to destroy the handle
PF_VALIDATOR: ERROR: -6 . Error while executing the sum function.
PF_VALIDATOR: ERROR: Please use testsig if using unsigned images.
PF_VALIDATOR: ERROR: Also make sure ADSP_LIBRARY_PATH points to directory containing skels.
Unit Test on the backend DSP: Failed.
QNN is NOT supported for backend DSP on the device.
```

这个报错容易误判成手机不支持 DSP。实际情况是：

- `--coreVersion` 能看到 `Hexagon Architecture V75`，说明 DSP 硬件和基础库存在。
- `--testBackend` 会加载 DSP 侧 calculator skel。
- 如果 `ADSP_LIBRARY_PATH` 没有正确指向 skel 文件目录，就会报 skel / unsigned image / testsig 相关错误。

修正环境变量后，DSP unit test 已经通过。

关键修正：

```sh
export ADSP_LIBRARY_PATH="$PWD/dsp;$PWD/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
```

建议同时确认 skel 文件在手机上存在：

```sh
ls -l /data/local/tmp/qnn/dsp/libCalculator_skel.so
ls -l /data/local/tmp/qnn/dsp/libQnnHtpV75Skel.so
ls -l /data/local/tmp/qnn/lib/libQnnHtpV75CalculatorStub.so
```

## 9. 当前状态

截至目前，本地 QNN SDK 基础验证结果：

| Backend | Hardware | Libraries | Core / Runtime | Unit Test | 结论 |
|---|---:|---:|---|---:|---|
| GPU | Supported | Found | Adreno 750 / OpenCL 3.0 | Passed | 可用 |
| DSP / HTP | Supported | Found | Hexagon V75 | Passed | 可用 |

这说明 OnePlus 12 本地已经具备运行 QNN GPU 和 QNN DSP/HTP backend 的基础条件。

## 10. 下一步

建议下一步不要直接上 Qwen / LLM，而是先用 MobileNet V2 这样的小模型打通本地 QNN 执行链路：

1. 从 AI Hub 导出 QNN context binary 或 QNN DLC。
2. 把模型产物、输入数据、QNN runtime 推到手机。
3. 用 `qnn-net-run` 单次运行。
4. 用 `qnn-throughput-net-run` 做 benchmark。
5. 分别测试 GPU backend 和 DSP / HTP backend。
6. 确认小模型本地 QNN 跑通后，再进入 Genie / LLM 路线。

当前已经解决的是最关键的第一关：手机本地 QNN backend validation。

## 11. MobileNet V2 throughput 脚本

已经添加手机端脚本：

```bash
scripts/run_qnn_throughput_all.sh
```

这个脚本会在手机 `/data/local/tmp/qnn/mobilenet_v2` 下自动生成三份配置：

```text
throughput_cpu.json
throughput_gpu.json
throughput_htp.json
```

然后依次运行：

```text
QNN CPU backend
QNN GPU backend
QNN HTP / DSP backend
```

默认输入和模型路径：

```text
QNN_DIR     = /data/local/tmp/qnn
MODEL_BIN   = /data/local/tmp/qnn/mobilenet_v2/mobilenet_v2.bin
INPUT_LIST  = /data/local/tmp/qnn/mobilenet_v2/input/input_list.txt
LOG         = /data/local/tmp/qnn/mobilenet_v2/qnn_throughput_all.log
```

推送脚本到手机：

```bash
adb push scripts/run_qnn_throughput_all.sh /data/local/tmp/qnn/
adb shell "chmod +x /data/local/tmp/qnn/run_qnn_throughput_all.sh"
```

手机端运行：

```sh
cd /data/local/tmp/qnn
./run_qnn_throughput_all.sh
```

如果想把每个 backend 跑 30 秒：

```sh
LOOP_SECONDS=30 ./run_qnn_throughput_all.sh
```

查看结果：

```sh
cat /data/local/tmp/qnn/mobilenet_v2/qnn_throughput_all.log
```

注意：`qnn_context_binary` 通常和生成它的 backend / SoC 绑定。当前 `mobilenet_v2.bin` 是 AI Hub 为 Snapdragon 8 Gen 3 生成的 context binary，更可能主要面向 HTP。因此如果 CPU 或 GPU backend 失败，而 HTP backend 成功，这不一定说明 CPU/GPU 后端不可用，而可能是当前这份 context binary 不适合 CPU/GPU。后续如果要严谨比较 CPU/GPU/HTP，需要分别导出或生成对应 backend 可加载的 QNN 产物。

### 11.1 脚本实测结果

手机端运行：

```sh
cd /data/local/tmp/qnn
./run_qnn_throughput_all.sh
```

脚本成功完成了 CPU、GPU、HTP 三个 backend 的测试流程，并把结果保存到：

```text
/data/local/tmp/qnn/mobilenet_v2/qnn_throughput_all.log
```

运行环境：

```text
model_bin    : /data/local/tmp/qnn/mobilenet_v2/mobilenet_v2.bin
input_list   : /data/local/tmp/qnn/mobilenet_v2/input/input_list.txt
loop_seconds : 10
```

CPU backend 结果：

```text
Could not create context from binary
Failed to create context binary Returned StatusCode: FAILURE
A Context with Name = cpu_context does not exist in this Backend
return_code: 1
```

GPU backend 结果：

```text
Could not create context from binary
Failed to create context binary Returned StatusCode: FAILURE
A Context with Name = gpu_context does not exist in this Backend
return_code: 1
```

HTP backend 结果：

```text
PerformanceMonitor
                                   Name:   Time(us)
                               TotalTime:    10285222
                                   name:      repeat     average         min         max
              htp_thread_1:threadExecute:           1    10001512    10001512    10001512
  htp_thread_1:mobilenet_v2:graphExecute:        4480        2229        1905        3315
Segmentation fault
return_code: 139
```

换算：

| Backend | 结果 | graphExecute avg | graphExecute min | graphExecute max | Throughput | 说明 |
|---|---|---:|---:|---:|---:|---|
| CPU | Failed | - | - | - | - | 当前 `.bin` 不是 CPU context binary |
| GPU | Failed | - | - | - | - | 当前 `.bin` 不是 GPU context binary |
| HTP / DSP | Success, exit segfault | 2.229 ms | 1.905 ms | 3.315 ms | 约 448 inf/s | graph 执行成功，退出阶段崩溃 |

这里的 CPU/GPU 失败不是因为 CPU/GPU QNN backend 不可用，而是因为当前 `mobilenet_v2.bin` 是 backend-specific 的 context binary。它可以被 HTP backend 正确加载，但不能被 CPU/GPU backend 直接加载。

当前结论：

```text
MobileNet V2 W8A8 qnn_context_binary 已经在 OnePlus 12 本地 QNN HTP 上跑通。
CPU/GPU 对比需要单独准备对应 backend 可加载的产物，不能复用这一份 HTP context binary。
```

## 12. qnn-throughput-net-run 退出崩溃记录

当前使用 `qnn-throughput-net-run` 跑 MobileNet V2 HTP benchmark 时，模型执行和性能结果可以正常输出，但程序退出阶段会出现：

```text
Segmentation fault
```

logcat 中的关键栈：

```text
Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), tid: forw_prop_thrup
fastrpc_apps_user_deinit done
backtrace:
#00 <unknown>
#01 libc.so (pthread_key_clean_all)
#02 libc.so (pthread_exit)
#03 libc.so (__pthread_start)
```

这说明崩溃发生在线程退出时的 TLS / 资源清理阶段，而不是 graph 执行阶段。结合现象看，更像是 `qnn-throughput-net-run` 在 Android + HTP + cached context binary 组合下的清理顺序问题：

```text
benchmark 已完成
PerformanceMonitor 已打印
FastRPC deinit 已发生
worker thread 退出时清理 TLS
访问已失效地址并触发 SIGSEGV
```

对后续自己写 runner 的启发：

1. 第一版先写单线程循环 benchmark。
2. 不照搬 `qnn-throughput-net-run` 的多线程退出方式。
3. 所有执行线程必须先停止并 `join`。
4. 确认没有 graph execution 正在运行后，再释放 graph / context / backend。
5. 第一版可以避免手动 `dlclose` QNN `.so`，让进程退出时由系统回收。

推荐释放顺序：

```text
stop worker threads
join worker threads
free input/output buffers
free graph/context handles
free backend/device/profile handles
exit process
```
