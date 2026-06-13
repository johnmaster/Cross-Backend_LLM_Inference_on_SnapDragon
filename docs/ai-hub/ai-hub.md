# Qualcomm AI Hub 学习记录

本文记录当前在 Linux 主机上安装和使用 Qualcomm AI Hub 的过程，以及已经遇到的问题和解决方式。

## 1. 当前目标

当前阶段先不直接学习 QNN SDK，而是通过 Qualcomm AI Hub 熟悉 Snapdragon 设备上的模型 benchmark、profile、compile 和部署产物导出流程。

学习路线：

1. 安装 AI Hub 客户端。
2. 配置 API token。
3. 查看云端可用设备。
4. 选择接近 OnePlus 12 / Snapdragon 8 Gen 3 的 proxy device。
5. 先用官方小模型跑通 benchmark。
6. 再逐步过渡到 QNN / HTP / NPU。

## 2. Python 环境

建议使用单独的 conda 环境，避免和已有 PyTorch / boto / aiobotocore 环境冲突。

创建环境：

```bash
conda create -n qai_hub python=3.10 -y
```

如果 `conda activate` 报错：

```text
CondaError: Run 'conda init' before 'conda activate'
```

先执行：

```bash
conda init
```

`conda init` 会修改 `~/.bashrc`，但当前 shell 不会立刻生效。可以关闭并重新打开终端，或者在当前终端执行：

```bash
source ~/.bashrc
```

也可以直接加载 conda 脚本：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
```

激活环境：

```bash
conda activate qai_hub
```

退出当前环境时不要带环境名：

```bash
conda deactivate
```

错误写法：

```bash
conda deactivate qai_hub
```

会报：

```text
ArgumentError: deactivate does not accept arguments
```

## 3. 安装 AI Hub

在 `qai_hub` 环境中安装：

```bash
pip install --upgrade pip
pip install qai-hub
```

如果要使用官方模型示例，再安装：

```bash
pip install qai-hub-models
```

安装过程中曾遇到依赖冲突提示，例如：

```text
aiobotocore requires botocore<1.34.70
torchvision requires torch==2.6.0
torchaudio requires torch==2.6.0
```

这类提示说明当前 Python 环境里已有包版本和新安装包不完全兼容。`qai-hub` 本身可能已经安装成功，但为了避免污染已有环境，推荐使用独立的 `qai_hub` conda 环境。

## 4. 配置 token

已经拿到 Qualcomm AI Hub token 后，在本机配置：

```bash
qai-hub configure --api_token <YOUR_TOKEN>
```

不要把 token 写入仓库或粘贴到公开文档中。

验证：

```bash
qai-hub --help
qai-hub list-devices
```

如果能列出设备，说明客户端安装和 token 配置成功。

## 5. pip 代理问题

安装 `qai-hub-models` 时曾遇到：

```text
HTTPSConnection(host='127.0.0.1', port=10808): Failed to establish a new connection
```

原因是当前 shell 中设置了代理环境变量：

```text
http_proxy=http://127.0.0.1:10808/
https_proxy=http://127.0.0.1:10808/
all_proxy=socks://127.0.0.1:10808/
```

但本地 `127.0.0.1:10808` 代理服务没有启动，导致 pip 无法联网。

检查代理：

```bash
env | grep -Ei "proxy|http_proxy|https_proxy"
```

临时关闭代理：

```bash
unset http_proxy
unset https_proxy
unset ftp_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY
unset ALL_PROXY
```

然后重新安装：

```bash
pip install qai-hub-models
```

如果想找到代理是哪里设置的：

```bash
grep -nEi "10808|proxy|http_proxy|https_proxy|all_proxy" ~/.bashrc ~/.profile ~/.bash_profile 2>/dev/null
```

可以把相关 `export` 行注释掉，或者在 `~/.bashrc` 中写两个开关函数：

```bash
proxy_on() {
  export http_proxy=http://127.0.0.1:10808/
  export https_proxy=http://127.0.0.1:10808/
  export ftp_proxy=http://127.0.0.1:10808/
  export all_proxy=socks://127.0.0.1:10808/
  export HTTP_PROXY=$http_proxy
  export HTTPS_PROXY=$https_proxy
  export FTP_PROXY=$ftp_proxy
  export ALL_PROXY=$all_proxy
}

proxy_off() {
  unset http_proxy https_proxy ftp_proxy all_proxy
  unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
}
```

以后需要代理：

```bash
proxy_on
```

不需要代理：

```bash
proxy_off
```

## 6. 选择 AI Hub 目标设备

当前本地设备：

```text
OnePlus 12
Snapdragon 8 Gen 3
Adreno 750
Android
```

通过：

```bash
qai-hub list-devices | grep -Ei "8 gen 3|8gen3|snapdragon.*8|oneplus|samsung|xiaomi|qualcomm"
```

筛选后，最接近 OnePlus 12 / Snapdragon 8 Gen 3 的 AI Hub 设备是：

```text
Samsung Galaxy S24 (Family)
Android 14
qualcomm-snapdragon-8gen3, sm8650
```

推荐优先使用：

```bash
--device "Samsung Galaxy S24 (Family)" --device-os 14
```

也可以使用更具体的设备：

```bash
--device "Samsung Galaxy S24 Ultra" --device-os 14
```

当前建议先用 `Family`：

```text
Local device: OnePlus 12 / Snapdragon 8 Gen 3
AI Hub proxy: Samsung Galaxy S24 (Family) / Snapdragon 8 Gen 3 / Android 14
CLI: --device "Samsung Galaxy S24 (Family)" --device-os 14
```

## 7. 下一步：跑第一个模型

当前已经安装成功：

```text
qai-hub
qai-hub-models
```

建议第一个模型使用 MobileNet V2，因为它小、快、出错少，适合熟悉 AI Hub 流程。

先确认 demo 是否可用：

```bash
python -m qai_hub_models.models.mobilenet_v2.demo
```

查看 export 支持的参数：

```bash
python -m qai_hub_models.models.mobilenet_v2.export --help
```

如果参数支持设备选择，可以尝试：

```bash
python -m qai_hub_models.models.mobilenet_v2.export \
  --device "Samsung Galaxy S24 (Family)" \
  --device-os 14
```

如果参数不匹配，以 `--help` 输出为准。

第一阶段目标不是追求复杂模型，而是跑通：

```text
compile -> profile / benchmark -> download artifact
```

跑通后，把 benchmark 结果记录到仓库中，例如：

```text
data/ai-hub/
docs/ai-hub/
```

## 8. MobileNet V2 TFLite Float 跑通记录

当前已经使用 MobileNet V2 跑通 AI Hub 的完整流程：

```text
upload -> compile -> profile -> inference -> download
```

运行命令：

```bash
python -m qai_hub_models.models.mobilenet_v2.export \
  --device "Samsung Galaxy S24 (Family)" \
  --device-os 14 \
  --target-runtime tflite \
  --precision float \
  --output-dir /home/lingbok/export_assets/mobilenet_v2-s24-tflite-float
```

AI Hub 任务 ID：

```text
compile job: jgnxdy3v5
profile job: jgl79z3e5
inference job: j56v9jnvp
hub model id: mqk9y98kn
```

结果摘要：

| 项目 | 值 |
| --- | --- |
| Device | Samsung Galaxy S24 (Family), Android 14 |
| Runtime | TFLITE |
| Estimated inference time | 0.6 ms |
| Estimated peak memory usage | [0, 52] MB |
| Total ops | 71 |
| Compute units | npu 71 ops, gpu 0 ops, cpu 0 ops |
| LiteRT | 1.4.4 |
| QAIRT | 2.45.0.260326154327 |

校验结果：

| Output | Shape | PSNR |
| --- | --- | ---: |
| class_logits | (1, 1000) | 51.4852 |

PSNR 大于 30 dB 通常可以认为 on-device 输出和 local CPU 输出足够接近。本次 `class_logits` 的 PSNR 为 51.4852，说明精度校验正常。

本地导出目录：

```text
/home/lingbok/export_assets/mobilenet_v2-s24-tflite-float/mobilenet_v2-tflite-float/
```

目录中包含：

```text
labels.txt
metadata.json
mobilenet_v2.tflite
```

结论：

- MobileNet V2 TFLite float 已经在 AI Hub hosted device 上成功编译、profile、推理并下载。
- 虽然 target runtime 是 `tflite`，但在 Samsung Galaxy S24 Family 上，AI Hub 把 71 个 op 全部映射到了 NPU。
- 这是后续学习 AI Hub / QNN / NPU benchmark 的第一个成功样例。

## 9. 导出文件说明

MobileNet V2 导出的三个主要文件作用如下。

### `mobilenet_v2.tflite`

这是核心模型文件。它是 AI Hub 编译/优化后的 TFLite 模型，可以在 Android / LiteRT / TFLite runtime 上加载运行。

真正执行推理时，主要使用这个文件。

### `labels.txt`

这是分类标签表。MobileNet V2 输出的是 1000 维分类 logits，每个输出 index 对应 ImageNet 的一个类别。

例如模型输出某个 index 最高时，需要通过 `labels.txt` 把 index 转成人类可读的类别名：

```text
model output index -> labels.txt -> class name
```

因此，如果只做性能 benchmark，`labels.txt` 不是必需的；如果要展示类似下面的分类结果，就需要它：

```text
Samoyed: 74.6%
Pomeranian: 10.2%
Arctic fox: 7.8%
```

### `metadata.json`

这是模型元信息文件。它通常记录导出配置、模型信息、输入输出信息、runtime、精度、AI Hub 相关信息等。

它一般不直接参与推理，但非常适合做实验归档和复现记录。查看方式：

```bash
cat /home/lingbok/export_assets/mobilenet_v2-s24-tflite-float/mobilenet_v2-tflite-float/metadata.json
```

简单总结：

| 文件 | 作用 |
| --- | --- |
| `mobilenet_v2.tflite` | 真正用于推理的模型文件 |
| `labels.txt` | 把分类 index 转成人类可读标签 |
| `metadata.json` | 记录模型导出和部署相关元信息 |

## 10. AI Hub 上传超时问题

运行 MobileNet V2 export 时曾遇到上传模型到 AI Hub 失败：

```text
Timeout occurred while communicating with tetrahub-qprod-userdata.s3-accelerate.amazonaws.com
Read timed out. (read timeout=4)
```

当时正在上传：

```text
mobilenet_v2.pt
```

文件大小约 13.9 MB。AI Hub 自动重试多次后仍失败。

这个问题不是模型编译失败，也不是设备不支持，而是本机网络到 AI Hub 使用的 S3 endpoint 不稳定。

处理建议：

1. 检查是否设置了失效代理：

```bash
env | grep -Ei "proxy|http_proxy|https_proxy"
```

2. 如果代理没有启动，先清掉代理：

```bash
unset http_proxy https_proxy ftp_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
```

3. 如果需要代理，则先启动代理软件，并确认 S3 endpoint 能连接：

```bash
curl -I https://tetrahub-qprod-userdata.s3-accelerate.amazonaws.com
```

即使返回 403 / 404，也说明网络能连到 endpoint；真正的问题是 timeout。

4. 重新运行同一条 export 命令。该问题可能只是临时网络抖动，重跑后可以成功。

本次实际情况：第一次上传超时失败，第二次重跑后成功完成 upload、compile、profile、inference 和 download。
