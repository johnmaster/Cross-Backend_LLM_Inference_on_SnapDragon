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

## 11. 常用 AI Hub 命令速查

这一节把两个常用入口分开记录：

```text
qai-hub        AI Hub 服务端交互 CLI：配置 token、列设备、上传模型、提交 compile/profile/link job
qai-hub-models 官方模型包辅助 CLI：查看/获取 qai-hub-models 版本资产；具体模型通常用 python -m ...export 运行
```

### 环境和版本

确认当前 Python 环境中安装的版本：

```bash
pip show qai-hub | grep Version
pip show qai-hub-models | grep Version
```

也可以用 Python 读取 package metadata：

```bash
python - <<'PY'
import importlib.metadata as m

for pkg in ["qai-hub", "qai-hub-models", "huggingface-hub", "transformers"]:
    try:
        print(pkg, m.version(pkg))
    except m.PackageNotFoundError:
        print(pkg, "not installed")
PY
```

查看 PyPI 上 `qai-hub-models` 可用版本：

```bash
pip index versions qai-hub-models -i https://pypi.org/simple
```

如果当前 pip 源是 HTTP 镜像，可能会出现：

```text
The repository located at mirrors.cloud.aliyuncs.com is not a trusted or secure host
ERROR: No matching distribution found for qai-hub-models
```

这不是包不存在，而是 pip 源或 trusted-host 配置问题。临时指定官方 PyPI 即可。

### `qai-hub` 常用命令

查看主帮助：

```bash
qai-hub --help
```

当前 `qai-hub` CLI 常见子命令：

| 命令 | 作用 |
| --- | --- |
| `qai-hub configure` | 配置 AI Hub token |
| `qai-hub list-devices` | 查看 hosted devices |
| `qai-hub list-frameworks` | 查看支持的框架/runtime |
| `qai-hub upload-model` | 上传本地模型到 AI Hub |
| `qai-hub submit-compile-job` | 提交单独 compile job |
| `qai-hub submit-profile-job` | 提交单独 profile job |
| `qai-hub submit-compile-and-profile-jobs` | 一次提交 compile + profile |
| `qai-hub submit-link-job` | 提交 link job，常用于多个 QNN DLC / LLM 分片链接 |

配置 token：

```bash
qai-hub configure --api_token <YOUR_TOKEN>
```

如果要使用另一个 profile：

```bash
qai-hub configure --profile test --api_token <YOUR_TOKEN>
qai-hub --profile test list-devices
```

查看 hosted devices：

```bash
qai-hub list-devices
```

查看某个设备的详细信息：

```bash
qai-hub list-devices --format details --device "Samsung Galaxy S24 (Family)" --device-os 14
```

筛选 Snapdragon 8 Gen 3 / Samsung S24：

```bash
qai-hub list-devices | grep -Ei "8 gen 3|8gen3|sm8650|Samsung Galaxy S24"
```

当前用于近似 OnePlus 12 的推荐设备参数：

```bash
--device "Samsung Galaxy S24 (Family)" --device-os 14
```

查看支持框架/runtime：

```bash
qai-hub list-frameworks
qai-hub list-frameworks --json
```

上传模型：

```bash
qai-hub upload-model --model resnet50.pt
qai-hub upload-model --model resnet50.pt --name resnet50_test
```

提交 compile job：

```bash
qai-hub submit-compile-job \
  --model resnet50.pt \
  --device "Samsung Galaxy S24 (Family)" \
  --device-os 14 \
  --input_specs '{"x": (1, 3, 224, 224)}' \
  --wait
```

指定 compile options，例如 target runtime：

```bash
qai-hub submit-compile-job \
  --model resnet50.pt \
  --device "Samsung Galaxy S24 (Family)" \
  --device-os 14 \
  --compile_options " --target_runtime tflite" \
  --wait
```

提交 profile job：

```bash
qai-hub submit-profile-job \
  --model resnet50.tflite \
  --device "Samsung Galaxy S24 (Family)" \
  --device-os 14 \
  --wait
```

指定 compute unit：

```bash
qai-hub submit-profile-job \
  --model resnet50.tflite \
  --device "Samsung Galaxy S24 (Family)" \
  --device-os 14 \
  --profile_options " --compute_unit cpu" \
  --wait
```

一次提交 compile + profile：

```bash
qai-hub submit-compile-and-profile-jobs \
  --model resnet50.pt \
  --input_specs '{"x": (1, 3, 224, 224)}' \
  --device "Samsung Galaxy S24 (Family)" \
  --device-os 14 \
  --wait
```

基于已有 job 重新提交：

```bash
qai-hub submit-compile-job --clone <JOB_ID>
qai-hub submit-profile-job --clone <JOB_ID>
qai-hub submit-compile-and-profile-jobs --clone <JOB_ID>
```

提交 link job：

```bash
qai-hub submit-link-job \
  --device "Samsung Galaxy S24 (Family)" \
  --device-os 14 \
  --models <MODEL_ID_1> <MODEL_ID_2> \
  --wait
```

说明：

- `qai-hub` 更偏底层，可以手动上传模型、提交 job。
- `python -m qai_hub_models.models.<model>.export` 更偏官方模型一键流程，会自动完成上传、compile、profile、download 等步骤。
- 对新手来说，先用 `qai-hub-models` 的 `export.py` 跑通官方模型，再逐步学习 `qai-hub submit-*` 更稳。

### `qai-hub-models` 常用命令

查看帮助：

```bash
qai-hub-models --help
```

当前常见子命令：

| 命令 | 作用 |
| --- | --- |
| `qai-hub-models fetch` | 获取官方模型相关静态资产 |
| `qai-hub-models versions` | 查看 `qai-hub-models` 版本信息 |

注意：`qai-hub-models` 没有 `list` 子命令。下面这个命令会报错：

```bash
qai-hub-models list
```

报错类似：

```text
qai_hub_models: error: argument {fetch,versions}: invalid choice: 'list' (choose from 'fetch', 'versions')
```

查看版本相关信息：

```bash
qai-hub-models versions
```

查看 fetch 用法：

```bash
qai-hub-models fetch --help
```

`qai-hub-models` 的模型运行通常不是通过 CLI 子命令，而是通过 Python module：

```bash
python -m qai_hub_models.models.mobilenet_v2.demo
python -m qai_hub_models.models.mobilenet_v2.export --help
python -m qai_hub_models.models.mobilenet_v2.export \
  --device "Samsung Galaxy S24 (Family)" \
  --device-os 14 \
  --target-runtime tflite \
  --precision float \
  --output-dir ~/export_assets/mobilenet_v2-s24-tflite-float
```

LLM / Genie 模型也是类似：

```bash
python -m qai_hub_models.models.qwen3_4b_instruct_2507.export --help
```

实际导出：

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 python -u -m qai_hub_models.models.qwen3_4b_instruct_2507.export \
  --target-runtime genie \
  --checkpoint DEFAULT_W4A16 \
  --device "Samsung Galaxy S24 (Family)" \
  --device-os 14 \
  --context-length 512 \
  --sequence-length 128,1 \
  --skip-profiling \
  --skip-inferencing \
  --synchronous \
  --model-cache-mode enable \
  --output-dir ~/export_assets/qwen3_4b_instruct_2507-s24-genie-c512
```

### 查看 `qai-hub-models` 支持哪些模型

列出本地 `qai_hub_models.models` 包中的所有模型模块：

```bash
python - <<'PY'
import pkgutil
import qai_hub_models.models as models

for m in sorted(pkgutil.iter_modules(models.__path__), key=lambda x: x.name):
    if m.ispkg:
        print(m.name)
PY
```

筛选 LLM / Qwen / Llama / Gemma / Phi 等模型：

```bash
python - <<'PY'
import pkgutil
import qai_hub_models.models as models

keywords = ["qwen", "llama", "gemma", "phi", "mistral", "bert", "bge"]
for m in sorted(pkgutil.iter_modules(models.__path__), key=lambda x: x.name):
    name = m.name.lower()
    if any(k in name for k in keywords):
        print(m.name)
PY
```

当前 `qai-hub-models 0.55.0` 中能看到的 Qwen 模块包括：

```text
qwen2_5_7b_instruct
qwen2_5_vl_7b_instruct
qwen2_7b_instruct
qwen3_4b
qwen3_4b_instruct_2507
```

注意：这里没有 `qwen2_5_3b_instruct`，所以不能直接用官方 `qai-hub-models` export 路线复现之前 llama.cpp 中的 Qwen2.5-3B GGUF 实验。

### 查看某个模型支持什么参数

对目标模型运行 `export --help`：

```bash
python -m qai_hub_models.models.mobilenet_v2.export --help
python -m qai_hub_models.models.qwen3_4b_instruct_2507.export --help
```

`--help` 能正常输出只说明模块存在、参数解析正常；真正能否导出，还要看该模型是否有可用 checkpoint、static assets，以及包内 `info.yaml` 是否正常。

查看每个模型目录中是否包含 `export.py`、`demo.py`、`requirements.txt`、`info.yaml`：

```bash
python - <<'PY'
import pathlib
import qai_hub_models.models as models

root = pathlib.Path(models.__path__[0])
for d in sorted(root.iterdir()):
    if not d.is_dir() or d.name.startswith("_"):
        continue
    files = {p.name for p in d.iterdir() if p.is_file()}
    caps = []
    if "export.py" in files:
        caps.append("export")
    if "demo.py" in files:
        caps.append("demo")
    if "requirements.txt" in files:
        caps.append("requirements")
    if "info.yaml" in files:
        caps.append("info")
    if caps:
        print(f"{d.name}: {', '.join(caps)}")
PY
```

### 安装某个模型的额外依赖

有些 LLM 模型需要额外依赖，例如 `transformers`、`accelerate`、`safetensors`。先找到 requirements 文件：

```bash
find "$(python - <<'PY'
import pathlib
import qai_hub_models.models as models
print(pathlib.Path(models.__path__[0]) / "qwen3_4b_instruct_2507")
PY
)" -maxdepth 2 -type f -iname "*requirements*.txt" -print
```

安装：

```bash
pip install -r /path/to/requirements.txt
```

检查关键依赖：

```bash
python - <<'PY'
import transformers
import accelerate
import safetensors

print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("deps ok")
PY
```

如果安装 Hugging Face CLI 时把 `huggingface_hub` 升到 `1.x`，可能和 `qai-hub-models 0.55.0`、`transformers 4.51.0`、`tokenizers` 冲突。当前更稳的版本是：

```bash
pip install "huggingface_hub==0.36.2"
```

### Hugging Face 下载和镜像

国内服务器访问 Hugging Face 官方站点可能失败：

```text
Network is unreachable
HTTPSConnectionPool(host='huggingface.co', port=443)
```

可以临时使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

验证：

```bash
HF_ENDPOINT=https://hf-mirror.com python - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("Qwen/Qwen3-4B-Instruct-2507", "config.json")
print(p)
PY
```

如果同时设置了代理，先检查：

```bash
env | grep -Ei "proxy|http_proxy|https_proxy"
```

不需要代理时清掉：

```bash
unset ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy
```

### 运行 Qwen3-4B Instruct 2507 Genie 导出

当前推荐先跑官方支持的 Qwen3-4B Instruct 2507，而不是手动处理 Qwen2.5-3B：

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 python -u -m qai_hub_models.models.qwen3_4b_instruct_2507.export \
  --target-runtime genie \
  --checkpoint DEFAULT_W4A16 \
  --device "Samsung Galaxy S24 (Family)" \
  --device-os 14 \
  --context-length 512 \
  --sequence-length 128,1 \
  --skip-profiling \
  --skip-inferencing \
  --synchronous \
  --model-cache-mode enable \
  --output-dir ~/export_assets/qwen3_4b_instruct_2507-s24-genie-c512
```

建议先用较小的 context：

```text
context-length: 512
sequence-length: 128,1
```

这样可以降低导出和编译成本，适合先跑通端到端流程。

### 观察长时间运行的 export

查看进程：

```bash
ps -eo pid,%cpu,%mem,etime,cmd | grep qwen3_4b_instruct_2507 | grep -v grep
```

查看缓存和输出目录大小：

```bash
du -sh ~/.cache/huggingface ~/.qaihm ~/export_assets/qwen3_4b_instruct_2507-s24-genie-c512 2>/dev/null
```

查看网络连接：

```bash
ss -tpn | grep -E "python|443|amazonaws|qualcomm|aihub"
```

组合观察：

```bash
watch -n 20 'date; ps -eo pid,%cpu,%mem,etime,cmd | grep qwen3_4b_instruct_2507 | grep -v grep; du -sh ~/.qaihm ~/export_assets/qwen3_4b_instruct_2507-s24-genie-c512 2>/dev/null; ss -tpn | grep -E "python|443|amazonaws|qualcomm|aihub" || true'
```

如果进程 CPU 仍有占用、`ss` 中还有 established 连接，就不要轻易中断。

### `info.yaml` release assets 校验问题

`qai-hub-models 0.55.0` 中，`qwen3_4b` 和 `qwen3_4b_instruct_2507` 曾遇到：

```text
pydantic_core._pydantic_core.ValidationError
Value error, Model cannot be published: no release assets available
```

实际现象：

- Hugging Face 权重和 tokenizer 已经下载成功。
- 报错发生在 `QAIHMModelInfo.from_model(model_id).name`。
- 该代码只是为了获取模型显示名，却触发了包内 `info.yaml` 的 release assets 校验。

确认当前 PyPI 最新版：

```bash
pip index versions qai-hub-models -i https://pypi.org/simple
```

当前记录：

```text
INSTALLED: 0.55.0
LATEST:    0.55.0
```

临时绕过方式是在虚拟环境中 patch 这一行：

```bash
python - <<'PY'
import pathlib
import qai_hub_models.models._shared.llm.export as e

p = pathlib.Path(e.__file__)
print("file:", p)

text = p.read_text()
old = "model_display_name = QAIHMModelInfo.from_model(model_id).name"
new = 'model_display_name = model_id  # patched: bypass broken release-assets validation'

if old not in text:
    print("target line not found")
else:
    backup = p.with_suffix(p.suffix + ".bak")
    backup.write_text(text)
    p.write_text(text.replace(old, new, 1))
    print("patched")
    print("backup:", backup)
PY
```

说明：

- 这是临时 workaround，不是长期方案。
- 只修改当前 venv 中的 site-packages，不修改仓库代码。
- patch 后 export 可以继续进入 static assets 下载和 AI Hub 编译流程。
- 如果后续 Qualcomm 发布新版 `qai-hub-models` 修复该问题，应优先升级官方版本。

## 12. 租服务器尝试 Genie LLM 的记录和结论

本节记录一次在云服务器上尝试通过 `qai-hub-models` 导出 Genie LLM 的过程。结论是：当前网络和模型授权条件下，不适合继续烧服务器时间硬跑。

### 初始目标

原始目标是跑通 Genie + LLM，并尽量选择 Qwen 系列，方便和之前在手机上用 llama.cpp 跑的 Qwen2.5-3B CPU / OpenCL / Vulkan 结果做对比。

本地和服务器上 `qai-hub-models 0.55.0` 能看到的 Qwen 模块为：

```text
qwen2_5_7b_instruct
qwen2_5_vl_7b_instruct
qwen2_7b_instruct
qwen3_4b
qwen3_4b_instruct_2507
```

没有官方 `qwen2_5_3b_instruct` 模块，因此不能直接用 AI Hub 官方模型路线复现 Qwen2.5-3B。

### 服务器环境

服务器中确认：

```bash
pip index versions qai-hub-models -i https://pypi.org/simple
```

结果：

```text
INSTALLED: 0.55.0
LATEST:    0.55.0
```

说明当时已经是 PyPI 最新公开版，不能通过普通升级解决问题。

`qwen3_4b` 和 `qwen3_4b_instruct_2507` 都需要额外依赖，例如：

```text
transformers
accelerate
safetensors
```

曾遇到：

```text
ModuleNotFoundError: No module named 'transformers'
ModuleNotFoundError: No module named 'accelerate'
```

安装对应 requirements 后解决。

### `qwen3_4b` / `qwen3_4b_instruct_2507` 的 metadata 校验问题

`qwen3_4b` 导出时，模型权重还没有真正开始编译，就先失败在：

```text
Value error, Model cannot be published: no release assets available
```

`qwen3_4b_instruct_2507` 也遇到同类问题。报错位置为：

```text
QAIHMModelInfo.from_model(model_id).name
```

该代码只是为了取得显示名，却触发了包内 `info.yaml` 的 release assets 校验。临时 patch 后可以绕过：

```python
model_display_name = model_id
```

patch 后，`qwen3_4b_instruct_2507` 继续进入后续流程，开始下载 Qualcomm 预制 W4A16 static asset。

### Qwen3-4B Instruct 2507 的瓶颈

`qwen3_4b_instruct_2507` 的 Hugging Face 权重和 tokenizer 可以下载完成：

```text
model-00001-of-00003.safetensors 3.96G
model-00002-of-00003.safetensors 3.99G
model-00003-of-00003.safetensors 99.6M
Loading checkpoint shards: 100%
```

但后续需要下载 Qualcomm S3 上的 static asset：

```text
https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b_instruct_2507/v1/qwen34instruct2507_w4a16_adascale.zip
```

该文件体积约十几个 GB。服务器到 Qualcomm S3 的下载速度非常慢，曾观察到十几分钟只增长到几 MB：

```text
/root/.qaihm 2.5M
elapsed 13:42
```

因此继续下载不划算。

### 美国服务器 SOCKS5 代理尝试

尝试使用美国小服务器作为 SOCKS5 代理，只让阿里云服务器通过它访问外部资源：

```bash
ssh -N -D 127.0.0.1:10809 root@<US_SERVER_IP>
```

如果端口被占用：

```text
bind [127.0.0.1]:10808: Address already in use
```

可以换端口，例如 `10809`。

设置代理：

```bash
export ALL_PROXY=socks5h://127.0.0.1:10809
export HTTPS_PROXY=socks5h://127.0.0.1:10809
export HTTP_PROXY=socks5h://127.0.0.1:10809
```

测试：

```bash
curl --socks5-hostname 127.0.0.1:10809 -I https://qaihub-public-assets.s3.us-west-2.amazonaws.com
```

这个方法可以解决“是否能连通”的问题，但对于十几个 GB 的 Qwen static asset，实际下载速度仍然不适合作为按小时计费服务器上的主路线。

### Llama 3.2 1B 尝试

为了避免 Qwen3-4B 的十几个 GB static asset，尝试了较小的官方 LLM：

```text
llama_v3_2_1b_instruct
```

导出命令使用的参数名和 Qwen 不同：

```bash
python -m qai_hub_models.models.llama_v3_2_1b_instruct.export \
  --target-runtime genie \
  --device "Samsung Galaxy S24 (Family)" \
  --device-os 14 \
  --context-lengths 512 \
  --sequence-lengths 128,1 \
  --skip-profiling \
  --skip-inferencing \
  --skip-downloading \
  --output-dir ~/export_assets/llama-v3_2-1b-s24-genie-c512
```

注意：

```text
Qwen:  --context-length / --sequence-length
Llama: --context-lengths / --sequence-lengths
```

Llama 1B 需要下载 Qualcomm S3 的 `model.encodings`，但更关键的问题是它还需要访问 Hugging Face 上的 Meta gated repo：

```text
meta-llama/Llama-3.2-1B-Instruct
```

通过 SOCKS5 代理后网络可达，但报错变为权限问题：

```text
Cannot access gated repo
Access to model meta-llama/Llama-3.2-1B-Instruct is restricted.
401 Client Error
```

这说明网络已经通了，但 Hugging Face 账号没有该 Llama 模型的授权，或者服务器上没有用已授权账号登录。

需要在 Hugging Face 模型页申请/同意 Meta Llama license 后，再使用：

```bash
huggingface-cli login
```

或：

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
```

但本次没有继续推进，因为服务器已经按成本考虑关停。

### 关于 `HF_ENDPOINT`

曾尝试：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

这只影响 Hugging Face 下载，不影响 Qualcomm S3：

```text
qaihub-public-assets.s3.us-west-2.amazonaws.com
```

因此：

- `HF_ENDPOINT` 可以帮助下载 Hugging Face 模型配置和权重。
- 它不能加速 Qualcomm AI Hub static assets。
- 如果走官方 Hugging Face + 美国 SOCKS5，应取消 `HF_ENDPOINT`：

```bash
unset HF_ENDPOINT
export ALL_PROXY=socks5h://127.0.0.1:10809
export HTTPS_PROXY=socks5h://127.0.0.1:10809
export HTTP_PROXY=socks5h://127.0.0.1:10809
```

### 本次止损结论

本次 Genie LLM 路线没有继续的原因不是单一错误，而是多个现实条件叠加：

1. 官方 `qai-hub-models` 没有 Qwen2.5-3B 模块，无法直接和之前 GGUF 实验完全对齐。
2. `qwen3_4b` / `qwen3_4b_instruct_2507` 在 `qai-hub-models 0.55.0` 中有 `info.yaml` release assets 校验问题，需要临时 patch。
3. Qwen3-4B Instruct 2507 的 Qualcomm W4A16 static asset 约十几个 GB，服务器下载速度过慢。
4. Llama 3.2 1B 更小，但 Hugging Face 模型是 gated repo，需要 Meta 授权和已登录 token。
5. 按小时计费服务器不适合在外部下载链路不确定时长时间空跑。

最终选择关停服务器止损是合理的。

### 后续更稳的路线

建议后续分两条线推进：

1. 本地继续 QNN SDK / Genie 小模型路线：

```text
MobileNet V2 QNN context -> qnn-net-run -> qnn-throughput-net-run -> 自己写 QNN runner
BGE Genie embedding -> 继续熟悉 Genie config / tokenizer / model.bin
```

这条路线不依赖大模型下载，也不需要高配云服务器。

2. LLM Genie 等网络和授权准备好后再做：

```text
确认 Hugging Face token 和 gated model 授权
确认 Qualcomm S3 下载速度可接受
优先选择 1B 级别官方模型跑通
再回到 Qwen / 3B / 4B 模型
```

下一次租服务器前，建议先做三项预检查：

```bash
qai-hub list-devices | grep -i "Samsung Galaxy S24"
python -m qai_hub_models.models.llama_v3_2_1b_instruct.export --help
python - <<'PY'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
print(cfg.model_type)
PY
```

如果第三条因为 gated repo 失败，先不要租高配服务器，应该先处理 Hugging Face 授权。
