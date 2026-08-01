# QNN 辅助工具

本目录保存自定义算子测试和 Qwen decode 实验使用的宿主侧工具。目前包含：

```text
tools/
└── qnn_sample_app_profile/
    ├── src/                 # 定制的 QNN SampleApp 源码
    ├── make/                # Android NDK 构建文件
    ├── libs/                # ndk-build 生成的可执行文件
    ├── obj/                 # ndk-build 中间产物
    └── qnn-learning.md      # SampleApp API 调用时序
```

`qnn_sample_app_profile` 基于 QAIRT 2.47 的 `qnn-sample-app`，用于输出详细 profiling
事件，并支持仓库中固定 past128 Qwen decode graph 的持久化执行。具体构建方式、
参数和适用范围见 [qnn_sample_app_profile/README.md](qnn_sample_app_profile/README.md)。

`libs/`、`bin/` 和 `obj/` 均为构建产物；修改源码后应重新构建，不能假设仓库中
已有的二进制与当前源码一致。
