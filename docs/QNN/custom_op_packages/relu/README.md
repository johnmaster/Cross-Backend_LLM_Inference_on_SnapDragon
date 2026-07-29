# QNN Custom Relu Op Package

本目录包含 QAIRT/QNN SDK 2.47 下 CPU 与 HTP Custom Relu Op Package 的完整实验文件，包括配置、源码、测试模型、输入输出和编译产物。

## 目录结构

```text
relu/
├── cpu/
│   ├── README.md
│   ├── ReluOpPackage/
│   │   ├── config/
│   │   ├── src/
│   │   ├── makefiles/
│   │   ├── libs/
│   │   └── obj/
│   ├── model/
│   ├── model_libs/
│   └── test_data/
└── htp/
    ├── README.md
    ├── ReluOpPackage/
    │   ├── config/
    │   ├── src/
    │   └── build/
    ├── model/
    ├── model_libs/
    ├── test_data/
    └── output/
```

## 文档入口

- [CPU Custom Relu Op Package](cpu/README.md)
- [HTP Custom Relu Op Package](htp/README.md)

CPU 目录中的 `libReluOpPackage.so` 面向 QNN CPU backend。HTP 目录同时保留 ARM64 Prepare package 和 Hexagon v75 execute package：

```text
htp/ReluOpPackage/build/aarch64-android/libQnnReluOpPackage.so
htp/ReluOpPackage/build/hexagon-v75/libQnnReluOpPackage.so
```
