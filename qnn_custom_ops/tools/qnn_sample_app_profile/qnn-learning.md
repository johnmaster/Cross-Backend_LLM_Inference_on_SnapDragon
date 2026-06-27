# QNN SampleApp 调用时序图

本文档根据 `examples/QNN/SampleApp/SampleApp/src/main.cpp` 和
`examples/QNN/SampleApp/SampleApp/src/QnnSampleApp.cpp` 梳理
`qnn-sample-app` 的主要执行链路。

## 主流程

```mermaid
sequenceDiagram
    autonumber
    participant User as User/CLI
    participant Main as main.cpp
    participant App as QnnSampleApp
    participant DL as DynamicLoadUtil/PAL
    participant Backend as QNN Backend
    participant Model as model.so / DLC
    participant IO as IOTensor
    participant FS as File System

    User->>Main: 启动 qnn-sample-app(argc, argv)
    Main->>Main: qnn::log::initializeLogging()
    Main->>App: processCommandLine(argc, argv, loadFromCachedBinary)
    App->>DL: 加载 backend library / model library
    DL-->>App: 填充 QNN interface 与 composeGraphs 函数指针
    Main->>App: getBackendBuildId()
    App-->>Main: Backend build version

    Main->>App: initialize()
    App->>FS: 创建 output 目录
    App->>FS: readInputLists(input_list)
    App->>Backend: logCreate(callback, level)
    Backend-->>App: log handle

    Main->>App: initializeBackend()
    App->>Backend: backendCreate(logHandle, backendConfig)
    Backend-->>App: backendHandle

    Main->>App: isDevicePropertySupported()
    alt 支持 device property
        Main->>App: createDevice()
        App->>Backend: deviceCreate(logHandle, config)
        Backend-->>App: deviceHandle
    end

    Main->>App: initializeProfiling()
    opt profiling enabled
        App->>Backend: profileCreate(backendHandle, level)
        Backend-->>App: profileBackendHandle
        opt serialize profile logs
            App->>Backend: systemProfileCreateSerializationTarget(...)
            Backend-->>App: serializationTargetHandle
        end
    end

    Main->>App: registerOpPackages()
    opt provided op packages
        App->>Backend: backendRegisterOpPackage(path, provider)
    end

    alt 从 model.so / DLC 创建 context 和 graph
        Main->>App: createContext()
        App->>Backend: contextCreate(backendHandle, deviceHandle, contextConfig)
        Backend-->>App: contextHandle

        Main->>App: composeGraphs()
        alt 使用 DLC
            App->>Model: composeGraphsFromDlc(systemInterface, backendHandle, contextHandle, ...)
            Model-->>App: graphsInfo / graphsCount
        else 使用 model.so
            App->>Model: QnnModel_composeGraphs(backendHandle, qnnInterface, contextHandle, graphConfigs, ...)
            Model-->>App: graphsInfo / graphsCount
        end

        Main->>App: finalizeGraphs()
        loop each graph
            App->>Backend: graphFinalize(graph, profileBackendHandle, nullptr)
            Backend-->>App: graph finalized
            opt profiling enabled
                App->>Backend: extractBackendProfilingInfo(...)
            end
        end
        opt save context binary
            App->>Backend: contextGetBinarySize / contextGetBinary
            App->>FS: 写入 context binary
        end
    else 从 cached binary 恢复 context
        Main->>App: createFromBinary()
        App->>FS: 读取 cached binary
        App->>Backend: contextCreateFromBinary(...)
        Backend-->>App: contextHandle + graphsInfo
        opt backend 支持 finalize deserialized graph
            Main->>App: finalizeGraphs()
            App->>Backend: graphFinalize(...)
        end
    end

    Main->>App: executeGraphs()
    App->>IO: setupInputAndOutputTensors(graphInfo)
    App->>IO: populateInputTensors(...)
    App->>Backend: graphExecute(graph, inputs, outputs, profileBackendHandle, nullptr)
    Backend-->>App: output tensors filled
    App->>IO: writeOutputTensors(...)
    App->>IO: tearDownInputAndOutputTensors(...)
    App->>App: freeGraphsInfo()

    Main->>App: freeContext()
    App->>Backend: contextFree(contextHandle, profileBackendHandle)
    Main->>App: freeDevice()
    App->>Backend: deviceFree(deviceHandle)
    Main->>App: terminateBackend()
    App->>Backend: profileFree(profileBackendHandle)
    App->>Backend: backendFree(backendHandle)
    Main->>DL: dlClose(backendHandle/modelHandle)
```

## executeGraphs 内部流程

```mermaid
sequenceDiagram
    autonumber
    participant App as QnnSampleApp::executeGraphs
    participant IO as IOTensor
    participant Backend as QNN Backend
    participant FS as File System

    loop run < m_numInferences
        loop each graph in m_graphsInfo
            App->>IO: setupInputAndOutputTensors(&inputs, &outputs, graphInfo)
            IO-->>App: Qnn_Tensor_t inputs / outputs

            App->>App: inputFileList = m_inputFileLists[graphIdx]
            loop inputFileIndexOffset < totalCount
                App->>IO: populateInputTensors(graphIdx, inputFileList, offset, ...)
                IO->>FS: 读取 input raw files
                IO-->>App: numInputFilesPopulated / batchSize

                App->>Backend: graphExecute(graph, inputs, numInputs, outputs, numOutputs, profile, nullptr)
                Backend-->>App: executeStatus

                opt profiling enabled
                    App->>Backend: extractBackendProfilingInfo(profileBackendHandle, ...)
                end

                App->>IO: writeOutputTensors(graphIdx, offset, graphName, outputs, ...)
                IO->>FS: 写出 output raw files

                App->>App: inputFileIndexOffset += numInputFilesPopulated
            end

            App->>IO: tearDownInputAndOutputTensors(inputs, outputs, numInputs, numOutputs)
        end
    end

    App->>App: qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount)
```

## 关键函数对应关系

| 阶段 | SampleApp 函数 | 主要 QNN / 工具调用 |
| --- | --- | --- |
| 参数解析与动态库加载 | `processCommandLine` | 加载 backend/model library，解析命令行参数 |
| 基础初始化 | `QnnSampleApp::initialize` | `readInputLists`、`logCreate` |
| Backend 初始化 | `QnnSampleApp::initializeBackend` | `backendCreate` |
| Device 创建 | `QnnSampleApp::createDevice` | `deviceCreate` |
| Profiling 初始化 | `QnnSampleApp::initializeProfiling` | `profileCreate`、`systemProfileCreateSerializationTarget` |
| Context 创建 | `QnnSampleApp::createContext` | `contextCreate` |
| Graph 组成 | `QnnSampleApp::composeGraphs` | `QnnModel_composeGraphs` 或 `composeGraphsFromDlc` |
| Graph finalize | `QnnSampleApp::finalizeGraphs` | `graphFinalize`，可选 `saveBinary` |
| Cached binary 恢复 | `QnnSampleApp::createFromBinary` | 从 context binary 创建 context 和 graph |
| Graph 执行 | `QnnSampleApp::executeGraphs` | `setupInputAndOutputTensors`、`populateInputTensors`、`graphExecute`、`writeOutputTensors` |
| 资源释放 | `freeContext`、`freeDevice`、`terminateBackend` | `contextFree`、`deviceFree`、`profileFree`、`backendFree` |

## 阅读提示

- `main.cpp` 控制整体顺序，并在每一步失败时通过 `reportError` 记录错误。
- `QnnSampleApp.cpp` 把 QNN C API 包装成 SampleApp 的成员函数。
- 非 cached binary 路径是 `createContext -> composeGraphs -> finalizeGraphs -> executeGraphs`。
- cached binary 路径是 `createFromBinary -> 可选 finalizeGraphs -> executeGraphs`。
- `executeGraphs` 会在函数末尾调用 `freeGraphsInfo`，而 `freeContext` 负责释放 context 以及残留的 graph/tensor 元数据。
