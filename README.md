# Cross-Backend LLM Inference on Snapdragon

This repository records local LLM inference experiments on a Snapdragon 8 Gen 3 phone, using OnePlus 12 as the test device. The current benchmark focus is Qwen2.5-3B-Instruct GGUF models running through `llama.cpp` on CPU and OpenCL GPU backends.

The project is still experimental. CPU results are the most stable baseline; OpenCL results show useful GPU acceleration for prompt prefill, but not yet for token-by-token decode.

## Device and Runtime

- Device: OnePlus 12
- SoC: Snapdragon 8 Gen 3
- GPU: Adreno 750
- Runtime environment: Android / Termux / ADB
- Inference engine: `llama.cpp`
- Benchmark tool: `llama-bench`
- CPU profiling: `simpleperf`
- GPU backend tested: OpenCL on Qualcomm Adreno 750

## Repository Layout

```text
.
├── data/
│   ├── cpu/
│   │   ├── baseline_threads.log
│   │   ├── quant_sweep.log
│   │   └── llama-bench-report.html
│   └── gpu/
│       └── opencl_ngl_sweep.log
├── docs/
│   ├── analysis.md
│   └── oneplus12-llm-startup.md
└── scripts/
    ├── run_baseline.sh
    ├── run_quant_sweep.sh
    ├── run_gpu_ngl_sweep.sh
    └── run_opencl_ngl_sweep.sh
```

## Key Results

CPU remains the best backend for decode speed in the current setup.

| Model | Backend | Config | pp512 tok/s | tg128 tok/s | Note |
| --- | --- | --- | ---: | ---: | --- |
| Q4_0 | CPU | 6 threads | 107.03 | 21.18 | Best CPU prefill for Q4_0 |
| Q4_0 | CPU | 4 threads | 87.94 | 21.67 | Best CPU decode for Q4_0 |
| Q4_K_M | CPU | 6 threads | 61.24 | 16.70 | Balanced CPU option |
| Q4_0 | OpenCL | `ngl=99`, 4 threads | 116.45 | 14.05 | Best prefill result |
| Q6_K | OpenCL | `ngl=99`, 4 threads | 51.57 | 7.27 | Prefill improves, decode drops |
| Q8_0 | OpenCL | `ngl=99`, 4 threads | 65.06 | 9.06 | Slower than CPU baseline |

Main observations:

- CPU `Q4_0` gives the fastest text generation: about `21.67 tok/s`.
- CPU `Q4_0` gets its best prefill at 6 threads: about `107.03 tok/s`.
- OpenCL `Q4_0 + ngl=99` improves prefill from the CPU best `107.03` to `116.45 tok/s`.
- OpenCL decode is slower than CPU for all tested models.
- GPU offload helps larger prompt processing, but current Adreno OpenCL decode is limited by per-token latency, synchronization, and kernel launch overhead.

CPU vs OpenCL GPU comparison:

| Model | CPU pp512 | GPU OpenCL pp512 | pp512 Change | CPU tg128 | GPU OpenCL tg128 | tg128 Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_0 | 107.03 | 116.45 | +8.8% | 21.67 | 14.05 | -35.2% |
| Q6_K | 44.92 | 51.57 | +14.8% | 12.69 | 7.27 | -42.7% |
| Q8_0 | 90.01 | 65.06 | -27.7% | 12.87 | 9.06 | -29.6% |

See [docs/analysis.md](docs/analysis.md) for the full analysis.

## Scripts

CPU thread sweep:

```bash
bash scripts/run_baseline.sh
```

CPU quantization sweep:

```bash
bash scripts/run_quant_sweep.sh
```

OpenCL `ngl` sweep on phone:

```bash
adb push scripts/run_opencl_ngl_sweep.sh /data/local/tmp/llama-opencl/
adb shell chmod +x /data/local/tmp/llama-opencl/run_opencl_ngl_sweep.sh

adb shell
cd /data/local/tmp/llama-opencl
./run_opencl_ngl_sweep.sh
```

The OpenCL sweep log is written to:

```text
/data/local/tmp/llama-opencl/opencl_ngl_sweep.log
```

Pull it back to the repo with:

```bash
adb pull /data/local/tmp/llama-opencl/opencl_ngl_sweep.log data/gpu/opencl_ngl_sweep.log
```

## Documentation

- [docs/oneplus12-llm-startup.md](docs/oneplus12-llm-startup.md): full setup notes for Termux, ADB, CPU benchmark, Vulkan attempts, OpenCL setup, Snapdragon Profiler, and `simpleperf`.
- [docs/analysis.md](docs/analysis.md): CPU vs OpenCL benchmark analysis.
