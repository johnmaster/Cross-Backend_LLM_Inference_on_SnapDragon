#!/usr/bin/env python3

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
QUANT_ROOT = ROOT.parent
CASES = {
    "representative min-max": {
        "scale": np.float32(0.0039213779382407665),
        "offset": -128,
        "output_scale": np.float32(0.03154205158352852),
        "output_offset": -124,
        "output": QUANT_ROOT
        / "06_onnx_qdq_matmul/device_output/output_float.raw",
    },
    "outlier min-max": {
        "scale": np.float32(0.033333249390125275),
        "offset": -15,
        "output_scale": np.float32(0.030052650719881058),
        "output_offset": -118,
        "output": QUANT_ROOT
        / "09_calibration_sensitivity/device_output/outlier.raw",
    },
    "outlier percentile 99.9": {
        "scale": np.float32(0.003906242549419403),
        "offset": -124,
        "output_scale": np.float32(0.021659038960933685),
        "output_offset": -123,
        "output": ROOT / "device_output/output.raw",
    },
    "outlier percentile 99.99": {
        "scale": np.float32(0.003906242549419403),
        "offset": -124,
        "output_scale": np.float32(0.02817436493933201),
        "output_offset": -116,
        "output": ROOT / "device_output/output_9999.raw",
    },
}


def metrics(reference, result):
    difference = result - reference
    signal = np.sum(reference.astype(np.float64) ** 2)
    noise = np.sum(difference.astype(np.float64) ** 2)
    return (
        np.max(np.abs(difference)),
        np.mean(np.abs(difference)),
        10.0 * np.log10(signal / noise),
    )


def main():
    lhs = np.fromfile(
        QUANT_ROOT / "05_quantized_matmul/input/lhs_float.raw", np.float32
    )
    fp32 = np.fromfile(
        QUANT_ROOT / "05_quantized_matmul/reference/fp32_output.raw",
        np.float32,
    )
    print(
        f"{'calibration':25s} {'scale':>11s} {'in sat':>9s} "
        f"{'levels':>7s} {'out sat':>9s} {'mean error':>12s} "
        f"{'SQNR(dB)':>10s}"
    )
    for name, case in CASES.items():
        scale = case["scale"]
        offset = case["offset"]
        unclipped = np.rint(lhs / scale - offset)
        quantized = np.clip(unclipped, 0, 255).astype(np.uint8)
        saturation = np.count_nonzero((unclipped < 0) | (unclipped > 255))
        output_unclipped = np.rint(
            fp32 / case["output_scale"] - case["output_offset"]
        )
        output_saturation = np.count_nonzero(
            (output_unclipped < 0) | (output_unclipped > 255)
        )
        output = np.fromfile(case["output"], np.float32)
        _, mean, sqnr = metrics(fp32, output)
        print(
            f"{name:25s} {scale:11.8f} "
            f"{100.0 * saturation / lhs.size:8.3f}% "
            f"{np.unique(quantized).size:7d} "
            f"{100.0 * output_saturation / fp32.size:8.3f}% "
            f"{mean:12.7f} {sqnr:10.3f}"
        )


if __name__ == "__main__":
    main()
