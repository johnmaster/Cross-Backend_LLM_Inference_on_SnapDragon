#!/usr/bin/env python3

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
QUANT_ROOT = ROOT.parent
ENCODINGS = {
    "representative": (np.float32(0.0039213779382407665), -128),
    "narrow": (np.float32(0.0009803444845601916), -128),
    "outlier": (np.float32(0.033333249390125275), -15),
}


def metrics(reference, result):
    difference = result - reference
    signal = np.sum(reference.astype(np.float64) ** 2)
    noise = np.sum(difference.astype(np.float64) ** 2)
    cosine = np.dot(reference, result) / (
        np.linalg.norm(reference) * np.linalg.norm(result)
    )
    return (
        np.max(np.abs(difference)),
        np.mean(np.abs(difference)),
        cosine,
        10.0 * np.log10(signal / noise),
    )


def main():
    lhs = np.fromfile(ROOT / "input/lhs_float.raw", np.float32)
    fp32 = np.fromfile(
        QUANT_ROOT / "05_quantized_matmul/reference/fp32_output.raw",
        np.float32,
    )
    outputs = {
        "representative": np.fromfile(
            QUANT_ROOT
            / "06_onnx_qdq_matmul/device_output/output_float.raw",
            np.float32,
        ),
        "narrow": np.fromfile(
            ROOT / "device_output/narrow.raw", np.float32
        ),
        "outlier": np.fromfile(
            ROOT / "device_output/outlier.raw", np.float32
        ),
    }

    print(
        f"{'calibration':16s} {'scale':>11s} {'sat %':>9s} "
        f"{'levels':>7s} {'mean error':>12s} {'SQNR(dB)':>10s}"
    )
    for name, output in outputs.items():
        scale, offset = ENCODINGS[name]
        unclipped = np.rint(lhs / scale - offset)
        quantized = np.clip(unclipped, 0, 255).astype(np.uint8)
        saturated = np.count_nonzero((unclipped < 0) | (unclipped > 255))
        _, mean, _, sqnr = metrics(fp32, output)
        print(
            f"{name:16s} {scale:11.8f} "
            f"{100.0 * saturated / lhs.size:8.3f}% "
            f"{np.unique(quantized).size:7d} {mean:12.7f} {sqnr:10.3f}"
        )


if __name__ == "__main__":
    main()
