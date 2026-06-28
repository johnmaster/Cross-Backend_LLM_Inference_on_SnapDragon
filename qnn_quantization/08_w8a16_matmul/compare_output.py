#!/usr/bin/env python3

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
QUANT_ROOT = ROOT.parent


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
    fp32 = np.fromfile(
        QUANT_ROOT / "05_quantized_matmul/reference/fp32_output.raw",
        np.float32,
    )
    w8a8 = np.fromfile(
        QUANT_ROOT / "06_onnx_qdq_matmul/device_output/output_float.raw",
        np.float32,
    )
    w8a16 = np.fromfile(
        ROOT / "device_output/output_float.raw", np.float32
    )

    print(
        f"{'model':24s} {'max error':>12s} {'mean error':>12s} "
        f"{'cosine':>12s} {'SQNR(dB)':>10s}"
    )
    for name, output in (("W8A8 per-axis", w8a8), ("W8A16 per-axis", w8a16)):
        maximum, mean, cosine, sqnr = metrics(fp32, output)
        print(
            f"{name:24s} {maximum:12.7f} {mean:12.7f} "
            f"{cosine:12.9f} {sqnr:10.3f}"
        )


if __name__ == "__main__":
    main()
