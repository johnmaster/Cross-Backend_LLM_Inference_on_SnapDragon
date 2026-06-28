#!/usr/bin/env python3

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "05_quantized_matmul"
DEVICE_OUTPUT = (
    ROOT.parent
    / "06_onnx_qdq_matmul"
    / "device_output"
    / "output_float.raw"
)

# Exact encodings emitted by QAIRT 2.47 for the calibrated converter model.
LHS_SCALE = np.float32(0.0039213779382407665)
LHS_OFFSET = -128
OUTPUT_SCALE = np.float32(0.03154205158352852)
OUTPUT_OFFSET = -124


def quantize_dequantize(values, scale, offset, qmin, qmax):
    quantized = np.clip(np.rint(values / scale - offset), qmin, qmax)
    return (quantized + offset).astype(np.float32) * scale


def metrics(reference, result):
    difference = result - reference
    signal = np.sum(reference.astype(np.float64) ** 2)
    noise = np.sum(difference.astype(np.float64) ** 2)
    cosine = np.dot(reference.ravel(), result.ravel()) / (
        np.linalg.norm(reference) * np.linalg.norm(result)
    )
    return (
        float(np.max(np.abs(difference))),
        float(np.mean(np.abs(difference))),
        float(cosine),
        float(10.0 * np.log10(signal / noise)),
    )


def main():
    lhs = np.fromfile(SOURCE / "input" / "lhs_float.raw", np.float32).reshape(
        1, 1, 128, 256
    )
    rhs = np.fromfile(SOURCE / "input" / "rhs_float.raw", np.float32).reshape(
        1, 1, 256, 256
    )
    fp32 = np.matmul(lhs, rhs).astype(np.float32)

    rhs_scales = (
        np.geomspace(0.02, 0.5, 256).astype(np.float32) / np.float32(127.0)
    ).reshape(1, 1, 1, -1)
    rhs_dq = quantize_dequantize(rhs, rhs_scales, 0, -128, 127)
    lhs_dq = quantize_dequantize(
        lhs, LHS_SCALE, LHS_OFFSET, 0, 255
    )

    weight_only = np.matmul(lhs, rhs_dq).astype(np.float32)
    activation_weight = np.matmul(lhs_dq, rhs_dq).astype(np.float32)
    with_output = quantize_dequantize(
        activation_weight, OUTPUT_SCALE, OUTPUT_OFFSET, 0, 255
    )
    qnn_output = np.fromfile(DEVICE_OUTPUT, np.float32)

    rows = [
        ("Weight per-axis only", weight_only),
        ("Activation + weight", activation_weight),
        ("Activation + weight + output", with_output),
        ("QNN HTP output", qnn_output),
    ]
    print(
        f"{'stage':30s} {'max error':>12s} {'mean error':>12s} "
        f"{'cosine':>12s} {'SQNR(dB)':>10s}"
    )
    for name, result in rows:
        maximum, mean, cosine, sqnr = metrics(fp32.ravel(), result.ravel())
        print(
            f"{name:30s} {maximum:12.7f} {mean:12.7f} "
            f"{cosine:12.9f} {sqnr:10.3f}"
        )

    modeled_difference = np.abs(with_output.ravel() - qnn_output)
    print("\nModeled requantization vs QNN HTP")
    print(f"  max_abs_difference:  {modeled_difference.max():.10g}")
    print(f"  mean_abs_difference: {modeled_difference.mean():.10g}")
    print(
        "  within_one_output_lsb: "
        f"{bool(np.all(modeled_difference <= OUTPUT_SCALE + 1e-7))}"
    )


if __name__ == "__main__":
    main()
