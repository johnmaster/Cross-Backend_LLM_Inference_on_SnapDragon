#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def metrics(reference, result):
    difference = result - reference
    denominator = np.linalg.norm(reference) * np.linalg.norm(result)
    signal_power = np.sum(reference.astype(np.float64) ** 2)
    noise_power = np.sum(difference.astype(np.float64) ** 2)
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "cosine_similarity": float(
            np.dot(reference.ravel(), result.ravel()) / denominator
        ),
        "sqnr_db": float(10.0 * np.log10(signal_power / noise_power)),
    }


def print_metrics(name, reference, result):
    print(name)
    for key, value in metrics(reference, result).items():
        print(f"  {key:20s}: {value:.10g}")


def main():
    parameters = json.loads((ROOT / "parameters.json").read_text())
    fp32_reference = np.fromfile(
        ROOT / "reference" / "fp32_output.raw", dtype=np.float32
    )
    quantized_reference = np.fromfile(
        ROOT / "reference" / "per_tensor_dequant_output.raw",
        dtype=np.float32,
    )
    expected_u8 = np.fromfile(
        ROOT / "reference" / "per_tensor_output_uint8.raw",
        dtype=np.uint8,
    )
    qnn_fp32 = np.fromfile(
        ROOT / "device_output" / "fp32" / "output.raw",
        dtype=np.float32,
    )
    qnn_u8 = np.fromfile(
        ROOT / "device_output" / "int8_per_tensor" / "output_native.raw",
        dtype=np.uint8,
    )
    qnn_dequantized = (
        qnn_u8.astype(np.float32) + parameters["offset"]
    ) * parameters["output_scale"]

    print_metrics("QNN FP16 MatMul vs FP32 reference", fp32_reference, qnn_fp32)
    print()
    print_metrics(
        "QNN INT8 MatMul vs FP32 reference",
        fp32_reference,
        qnn_dequantized,
    )
    print()
    print_metrics(
        "QNN INT8 MatMul vs NumPy quantized reference",
        quantized_reference,
        qnn_dequantized,
    )

    lsb_difference = np.abs(
        qnn_u8.astype(np.int16) - expected_u8.astype(np.int16)
    )
    print("\nRaw quantized output")
    print(f"  different:          {np.count_nonzero(lsb_difference)}")
    print(f"  max_lsb_difference: {lsb_difference.max()}")
    print(f"  within_one_lsb:     {bool(np.all(lsb_difference <= 1))}")

    per_axis_path = (
        ROOT / "device_output" / "int8_per_axis" / "output_native.raw"
    )
    if per_axis_path.exists():
        per_axis_u8 = np.fromfile(per_axis_path, dtype=np.uint8)
        per_axis_dequantized = (
            per_axis_u8.astype(np.float32) + parameters["offset"]
        ) * parameters["output_scale"]
        per_axis_reference = np.fromfile(
            ROOT / "reference" / "per_axis_dequant_output.raw",
            dtype=np.float32,
        )
        print()
        print_metrics(
            "QNN INT8 per-axis weight MatMul vs FP32 reference",
            fp32_reference,
            per_axis_dequantized,
        )
        print()
        print_metrics(
            "QNN INT8 per-axis weight MatMul vs NumPy quantized reference",
            per_axis_reference,
            per_axis_dequantized,
        )
    else:
        print("\nPer-axis QNN output: unavailable (HTP prepare failure)")


if __name__ == "__main__":
    main()
