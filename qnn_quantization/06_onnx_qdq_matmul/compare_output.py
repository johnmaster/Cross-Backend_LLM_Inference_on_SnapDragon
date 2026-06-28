#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "05_quantized_matmul"


def print_metrics(name, reference, result):
    difference = result - reference
    signal_power = np.sum(reference.astype(np.float64) ** 2)
    noise_power = np.sum(difference.astype(np.float64) ** 2)
    cosine = np.dot(reference, result) / (
        np.linalg.norm(reference) * np.linalg.norm(result)
    )
    print(name)
    print(f"  max_abs_error:     {np.max(np.abs(difference)):.10g}")
    print(f"  mean_abs_error:    {np.mean(np.abs(difference)):.10g}")
    print(f"  cosine_similarity: {cosine:.10g}")
    print(f"  sqnr_db:           {10.0 * np.log10(signal_power / noise_power):.10g}")


def main():
    fp32 = np.fromfile(
        BASELINE / "reference" / "fp32_output.raw", np.float32
    )
    numpy_per_axis = np.fromfile(
        BASELINE / "reference" / "per_axis_dequant_output.raw", np.float32
    )
    qnn_per_axis = np.fromfile(
        ROOT / "device_output" / "output_float.raw", np.float32
    )

    parameters = json.loads((BASELINE / "parameters.json").read_text())
    per_tensor_u8 = np.fromfile(
        BASELINE
        / "device_output"
        / "int8_per_tensor"
        / "output_native.raw",
        np.uint8,
    )
    qnn_per_tensor = (
        per_tensor_u8.astype(np.float32) + parameters["offset"]
    ) * parameters["output_scale"]

    print_metrics("QNN converter per-axis vs FP32", fp32, qnn_per_axis)
    print()
    print_metrics("NumPy per-axis vs FP32", fp32, numpy_per_axis)
    print()
    print_metrics("QNN per-tensor vs FP32", fp32, qnn_per_tensor)


if __name__ == "__main__":
    main()
