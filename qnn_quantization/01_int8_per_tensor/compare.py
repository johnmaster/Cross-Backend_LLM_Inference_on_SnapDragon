#!/usr/bin/env python3

import numpy as np


def symmetric_int8(values):
    scale = float(np.max(np.abs(values))) / 127.0
    if scale == 0.0:
        scale = 1.0
    quantized = np.clip(np.rint(values / scale), -128, 127).astype(np.int8)
    dequantized = quantized.astype(np.float32) * scale
    saturated = int(np.count_nonzero((quantized == -128) | (quantized == 127)))
    return quantized, dequantized, scale, 0, saturated


def asymmetric_uint8(values):
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    scale = (maximum - minimum) / 255.0
    if scale == 0.0:
        scale = 1.0
    zero_point = int(np.clip(np.rint(-minimum / scale), 0, 255))
    quantized = np.clip(
        np.rint(values / scale) + zero_point, 0, 255
    ).astype(np.uint8)
    dequantized = (
        quantized.astype(np.float32) - zero_point
    ) * scale
    saturated = int(np.count_nonzero((quantized == 0) | (quantized == 255)))
    return quantized, dequantized, scale, zero_point, saturated


def metrics(reference, result):
    difference = result - reference
    signal_power = float(np.sum(reference.astype(np.float64) ** 2))
    noise_power = float(np.sum(difference.astype(np.float64) ** 2))
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(result))
    cosine = float(np.dot(reference, result) / denominator) if denominator else 1.0
    sqnr = float("inf") if noise_power == 0.0 else 10.0 * np.log10(
        signal_power / noise_power
    )
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "cosine_similarity": cosine,
        "sqnr_db": sqnr,
    }


def print_result(name, reference, quantized, dequantized, scale,
                 zero_point, saturated):
    result = metrics(reference, dequantized)
    print(f"  {name}")
    print(f"    dtype:              {quantized.dtype}")
    print(f"    scale:              {scale:.10g}")
    print(f"    zero_point:         {zero_point}")
    print(f"    qnn_offset:         {-zero_point}")
    print(f"    max_abs_error:      {result['max_abs_error']:.10g}")
    print(f"    mean_abs_error:     {result['mean_abs_error']:.10g}")
    print(f"    cosine_similarity:  {result['cosine_similarity']:.10g}")
    print(f"    sqnr_db:            {result['sqnr_db']:.4f}")
    print(f"    saturated_elements: {saturated}")
    print(f"    storage_bytes:      {quantized.nbytes}")


def compare_distribution(name, values):
    values = values.astype(np.float32)
    print(f"\n{name}")
    print(f"  range: [{float(values.min()):.6f}, {float(values.max()):.6f}]")
    print(f"  fp32_storage_bytes: {values.nbytes}")

    symmetric = symmetric_int8(values)
    asymmetric = asymmetric_uint8(values)
    print_result("symmetric INT8", values, *symmetric)
    print_result("asymmetric UINT8", values, *asymmetric)


def main():
    rng = np.random.default_rng(20260627)
    centered = rng.normal(0.0, 0.35, 32768)
    shifted = rng.normal(0.8, 0.2, 32768)

    compare_distribution("Centered distribution", centered)
    compare_distribution("Shifted distribution", shifted)


if __name__ == "__main__":
    main()
