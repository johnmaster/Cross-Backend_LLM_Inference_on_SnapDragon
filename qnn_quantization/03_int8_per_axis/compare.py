#!/usr/bin/env python3

import numpy as np


K = 256
N = 64
AXIS = 1


def quantize_symmetric(values, scale):
    quantized = np.clip(np.rint(values / scale), -128, 127).astype(np.int8)
    return quantized, quantized.astype(np.float32) * scale


def metrics(reference, result):
    difference = result - reference
    signal = np.sum(reference.astype(np.float64) ** 2)
    noise = np.sum(difference.astype(np.float64) ** 2)
    cosine_denominator = np.linalg.norm(reference) * np.linalg.norm(result)
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "cosine_similarity": float(
            np.dot(reference.ravel(), result.ravel()) / cosine_denominator
        ),
        "sqnr_db": float(10.0 * np.log10(signal / noise)),
    }


def print_metrics(name, result):
    print(name)
    for key, value in result.items():
        print(f"  {key:20s}: {value:.10g}")


def main():
    rng = np.random.default_rng(20260627)

    # Different output columns intentionally have very different ranges.
    channel_std = np.geomspace(0.02, 1.0, N).astype(np.float32)
    weights = (
        rng.normal(0.0, 1.0, (K, N)).astype(np.float32) * channel_std
    )

    per_tensor_scale = float(np.max(np.abs(weights))) / 127.0
    _, per_tensor_dequant = quantize_symmetric(weights, per_tensor_scale)

    per_axis_scale = np.max(np.abs(weights), axis=0) / 127.0
    per_axis_scale = np.where(per_axis_scale == 0.0, 1.0, per_axis_scale)
    _, per_axis_dequant = quantize_symmetric(weights, per_axis_scale)

    print(f"weight_shape: {weights.shape}")
    print(f"qnn_axis: {AXIS}")
    print(f"per_tensor_scale: {per_tensor_scale:.10g}")
    print(
        "per_axis_scale_range: "
        f"[{float(per_axis_scale.min()):.10g}, "
        f"{float(per_axis_scale.max()):.10g}]"
    )
    print(f"fp32_storage_bytes: {weights.nbytes}")
    print(f"int8_storage_bytes: {weights.size}")
    print(f"per_axis_scale_bytes: {per_axis_scale.nbytes}\n")

    tensor_metrics = metrics(weights, per_tensor_dequant)
    axis_metrics = metrics(weights, per_axis_dequant)
    print_metrics("Per-tensor symmetric INT8", tensor_metrics)
    print()
    print_metrics("Per-axis symmetric INT8", axis_metrics)

    tensor_channel_mae = np.mean(np.abs(per_tensor_dequant - weights), axis=0)
    axis_channel_mae = np.mean(np.abs(per_axis_dequant - weights), axis=0)
    improvement = tensor_channel_mae / np.maximum(axis_channel_mae, 1e-30)
    print("\nPer-channel MAE improvement")
    print(f"  minimum: {float(improvement.min()):.4f}x")
    print(f"  median:  {float(np.median(improvement)):.4f}x")
    print(f"  maximum: {float(improvement.max()):.4f}x")


if __name__ == "__main__":
    main()
