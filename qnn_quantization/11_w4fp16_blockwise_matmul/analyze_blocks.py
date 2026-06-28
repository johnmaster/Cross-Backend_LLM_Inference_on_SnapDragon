#!/usr/bin/env python3

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "05_quantized_matmul"


def metrics(reference, result):
    difference = result - reference
    signal = np.sum(reference.astype(np.float64) ** 2)
    noise = np.sum(difference.astype(np.float64) ** 2)
    cosine = np.dot(reference.ravel(), result.ravel()) / (
        np.linalg.norm(reference) * np.linalg.norm(result)
    )
    return (
        np.max(np.abs(difference)),
        np.mean(np.abs(difference)),
        cosine,
        10.0 * np.log10(signal / noise),
    )


def block_quantize(rhs, block_size):
    block_count = rhs.shape[2] // block_size
    blocked = rhs.reshape(1, 1, block_count, block_size, rhs.shape[3])
    scales = np.max(np.abs(blocked), axis=3) / np.float32(7.0)
    scales = np.maximum(scales, np.finfo(np.float16).tiny).astype(np.float16)
    quantized = np.clip(
        np.rint(blocked / scales.astype(np.float32)[:, :, :, None, :]),
        -8,
        7,
    ).astype(np.int8)
    dequantized = (
        quantized.astype(np.float32)
        * scales.astype(np.float32)[:, :, :, None, :]
    )
    return dequantized.reshape(rhs.shape), scales


def main():
    lhs = np.fromfile(SOURCE / "input/lhs_float.raw", np.float32).reshape(
        1, 1, 128, 256
    )
    rhs = np.fromfile(SOURCE / "input/rhs_float.raw", np.float32).reshape(
        1, 1, 256, 256
    )
    reference = np.matmul(lhs, rhs).astype(np.float32)
    lhs_fp16 = lhs.astype(np.float16).astype(np.float32)

    print(
        f"{'block':>7s} {'weight KiB':>11s} {'ratio':>8s} "
        f"{'mean error':>12s} {'cosine':>12s} {'SQNR(dB)':>10s}"
    )
    fp16_bytes = rhs.size * 2
    for block_size in (32, 64, 128):
        rhs_dq, scales = block_quantize(rhs, block_size)
        output = np.matmul(lhs_fp16, rhs_dq).astype(np.float16).astype(np.float32)
        _, mean, cosine, sqnr = metrics(reference, output)
        packed_bytes = rhs.size // 2
        scale_bytes = scales.size * 2
        total_bytes = packed_bytes + scale_bytes
        print(
            f"{block_size:7d} {total_bytes / 1024:11.2f} "
            f"{fp16_bytes / total_bytes:8.2f}x {mean:12.7f} "
            f"{cosine:12.9f} {sqnr:10.3f}"
        )


if __name__ == "__main__":
    main()
