#!/usr/bin/env python3
"""Compare FlashAttention graph outputs with the adopted decode graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


NAMES = ("hidden_out.raw", "current_key.raw", "current_value.raw")


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    return float(np.dot(left64, right64) / (np.linalg.norm(left64) * np.linalg.norm(right64)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actual",
        type=Path,
        default=Path(
            "qnn_custom_ops/qwen_flash_attention_hvx/device_output/past128/Result_0"
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "qwen_block_custom_qnn/device_output/"
            "builtin_layer0_decode_past128_grouped_gqa_delta_kv/Result_0"
        ),
    )
    args = parser.parse_args()
    for name in NAMES:
        actual = np.fromfile(args.actual / name, dtype=np.float32)
        baseline = np.fromfile(args.baseline / name, dtype=np.float32)
        if actual.shape != baseline.shape:
            raise SystemExit(f"{name}: shape mismatch {actual.shape} != {baseline.shape}")
        difference = np.abs(actual - baseline)
        print(
            f"{name:17s} max_abs={difference.max():.8e} "
            f"mean_abs={difference.mean():.8e} "
            f"cosine={cosine(actual, baseline):.9f} "
            f"bit_equal={np.array_equal(actual, baseline)}"
        )


if __name__ == "__main__":
    main()
