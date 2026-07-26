#!/usr/bin/env python3
"""Compare hidden/current-KV outputs from a host-managed decode-cache graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--current-key-nhwc",
        action="store_true",
        help="convert QNN [1,1,64,2] current-key output to ONNX NCHW",
    )
    parser.add_argument(
        "--current-value-nhwc",
        action="store_true",
        help="convert QNN [1,1,64,2] current-value output to ONNX NCHW",
    )
    parser.add_argument(
        "--fp16-kv",
        action="store_true",
        help="read current key/value as FP16 while hidden remains FP32",
    )
    args = parser.parse_args()

    for name in ("hidden_out", "current_key", "current_value"):
        dtype = np.float16 if args.fp16_kv and name != "hidden_out" else np.float32
        expected = np.fromfile(args.reference / f"{name}.raw", dtype=dtype)
        output_path = args.output / f"{name}.raw"
        if args.fp16_kv and name != "hidden_out":
            native_path = args.output / f"{name}_native.raw"
            if native_path.exists():
                output_path = native_path
        actual = np.fromfile(output_path, dtype=dtype)
        if args.current_key_nhwc and name == "current_key":
            actual = (
                actual.reshape(1, 1, 64, 2)
                .transpose(0, 3, 1, 2)
                .copy()
                .reshape(-1)
            )
        if args.current_value_nhwc and name == "current_value":
            actual = (
                actual.reshape(1, 1, 64, 2)
                .transpose(0, 3, 1, 2)
                .copy()
                .reshape(-1)
            )
        if expected.shape != actual.shape:
            raise SystemExit(f"{name}: shape mismatch {expected.shape} vs {actual.shape}")
        expected_f32 = expected.astype(np.float32)
        actual_f32 = actual.astype(np.float32)
        diff = np.abs(expected_f32 - actual_f32)
        print(
            f"{name:14s} max_abs={float(diff.max()):.8e} "
            f"mean_abs={float(diff.mean()):.8e} "
            f"cosine={cosine(expected_f32, actual_f32):.9f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
