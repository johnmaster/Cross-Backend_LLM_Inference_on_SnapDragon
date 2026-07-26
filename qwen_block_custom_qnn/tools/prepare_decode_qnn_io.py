#!/usr/bin/env python3
"""Prepare device-native NHWC KV inputs for converter-generated decode graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_data", type=Path)
    parser.add_argument("--past-len", type=int, required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    for name in ("past_key", "past_value"):
        dtype = np.float16 if args.fp16 else np.float32
        source = np.fromfile(args.test_data / f"{name}.raw", dtype=dtype)
        nchw = source.reshape(1, 2, args.past_len, 64)
        nhwc = nchw.transpose(0, 2, 3, 1).copy()
        output = args.test_data / f"{name}_qnn_nhwc.raw"
        nhwc.tofile(output)
        print(f"Wrote {output}: {nhwc.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
