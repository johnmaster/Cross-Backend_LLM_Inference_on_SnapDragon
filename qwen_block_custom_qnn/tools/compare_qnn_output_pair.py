#!/usr/bin/env python3
"""Compare two qnn-net-run Result_0 directories bit-for-bit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    exact = True
    for name in ["hidden_out.raw", "present_key.raw", "present_value.raw"]:
        left = np.fromfile(args.left / name, dtype=np.float32)
        right = np.fromfile(args.right / name, dtype=np.float32)
        if left.shape != right.shape:
            raise SystemExit(
                f"{name}: shape mismatch {left.shape} vs {right.shape}"
            )
        difference = np.abs(left - right)
        equal = bool(np.array_equal(left, right))
        exact &= equal
        print(
            f"{name:17s} max_abs={float(difference.max()):.8e} "
            f"mean_abs={float(difference.mean()):.8e} "
            f"array_equal={equal}"
        )
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
