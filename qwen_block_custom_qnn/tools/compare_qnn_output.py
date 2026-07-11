#!/usr/bin/env python3
"""Compare Qwen block QNN raw outputs with exported reference raw files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "test_data" / "layer0_prefill_seq16"
DEFAULT_OUTPUT = ROOT / "device_output" / "builtin_layer0_prefill_seq16" / "Result_0"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for name in ["hidden_out.raw", "present_key.raw", "present_value.raw"]:
        expected = np.fromfile(args.reference / name, dtype=np.float32)
        actual = np.fromfile(args.output / name, dtype=np.float32)
        if expected.shape != actual.shape:
            raise SystemExit(f"{name}: shape mismatch {expected.shape} vs {actual.shape}")
        diff = np.abs(expected - actual)
        print(
            f"{name:17s} "
            f"shape={expected.shape!s:10s} "
            f"max_abs={float(diff.max()):.8e} "
            f"mean_abs={float(diff.mean()):.8e} "
            f"cosine={cosine(expected, actual):.9f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
