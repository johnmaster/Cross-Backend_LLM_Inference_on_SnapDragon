#!/usr/bin/env python3
"""Compare q_proj custom MatMul device output with local references."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "device_output" / "q_proj_prefill" / "output.raw"
DEFAULT_EXPECTED = ROOT / "test_data" / "q_proj_prefill" / "expected_q13_kernel.raw"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    args = parser.parse_args()

    actual = np.fromfile(args.output, dtype=np.float32)
    expected = np.fromfile(args.expected, dtype=np.float32)

    if actual.shape != expected.shape:
        raise SystemExit(
            f"shape mismatch: actual {actual.shape}, expected {expected.shape}"
        )

    diff = np.abs(actual - expected)
    print("elements:", actual.size)
    print("actual range:", float(actual.min()), float(actual.max()))
    print("expected range:", float(expected.min()), float(expected.max()))
    print("nonzero:", int(np.count_nonzero(actual)))
    print("max_abs_error:", float(diff.max()))
    print("mean_abs_error:", float(diff.mean()))
    print("allclose_1e-3:", bool(np.allclose(actual, expected, atol=1e-3, rtol=1e-3)))
    print("nan_count:", int(np.isnan(actual).sum()))
    print("inf_count:", int(np.isinf(actual).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
