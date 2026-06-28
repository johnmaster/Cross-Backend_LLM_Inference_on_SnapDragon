#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def compare(name: str, expected_path: Path, actual_path: Path) -> None:
    expected = np.fromfile(expected_path, dtype=np.float32)
    actual = np.fromfile(actual_path, dtype=np.float32)
    if expected.shape != actual.shape:
        raise ValueError(
            f"{name}: expected {expected.size} values, got {actual.size}"
        )
    difference = actual - expected
    cosine = np.dot(expected, actual) / (
        np.linalg.norm(expected) * np.linalg.norm(actual)
    )
    print(
        f"{name}: max={np.max(np.abs(difference)):.8e} "
        f"mean={np.mean(np.abs(difference)):.8e} cosine={cosine:.9f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all QNN block outputs.")
    parser.add_argument("mode", choices=("prefill", "decode"))
    parser.add_argument(
        "output_dir", type=Path, help="QNN Result_0 output directory"
    )
    args = parser.parse_args()
    expected_dir = ROOT / "test_data"
    compare(
        f"{args.mode} hidden",
        expected_dir / f"{args.mode}_expected.raw",
        args.output_dir / "hidden_out.raw",
    )
    compare(
        f"{args.mode} key",
        expected_dir / f"{args.mode}_expected_key.raw",
        args.output_dir / "present_key.raw",
    )
    compare(
        f"{args.mode} value",
        expected_dir / f"{args.mode}_expected_value.raw",
        args.output_dir / "present_value.raw",
    )


if __name__ == "__main__":
    main()
