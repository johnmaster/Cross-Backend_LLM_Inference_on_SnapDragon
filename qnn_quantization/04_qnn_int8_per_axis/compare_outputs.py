#!/usr/bin/env python3

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def compare(name, output_name, reference_name):
    output = np.fromfile(
        ROOT / "device_output" / "Result_0" / output_name,
        dtype=np.float32,
    )
    reference = np.fromfile(
        ROOT / "reference" / reference_name,
        dtype=np.float32,
    )
    difference = np.abs(output - reference)
    print(name)
    print(f"  elements:           {output.size}")
    print(f"  max_abs_difference: {float(difference.max()):.10g}")
    print(f"  mean_difference:    {float(difference.mean()):.10g}")


def main():
    compare(
        "Per-tensor",
        "per_tensor_output.raw",
        "per_tensor_dequant.raw",
    )
    compare(
        "Per-axis",
        "per_axis_output.raw",
        "per_axis_dequant.raw",
    )


if __name__ == "__main__":
    main()
