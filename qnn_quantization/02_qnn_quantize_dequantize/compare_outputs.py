#!/usr/bin/env python3

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def compare_quantized(name, device_file, reference_file):
    device = np.fromfile(device_file, dtype=np.uint8)
    reference = np.fromfile(reference_file, dtype=np.uint8)
    difference = np.abs(device.astype(np.int16) - reference.astype(np.int16))
    print(name)
    print(f"  elements:          {device.size}")
    print(f"  different:         {np.count_nonzero(difference)}")
    print(f"  max_lsb_difference:{difference.max()}")
    print(f"  within_one_lsb:    {bool(np.all(difference <= 1))}")


def main():
    output = ROOT / "device_output" / "Result_0"
    reference = ROOT / "reference"
    compare_quantized(
        "Symmetric UINT8 storage",
        output / "symmetric_int8_native.raw",
        reference / "symmetric_quantized.raw",
    )
    compare_quantized(
        "Asymmetric UINT8",
        output / "asymmetric_uint8_native.raw",
        reference / "asymmetric_quantized.raw",
    )


if __name__ == "__main__":
    main()
