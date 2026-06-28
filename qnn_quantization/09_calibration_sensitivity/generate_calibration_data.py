#!/usr/bin/env python3

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "05_quantized_matmul" / "input" / "lhs_float.raw"


def write_case(name, values):
    case_dir = ROOT / "calibration" / name
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_path = case_dir / "lhs_float.raw"
    values.astype(np.float32).tofile(raw_path)
    (case_dir / "converter_input_list.txt").write_text(
        str(raw_path.resolve()) + "\n", encoding="ascii"
    )


def main():
    representative = np.fromfile(SOURCE, np.float32).reshape(1, 1, 128, 256)
    narrow = representative * np.float32(0.25)
    outlier = representative.copy()
    outlier.reshape(-1)[0] = np.float32(8.0)

    write_case("representative", representative)
    write_case("narrow", narrow)
    write_case("outlier", outlier)

    test_dir = ROOT / "input"
    test_dir.mkdir(exist_ok=True)
    representative.tofile(test_dir / "lhs_float.raw")
    (test_dir / "input_list.txt").write_text(
        "lhs:=qnn_calibration/input/lhs_float.raw\n", encoding="ascii"
    )

    print("representative range:", representative.min(), representative.max())
    print("narrow range:        ", narrow.min(), narrow.max())
    print("outlier range:       ", outlier.min(), outlier.max())


if __name__ == "__main__":
    main()
