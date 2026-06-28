#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def main():
    rng = np.random.default_rng(20260627)
    rng.normal(0.0, 0.35, 32768)
    values = rng.normal(0.8, 0.2, 32768).astype(np.float32)
    htp_values = values.astype(np.float16).astype(np.float32)

    symmetric_scale = float(np.max(np.abs(values))) / 127.0
    symmetric_q = np.clip(
        np.rint(htp_values / symmetric_scale), -128, 127
    ).astype(np.int8)
    symmetric_storage = (
        symmetric_q.astype(np.int16) + 128
    ).astype(np.uint8)
    symmetric_dq = symmetric_q.astype(np.float32) * symmetric_scale

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    asymmetric_scale = (maximum - minimum) / 255.0
    zero_point = int(np.clip(np.rint(-minimum / asymmetric_scale), 0, 255))
    asymmetric_q = np.clip(
        np.rint(htp_values / asymmetric_scale) + zero_point, 0, 255
    ).astype(np.uint8)
    asymmetric_dq = (
        asymmetric_q.astype(np.float32) - zero_point
    ) * asymmetric_scale

    (ROOT / "input").mkdir(exist_ok=True)
    (ROOT / "reference").mkdir(exist_ok=True)
    values.tofile(ROOT / "input" / "shifted_float.raw")
    symmetric_dq.tofile(ROOT / "reference" / "symmetric_dequant.raw")
    symmetric_storage.tofile(ROOT / "reference" / "symmetric_quantized.raw")
    asymmetric_dq.tofile(ROOT / "reference" / "asymmetric_dequant.raw")
    asymmetric_q.tofile(ROOT / "reference" / "asymmetric_quantized.raw")
    (ROOT / "input" / "input_list.txt").write_text(
        "input:=qnn_quantize_dequantize/input/shifted_float.raw\n",
        encoding="ascii",
    )

    parameters = {
        "elements": int(values.size),
        "minimum": minimum,
        "maximum": maximum,
        "symmetric_scale": symmetric_scale,
        "symmetric_qnn_offset": -128,
        "asymmetric_scale": asymmetric_scale,
        "asymmetric_zero_point": zero_point,
        "asymmetric_qnn_offset": -zero_point,
    }
    (ROOT / "parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(parameters, indent=2))


if __name__ == "__main__":
    main()
