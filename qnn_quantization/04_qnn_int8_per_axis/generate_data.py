#!/usr/bin/env python3

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
K = 256
N = 64


def main():
    channel_range = np.geomspace(0.02, 1.0, N).astype(np.float32)
    row_pattern = np.linspace(-1.0, 1.0, K, dtype=np.float32)[:, None]
    weights = row_pattern * channel_range[None, :]

    # HTP graph precision is FP16, so model the input rounding before Quantize.
    htp_weights = weights.astype(np.float16).astype(np.float32)

    per_tensor_scale = 1.0 / 127.0
    per_tensor_q = np.clip(
        np.rint(htp_weights / per_tensor_scale), -128, 127
    ).astype(np.int8)
    per_tensor_storage = (
        per_tensor_q.astype(np.int16) + 128
    ).astype(np.uint8)

    per_axis_scale = channel_range / 127.0
    per_axis_q = np.clip(
        np.rint(htp_weights / per_axis_scale[None, :]), -128, 127
    ).astype(np.int8)
    per_axis_storage = (
        per_axis_q.astype(np.int16) + 128
    ).astype(np.uint8)

    (ROOT / "input").mkdir(exist_ok=True)
    (ROOT / "reference").mkdir(exist_ok=True)
    weights.tofile(ROOT / "input" / "weights_float.raw")
    per_tensor_storage.tofile(ROOT / "reference" / "per_tensor_uint8.raw")
    (
        per_tensor_q.astype(np.float32) * per_tensor_scale
    ).tofile(ROOT / "reference" / "per_tensor_dequant.raw")
    per_axis_storage.tofile(ROOT / "reference" / "per_axis_uint8.raw")
    (
        per_axis_q.astype(np.float32) * per_axis_scale[None, :]
    ).tofile(ROOT / "reference" / "per_axis_dequant.raw")
    per_axis_scale.tofile(ROOT / "reference" / "per_axis_scales.raw")
    (ROOT / "input" / "input_list.txt").write_text(
        "weights:=qnn_int8_per_axis/input/weights_float.raw\n",
        encoding="ascii",
    )

    print(f"shape: {weights.shape}")
    print(f"per_tensor_scale: {per_tensor_scale:.10g}")
    print(
        "per_axis_scale_range: "
        f"[{float(per_axis_scale.min()):.10g}, "
        f"{float(per_axis_scale.max()):.10g}]"
    )


if __name__ == "__main__":
    main()
