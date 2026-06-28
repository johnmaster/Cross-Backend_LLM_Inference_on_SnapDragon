#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LHS_SHAPE = (1, 1, 128, 256)
RHS_SHAPE = (1, 1, 256, 256)


def symmetric_uint8(values, scale):
    signed = np.clip(np.rint(values / scale), -128, 127).astype(np.int8)
    stored = (signed.astype(np.int16) + 128).astype(np.uint8)
    dequantized = signed.astype(np.float32) * scale
    return stored, dequantized


def symmetric_int8(values, scale):
    signed = np.clip(np.rint(values / scale), -128, 127).astype(np.int8)
    dequantized = signed.astype(np.float32) * scale
    return signed, dequantized


def main():
    rng = np.random.default_rng(20260627)
    lhs = rng.uniform(-0.5, 0.5, LHS_SHAPE).astype(np.float32)

    # Give output columns different ranges so per-axis weight quantization matters.
    channel_range = np.geomspace(0.02, 0.5, RHS_SHAPE[-1]).astype(np.float32)
    rhs = (
        rng.uniform(-1.0, 1.0, RHS_SHAPE).astype(np.float32)
        * channel_range.reshape(1, 1, 1, -1)
    )

    lhs_scale = float(np.max(np.abs(lhs))) / 127.0
    rhs_tensor_scale = float(np.max(np.abs(rhs))) / 127.0
    rhs_axis_scales = channel_range / 127.0

    lhs_u8, lhs_dq = symmetric_uint8(lhs, lhs_scale)
    rhs_tensor_u8, rhs_tensor_dq = symmetric_uint8(rhs, rhs_tensor_scale)
    rhs_axis_i8, rhs_axis_dq = symmetric_int8(
        rhs, rhs_axis_scales.reshape(1, 1, 1, -1)
    )

    fp32_reference = np.matmul(lhs, rhs).astype(np.float32)
    per_tensor_reference = np.matmul(lhs_dq, rhs_tensor_dq).astype(np.float32)
    per_axis_reference = np.matmul(lhs_dq, rhs_axis_dq).astype(np.float32)

    output_scale = float(np.max(np.abs(fp32_reference))) / 127.0
    per_tensor_output_u8, _ = symmetric_uint8(
        per_tensor_reference, output_scale
    )
    per_axis_output_u8, _ = symmetric_uint8(per_axis_reference, output_scale)

    for directory in ("input", "reference", "model_bin"):
        (ROOT / directory).mkdir(exist_ok=True)

    (ROOT / "input" / "rhs_per_axis_uint8.raw").unlink(missing_ok=True)
    lhs.tofile(ROOT / "input" / "lhs_float.raw")
    rhs.tofile(ROOT / "input" / "rhs_float.raw")
    lhs_u8.tofile(ROOT / "input" / "lhs_uint8.raw")
    rhs_tensor_u8.tofile(ROOT / "input" / "rhs_per_tensor_uint8.raw")
    rhs_axis_i8.tofile(ROOT / "input" / "rhs_per_axis_int8.raw")
    rhs_axis_i8.tofile(ROOT / "model_bin" / "rhs_per_axis_weight.raw")
    rhs_axis_scales.astype(np.float32).tofile(
        ROOT / "reference" / "rhs_axis_scales.raw"
    )
    fp32_reference.tofile(ROOT / "reference" / "fp32_output.raw")
    per_tensor_reference.tofile(
        ROOT / "reference" / "per_tensor_dequant_output.raw"
    )
    per_axis_reference.tofile(
        ROOT / "reference" / "per_axis_dequant_output.raw"
    )
    per_tensor_output_u8.tofile(
        ROOT / "reference" / "per_tensor_output_uint8.raw"
    )
    per_axis_output_u8.tofile(
        ROOT / "reference" / "per_axis_output_uint8.raw"
    )

    (ROOT / "input" / "fp32_input_list.txt").write_text(
        "lhs:=qnn_quantized_matmul/input/lhs_float.raw "
        "rhs:=qnn_quantized_matmul/input/rhs_float.raw\n",
        encoding="ascii",
    )
    (ROOT / "input" / "per_tensor_input_list.txt").write_text(
        "lhs:=qnn_quantized_matmul/input/lhs_uint8.raw "
        "rhs:=qnn_quantized_matmul/input/rhs_per_tensor_uint8.raw\n",
        encoding="ascii",
    )
    (ROOT / "input" / "per_axis_input_list.txt").write_text(
        "lhs:=qnn_quantized_matmul/input/lhs_uint8.raw\n",
        encoding="ascii",
    )

    parameters = {
        "lhs_scale": lhs_scale,
        "rhs_per_tensor_scale": rhs_tensor_scale,
        "rhs_per_axis_scale_min": float(rhs_axis_scales.min()),
        "rhs_per_axis_scale_max": float(rhs_axis_scales.max()),
        "output_scale": output_scale,
        "offset": -128,
        "weight_axis": 3,
    }
    (ROOT / "parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(parameters, indent=2))


if __name__ == "__main__":
    main()
