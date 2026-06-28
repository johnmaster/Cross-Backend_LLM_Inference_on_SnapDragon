#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "05_quantized_matmul"
MODEL_PATH = ROOT / "model" / "int8_per_axis_qdq_matmul.onnx"
FLOAT_MODEL_PATH = ROOT / "model" / "fp32_matmul.onnx"


def initializer(name, values):
    return numpy_helper.from_array(np.asarray(values), name=name)


def main():
    lhs = np.fromfile(SOURCE / "input" / "lhs_float.raw", np.float32).reshape(
        1, 1, 128, 256
    )
    rhs = np.fromfile(SOURCE / "input" / "rhs_float.raw", np.float32).reshape(
        1, 1, 256, 256
    )
    fp32_reference = np.matmul(lhs, rhs).astype(np.float32)

    lhs_scale = np.asarray([np.max(np.abs(lhs)) / 127.0], np.float32)
    rhs_ranges = np.geomspace(0.02, 0.5, 256).astype(np.float32)
    rhs_scales = rhs_ranges / np.float32(127.0)
    output_scale = np.asarray(
        [np.max(np.abs(fp32_reference)) / 127.0], np.float32
    )

    rhs_quantized = np.clip(
        np.rint(rhs / rhs_scales.reshape(1, 1, 1, -1)), -128, 127
    ).astype(np.int8)

    initializers = [
        initializer("lhs_scale", lhs_scale),
        initializer("lhs_zero_point", np.asarray([128], np.uint8)),
        initializer("rhs_quantized", rhs_quantized),
        initializer("rhs_scales", rhs_scales),
        initializer("rhs_zero_points", np.zeros(256, np.int8)),
        initializer("output_scale", output_scale),
        initializer("output_zero_point", np.asarray([128], np.uint8)),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["lhs", "lhs_scale", "lhs_zero_point"],
            ["lhs_quantized"],
            name="LhsQuantize",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["lhs_quantized", "lhs_scale", "lhs_zero_point"],
            ["lhs_dequantized"],
            name="LhsDequantize",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["rhs_quantized", "rhs_scales", "rhs_zero_points"],
            ["rhs_dequantized"],
            name="RhsPerAxisDequantize",
            axis=3,
        ),
        helper.make_node(
            "MatMul",
            ["lhs_dequantized", "rhs_dequantized"],
            ["matmul_float"],
            name="PerAxisWeightMatMul",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["matmul_float", "output_scale", "output_zero_point"],
            ["output_quantized"],
            name="OutputQuantize",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["output_quantized", "output_scale", "output_zero_point"],
            ["output"],
            name="OutputDequantize",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "Int8PerAxisQdqMatMul",
        [helper.make_tensor_value_info("lhs", TensorProto.FLOAT, lhs.shape)],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, fp32_reference.shape
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="qnn_quantization_learning",
    )
    model.ir_version = 9
    onnx.checker.check_model(model)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, MODEL_PATH)

    float_graph = helper.make_graph(
        [
            helper.make_node(
                "MatMul", ["lhs", "rhs"], ["output"], name="MatMul"
            )
        ],
        "Fp32MatMul",
        [helper.make_tensor_value_info("lhs", TensorProto.FLOAT, lhs.shape)],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, fp32_reference.shape
            )
        ],
        initializer=[initializer("rhs", rhs)],
    )
    float_model = helper.make_model(
        float_graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="qnn_quantization_learning",
    )
    float_model.ir_version = 9
    onnx.checker.check_model(float_model)
    onnx.save(float_model, FLOAT_MODEL_PATH)

    encodings = []
    for channel_max in rhs_ranges:
        value = float(channel_max)
        encodings.append(
            {
                "bitwidth": 8,
                "is_symmetric": "True",
                "min": -value,
                "max": value,
            }
        )
    overrides = {
        "activation_encodings": {},
        "param_encodings": {"rhs": encodings},
    }
    (ROOT / "model" / "quantization_overrides.json").write_text(
        json.dumps(overrides, indent=2) + "\n", encoding="ascii"
    )

    (ROOT / "input").mkdir(exist_ok=True)
    lhs.tofile(ROOT / "input" / "lhs_float.raw")
    (ROOT / "input" / "input_list.txt").write_text(
        "lhs:=qnn_qdq_matmul/input/lhs_float.raw\n", encoding="ascii"
    )
    (ROOT / "input" / "converter_input_list.txt").write_text(
        str((ROOT / "input" / "lhs_float.raw").resolve()) + "\n",
        encoding="ascii",
    )
    print(f"wrote {MODEL_PATH}")
    print(f"wrote {FLOAT_MODEL_PATH}")


if __name__ == "__main__":
    main()
