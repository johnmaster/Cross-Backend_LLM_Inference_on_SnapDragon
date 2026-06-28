#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "05_quantized_matmul"
RHS_SHAPE = (1, 1, 256, 256)


def packed_int4_tensor(name, values):
    flat = values.astype(np.int8).reshape(-1)
    nibbles = (flat.astype(np.int16) & 0xF).astype(np.uint8)
    packed = nibbles[0::2] | (nibbles[1::2] << 4)
    tensor = TensorProto()
    tensor.name = name
    tensor.data_type = TensorProto.INT4
    tensor.dims.extend(values.shape)
    tensor.raw_data = packed.tobytes()
    return tensor


def main():
    rhs = np.fromfile(SOURCE / "input" / "rhs_float.raw", np.float32).reshape(
        RHS_SHAPE
    )
    block_size = 32
    blocked = rhs.reshape(1, 1, 8, block_size, 256)
    scales = np.max(np.abs(blocked), axis=3) / np.float32(7.0)
    scales = np.maximum(scales, np.finfo(np.float16).tiny).astype(np.float16)
    quantized = np.clip(
        np.rint(blocked / scales.astype(np.float32)[:, :, :, None, :]),
        -8,
        7,
    ).astype(np.int8).reshape(RHS_SHAPE)

    initializers = [
        packed_int4_tensor("rhs_int4", quantized),
        numpy_helper.from_array(scales, name="rhs_block_scales"),
    ]
    nodes = [
        helper.make_node(
            "Cast", ["lhs"], ["lhs_fp16"], name="LhsToFp16", to=TensorProto.FLOAT16
        ),
        helper.make_node(
            "DequantizeLinear",
            ["rhs_int4", "rhs_block_scales"],
            ["rhs_fp16"],
            name="RhsInt4BlockDequantize",
            axis=2,
            block_size=block_size,
        ),
        helper.make_node(
            "MatMul", ["lhs_fp16", "rhs_fp16"], ["output_fp16"], name="MatMul"
        ),
        helper.make_node(
            "Cast",
            ["output_fp16"],
            ["output"],
            name="OutputToFp32",
            to=TensorProto.FLOAT,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "W4Fp16BlockMatMul",
        [
            helper.make_tensor_value_info(
                "lhs", TensorProto.FLOAT, [1, 1, 128, 256]
            )
        ],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [1, 1, 128, 256]
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21)],
        producer_name="qnn_quantization_learning",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    path = model_dir / "w4fp16_block32_qdq.onnx"
    onnx.save(model, path)
    print(f"wrote {path}")
    print(f"packed weight bytes: {len(initializers[0].raw_data)}")
    print(f"scale elements: {scales.size}")


if __name__ == "__main__":
    main()
