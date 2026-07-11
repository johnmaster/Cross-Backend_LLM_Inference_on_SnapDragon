#!/usr/bin/env python3
"""List MatMul/Gemm nodes and inferred tensor shapes from an ONNX model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


TARGET_OPS = {"MatMul", "Gemm"}


def _load_onnx():
    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "onnx is not importable. Run with the QAIRT Python packages, for example:\n"
            "PYTHONPATH=/home/lingbok/anaconda3/envs/qairt-2.47/lib/python3.12/site-packages "
            "python tiny_llm_block_custom_matmul/tools/inspect_onnx_matmuls.py "
            "tiny_llm_block/model/tiny_block_prefill_seq32.onnx"
        ) from exc
    return onnx


def _shape_from_value_info(value_info):
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None

    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(str(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return "[" + ", ".join(dims) + "]"


def _collect_shapes(graph):
    shapes = {}

    for initializer in graph.initializer:
        shapes[initializer.name] = "[" + ", ".join(str(d) for d in initializer.dims) + "]"

    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        shape = _shape_from_value_info(value_info)
        if shape is not None:
            shapes[value_info.name] = shape

    return shapes


def _infer_shapes(onnx, model):
    try:
        return onnx.shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001 - shape inference errors are still useful context.
        print(f"warning: ONNX shape inference failed: {exc}", file=sys.stderr)
        return model


def inspect_model(path: Path) -> int:
    onnx = _load_onnx()
    model = onnx.load(path)
    model = _infer_shapes(onnx, model)
    graph = model.graph
    shapes = _collect_shapes(graph)

    print(f"model: {path}")
    print(f"graph: {graph.name or '<unnamed>'}")

    nodes = [
        (index, node)
        for index, node in enumerate(graph.node)
        if node.op_type in TARGET_OPS
    ]

    print(f"matmul_or_gemm_nodes: {len(nodes)}")
    if not nodes:
        return 0

    for ordinal, (index, node) in enumerate(nodes, start=1):
        name = node.name or f"<node_{index}>"
        print()
        print(f"[{ordinal}] index={index} op={node.op_type} name={name}")
        for input_index, input_name in enumerate(node.input):
            shape = shapes.get(input_name, "unknown")
            kind = "initializer" if input_name in {i.name for i in graph.initializer} else "tensor"
            print(f"  input[{input_index}] {kind:11} {input_name} {shape}")
        for output_index, output_name in enumerate(node.output):
            shape = shapes.get(output_name, "unknown")
            print(f"  output[{output_index}] tensor      {output_name} {shape}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_model", type=Path)
    args = parser.parse_args()

    if not args.onnx_model.is_file():
        raise SystemExit(f"ONNX model not found: {args.onnx_model}")

    return inspect_model(args.onnx_model)


if __name__ == "__main__":
    raise SystemExit(main())
