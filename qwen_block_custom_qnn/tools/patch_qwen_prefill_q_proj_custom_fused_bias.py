#!/usr/bin/env python3
"""Patch Qwen layer0 prefill q_proj to use custom HTP MatMul with fused bias."""

from __future__ import annotations

import argparse
from pathlib import Path

from patch_qwen_prefill_q_proj_custom import (
    NEW_Q_PROJ_NODE as BASE_CUSTOM_Q_PROJ_NODE,
    NEW_WEIGHT_DIMS,
    NEW_WEIGHT_RANK,
    OLD_Q_PROJ_NODE,
    OLD_WEIGHT_DIMS,
    OLD_WEIGHT_RANK,
    replace_once,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "generated" / "qwen2_0_5b_layer0_prefill_seq16.cpp"
DEFAULT_OUTPUT = ROOT / "generated" / "qwen2_0_5b_layer0_prefill_seq16_q_proj_custom_fused_bias.cpp"


def fused_bias_q_proj_node() -> str:
    node = BASE_CUSTOM_Q_PROJ_NODE
    node = node.replace(
        """  const char* inputs__MatMul_custom[] = {
    "_MatMul_lhs_fp16",
    "_MatMul_rhs_fp16"
  };""",
        """  const char* inputs__MatMul_bias_reshape[] = {
    "self_attn_q_proj_bias"
  };
  uint32_t dimensions__MatMul_q_proj_bias_4d[] = {1, 1, 1, 896};
  Qnn_Tensor_t outputs__MatMul_bias_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_q_proj_bias_4d",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_q_proj_bias_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_bias_reshape",
                         "qti.aisw",
                         "Reshape",
                         nullptr,
                         0,
                         inputs__MatMul_bias_reshape,
                         1,
                         outputs__MatMul_bias_reshape,
                         1), err);

  const char* inputs__MatMul_custom[] = {
    "_MatMul_lhs_fp16",
    "_MatMul_rhs_fp16",
    "_MatMul_q_proj_bias_4d"
  };""",
        1,
    )
    node = node.replace(
        """                         inputs__MatMul_custom,
                         2,
                         outputs__MatMul_custom,""",
        """                         inputs__MatMul_custom,
                         3,
                         outputs__MatMul_custom,""",
        1,
    )
    node = node.replace('            .name= "_MatMul_custom_output_2d",', '            .name= "_Add_1_output_0_fc",', 1)

    bias_add_start = node.index("  Qnn_Param_t params__MatMul_bias_add[] = {")
    return node[:bias_add_start] + "  return err;\n}\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    text = replace_once(text, OLD_WEIGHT_DIMS, NEW_WEIGHT_DIMS, "q_proj weight dims")

    start = text.index(NEW_WEIGHT_DIMS)
    end = text.index("static ModelError_t addTensor_self_attn_q_proj_bias", start)
    section = text[start:end]
    section = replace_once(section, OLD_WEIGHT_RANK, NEW_WEIGHT_RANK, "q_proj weight rank")
    text = text[:start] + section + text[end:]

    text = replace_once(text, OLD_Q_PROJ_NODE, fused_bias_q_proj_node(), "q_proj FullyConnected node")
    text = text.replace(
        "/* Command Line used:",
        "/* Patched by qwen_block_custom_qnn/tools/patch_qwen_prefill_q_proj_custom_fused_bias.py\n"
        " * Replaces layer0 q_proj FullyConnected with MatMulQhpiHvx8RowLhsTileCacheFp32Store.\n"
        " * The q_proj bias is passed as the third custom-op input and fused into the output store.\n"
        " */\n\n/* Command Line used:",
        1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
