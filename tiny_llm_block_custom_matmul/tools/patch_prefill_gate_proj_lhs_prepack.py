#!/usr/bin/env python3
"""Patch prefill QNN source to use LHS-prepack custom gate_proj MatMul."""

from __future__ import annotations

import argparse
from pathlib import Path

from patch_prefill_q_proj_custom import DEFAULT_INPUT, replace_once


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "generated" / "tiny_block_prefill_gate_proj_lhs_prepack.cpp"


def replace_function(text: str, function_name: str, replacement: str) -> str:
    start = text.index(f"static ModelError_t {function_name}(QnnModel& model){{")
    next_start = text.index("\nstatic ModelError_t ", start + 1)
    return text[:start] + replacement + text[next_start:]


NEW_GATE_PROJ_NODE = """static ModelError_t addNode__MatMul_6(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* Replaces gate_proj FullyConnected with LHS-prepack custom HTP MatMul:
     input [32,256] -> [1,1,32,256],
     gate_proj_weight [768,256] -> [1,1,768,256] -> Transpose -> [1,1,256,768]. */
  const char* inputs__MatMul_6_lhs_reshape[] = {
    "_MatMul_6_pre_reshape"
  };
  uint32_t dimensions__MatMul_6_lhs_4d[] = {1, 1, 32, 256};
  Qnn_Tensor_t outputs__MatMul_6_lhs_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_lhs_4d",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_6_lhs_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6_lhs_reshape",
                         "qti.aisw",
                         "Reshape",
                         nullptr,
                         0,
                         inputs__MatMul_6_lhs_reshape,
                         1,
                         outputs__MatMul_6_lhs_reshape,
                         1), err);

  const char* inputs__MatMul_6_lhs_cast[] = {
    "_MatMul_6_lhs_4d"
  };
  Qnn_Tensor_t outputs__MatMul_6_lhs_cast[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_lhs_fp16",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_16,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_6_lhs_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6_lhs_cast_fp16",
                         "qti.aisw",
                         "Cast",
                         nullptr,
                         0,
                         inputs__MatMul_6_lhs_cast,
                         1,
                         outputs__MatMul_6_lhs_cast,
                         1), err);

  const char* inputs__MatMul_6_pack_lhs[] = {
    "_MatMul_6_lhs_fp16"
  };
  Qnn_Tensor_t outputs__MatMul_6_pack_lhs[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_lhs_q13_bits",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_16,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_6_lhs_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6_pack_lhs_q13",
                         "MatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage",
                         "MatMulQhpiHvxPackLhs",
                         nullptr,
                         0,
                         inputs__MatMul_6_pack_lhs,
                         1,
                         outputs__MatMul_6_pack_lhs,
                         1), err);

  uint32_t dimensions__MatMul_6_weight_transpose_perm[] = {4};
  uint32_t _MatMul_6_weight_transpose_perm[] = {0, 1, 3, 2};
  Qnn_Param_t params__MatMul_6_weight_transpose[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_weight_transpose_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions__MatMul_6_weight_transpose_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)_MatMul_6_weight_transpose_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char* inputs__MatMul_6_weight_transpose[] = {
    "gate_proj_weight"
  };
  uint32_t dimensions__MatMul_6_weight_kn[] = {1, 1, 256, 768};
  Qnn_Tensor_t outputs__MatMul_6_weight_transpose[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_weight_kn",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_6_weight_kn,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6_weight_transpose",
                         "qti.aisw",
                         "Transpose",
                         params__MatMul_6_weight_transpose,
                         1,
                         inputs__MatMul_6_weight_transpose,
                         1,
                         outputs__MatMul_6_weight_transpose,
                         1), err);

  const char* inputs__MatMul_6_rhs_cast[] = {
    "_MatMul_6_weight_kn"
  };
  Qnn_Tensor_t outputs__MatMul_6_rhs_cast[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_rhs_fp16",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_16,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_6_weight_kn,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6_rhs_cast_fp16",
                         "qti.aisw",
                         "Cast",
                         nullptr,
                         0,
                         inputs__MatMul_6_rhs_cast,
                         1,
                         outputs__MatMul_6_rhs_cast,
                         1), err);

  const char* inputs__MatMul_6_custom[] = {
    "_MatMul_6_lhs_q13_bits",
    "_MatMul_6_rhs_fp16"
  };
  uint32_t dimensions__MatMul_6_custom_output_4d[] = {1, 1, 32, 768};
  Qnn_Tensor_t outputs__MatMul_6_custom[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_custom_output_4d",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_6_custom_output_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6",
                         "MatMulQhpiHvx8RowLhsPrepackFp32StoreOpPackage",
                         "MatMulQhpiHvx8RowLhsPrepackFp32Store",
                         nullptr,
                         0,
                         inputs__MatMul_6_custom,
                         2,
                         outputs__MatMul_6_custom,
                         1), err);

  const char* inputs__MatMul_6_output_reshape[] = {
    "_MatMul_6_custom_output_4d"
  };
  uint32_t dimensions__MatMul_6_output_0_fc[] = {32, 768};
  Qnn_Tensor_t outputs__MatMul_6_output_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_6_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__MatMul_6_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_6_output_reshape",
                         "qti.aisw",
                         "Reshape",
                         nullptr,
                         0,
                         inputs__MatMul_6_output_reshape,
                         1,
                         outputs__MatMul_6_output_reshape,
                         1), err);
  return err;
}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "uint32_t dimensions_gate_proj_weight[] = {768, 256};",
        "uint32_t dimensions_gate_proj_weight[] = {1, 1, 768, 256};",
        "gate_proj weight dims",
    )

    weight_start = text.index("static ModelError_t addTensor_gate_proj_weight")
    weight_end = text.index("static ModelError_t addNode__MatMul_6", weight_start)
    weight_section = text[weight_start:weight_end]
    weight_section = replace_once(
        weight_section,
        ".rank= 2,\n                                 .dimensions=dimensions_gate_proj_weight,",
        ".rank= 4,\n                                 .dimensions=dimensions_gate_proj_weight,",
        "gate_proj weight rank",
    )
    text = text[:weight_start] + weight_section + text[weight_end:]

    text = replace_function(text, "addNode__MatMul_6", NEW_GATE_PROJ_NODE)
    text = text.replace(
        "/* Command Line used:",
        "/* Patched by tiny_llm_block_custom_matmul/tools/patch_prefill_gate_proj_lhs_prepack.py\n"
        " * Replaces gate_proj FullyConnected with MatMulQhpiHvx8RowLhsPrepackFp32Store.\n"
        " */\n\n/* Command Line used:",
        1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
