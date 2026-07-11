#!/usr/bin/env python3
"""Patch prefill QNN source to use both-prepack custom q_proj MatMul."""

from __future__ import annotations

import argparse
from pathlib import Path

from patch_prefill_q_proj_custom import (
    DEFAULT_INPUT,
    NEW_WEIGHT_DIMS,
    NEW_WEIGHT_RANK,
    OLD_Q_PROJ_NODE_START,
    OLD_WEIGHT_DIMS,
    OLD_WEIGHT_RANK,
    replace_once,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "generated" / "tiny_block_prefill_q_proj_both_prepack.cpp"


NEW_Q_PROJ_NODE = """static ModelError_t addNode__MatMul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* Replaces q_proj FullyConnected with both-prepack custom HTP MatMul:
     [32,256] -> [1,1,32,256], q_proj_weight -> [1,1,256,256]. */
  const char* inputs__MatMul_lhs_reshape[] = {
    "_MatMul_pre_reshape"
  };
  uint32_t dimensions__MatMul_lhs_4d[] = {1, 1, 32, 256};
  Qnn_Tensor_t outputs__MatMul_lhs_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_lhs_4d",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_lhs_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_lhs_reshape",
                         "qti.aisw",
                         "Reshape",
                         nullptr,
                         0,
                         inputs__MatMul_lhs_reshape,
                         1,
                         outputs__MatMul_lhs_reshape,
                         1), err);

  const char* inputs__MatMul_lhs_cast[] = {
    "_MatMul_lhs_4d"
  };
  Qnn_Tensor_t outputs__MatMul_lhs_cast[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_lhs_fp16",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_16,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_lhs_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_lhs_cast_fp16",
                         "qti.aisw",
                         "Cast",
                         nullptr,
                         0,
                         inputs__MatMul_lhs_cast,
                         1,
                         outputs__MatMul_lhs_cast,
                         1), err);

  const char* inputs__MatMul_pack_lhs[] = {
    "_MatMul_lhs_fp16"
  };
  Qnn_Tensor_t outputs__MatMul_pack_lhs[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_lhs_q13_bits",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_16,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_lhs_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_pack_lhs_q13",
                         "MatMulQhpiHvx8RowBothPrepackFp32StoreOpPackage",
                         "MatMulQhpiHvxPackLhs",
                         nullptr,
                         0,
                         inputs__MatMul_pack_lhs,
                         1,
                         outputs__MatMul_pack_lhs,
                         1), err);

  const char* inputs__MatMul_rhs_cast[] = {
    "q_proj_weight"
  };
  uint32_t dimensions_q_proj_weight_4d[] = {1, 1, 256, 256};
  Qnn_Tensor_t outputs__MatMul_rhs_cast[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_rhs_fp16",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_16,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_q_proj_weight_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_rhs_cast_fp16",
                         "qti.aisw",
                         "Cast",
                         nullptr,
                         0,
                         inputs__MatMul_rhs_cast,
                         1,
                         outputs__MatMul_rhs_cast,
                         1), err);

  const char* inputs__MatMul_pack_rhs[] = {
    "_MatMul_rhs_fp16"
  };
  Qnn_Tensor_t outputs__MatMul_pack_rhs[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_rhs_q13_bits",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_16,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_q_proj_weight_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_pack_rhs_q13",
                         "MatMulQhpiHvx8RowBothPrepackFp32StoreOpPackage",
                         "MatMulQhpiHvxPackRhs",
                         nullptr,
                         0,
                         inputs__MatMul_pack_rhs,
                         1,
                         outputs__MatMul_pack_rhs,
                         1), err);

  const char* inputs__MatMul_custom[] = {
    "_MatMul_lhs_q13_bits",
    "_MatMul_rhs_q13_bits"
  };
  uint32_t dimensions__MatMul_custom_output_4d[] = {1, 1, 32, 256};
  Qnn_Tensor_t outputs__MatMul_custom[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_MatMul_custom_output_4d",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__MatMul_custom_output_4d,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul",
                         "MatMulQhpiHvx8RowBothPrepackFp32StoreOpPackage",
                         "MatMulQhpiHvx8RowBothPrepackFp32Store",
                         nullptr,
                         0,
                         inputs__MatMul_custom,
                         2,
                         outputs__MatMul_custom,
                         1), err);

  const char* inputs__MatMul_output_reshape[] = {
    "_MatMul_custom_output_4d"
  };
  uint32_t dimensions__Add_1_output_0_fc[] = {32, 256};
  Qnn_Tensor_t outputs__MatMul_output_reshape[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "_Add_1_output_0_fc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions__Add_1_output_0_fc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "_MatMul_output_reshape",
                         "qti.aisw",
                         "Reshape",
                         nullptr,
                         0,
                         inputs__MatMul_output_reshape,
                         1,
                         outputs__MatMul_output_reshape,
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
    text = replace_once(text, OLD_WEIGHT_DIMS, NEW_WEIGHT_DIMS, "q_proj weight dims")

    start = text.index(NEW_WEIGHT_DIMS)
    end = text.index("static ModelError_t addTensor_q_proj_bias", start)
    section = text[start:end]
    section = replace_once(section, OLD_WEIGHT_RANK, NEW_WEIGHT_RANK, "q_proj weight rank")
    text = text[:start] + section + text[end:]

    text = replace_once(text, OLD_Q_PROJ_NODE_START, NEW_Q_PROJ_NODE, "q_proj node")
    text = text.replace(
        "/* Command Line used:",
        "/* Patched by tiny_llm_block_custom_matmul/tools/patch_prefill_q_proj_both_prepack.py\n"
        " * Replaces q_proj FullyConnected with MatMulQhpiHvx8RowBothPrepackFp32Store.\n"
        " */\n\n/* Command Line used:",
        1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
