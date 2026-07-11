#!/usr/bin/env python3
"""Patch converter-generated prefill QNN source to use custom q_proj MatMul."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_INPUT = REPO_ROOT / "tiny_llm_block" / "generated" / "tiny_block_prefill.cpp"
DEFAULT_OUTPUT = ROOT / "generated" / "tiny_block_prefill_q_proj_custom.cpp"


OLD_WEIGHT_DIMS = """static ModelError_t addTensor_q_proj_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_q_proj_weight[] = {256, 256};
  VALIDATE(model.addTensor("q_proj_weight", // Tensor Name"""

NEW_WEIGHT_DIMS = """static ModelError_t addTensor_q_proj_weight(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_q_proj_weight[] = {1, 1, 256, 256};
  VALIDATE(model.addTensor("q_proj_weight", // Tensor Name"""

OLD_WEIGHT_RANK = """.rank= 2,
                                 .dimensions=dimensions_q_proj_weight,"""

NEW_WEIGHT_RANK = """.rank= 4,
                                 .dimensions=dimensions_q_proj_weight,"""

OLD_Q_PROJ_NODE_START = """static ModelError_t addNode__MatMul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR _MatMul */
  const char*  inputs__MatMul[] = {
    "_MatMul_pre_reshape",
    "q_proj_weight",
    "q_proj_bias"
  };
  uint32_t dimensions__Add_1_output_0_fc[] = {32, 256};
  Qnn_Tensor_t outputs__MatMul[] = {
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
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "_MatMul", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs__MatMul, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs__MatMul, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}
"""

NEW_Q_PROJ_NODE = """static ModelError_t addNode__MatMul(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* Replaces q_proj FullyConnected with custom HTP MatMul:
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

  const char* inputs__MatMul_custom[] = {
    "_MatMul_lhs_fp16",
    "_MatMul_rhs_fp16"
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
                         "MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage",
                         "MatMulQhpiHvx8RowFp32StoreMultithread",
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


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one {label} match, found {count}")
    return text.replace(old, new, 1)


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
        "/* Patched by tiny_llm_block_custom_matmul/tools/patch_prefill_q_proj_custom.py\n"
        " * Replaces q_proj FullyConnected with MatMulQhpiHvx8RowFp32StoreMultithread.\n"
        " */\n\n/* Command Line used:",
        1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
