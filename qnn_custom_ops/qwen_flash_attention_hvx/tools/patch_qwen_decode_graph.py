#!/usr/bin/env python3
"""Replace past128 QK/Softmax/AV nodes with the standalone FlashAttention op."""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
QWEN = REPO / "qwen_block_custom_qnn"
SOURCE = QWEN / "generated/qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv.cpp"
OUTPUT = QWEN / "generated/qwen2_0_5b_layer0_decode_past128_flash_attention.cpp"
MODEL_OLD = "qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv"
MODEL_NEW = "qwen2_0_5b_layer0_decode_past128_flash_attention"


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8").replace(MODEL_OLD, MODEL_NEW)
    # QHPI Flat4 requires rank 4. [2,7,Q,64] and [14,Q,64] have identical
    # contiguous element order, so this is metadata-only.
    old_dims = "uint32_t dimensions__Reshape_5_output_0[] = {1, 2, 7, 1, 64};"
    new_dims = "uint32_t dimensions__Reshape_5_output_0[] = {1, 14, 1, 64};"
    if text.count(old_dims) != 1:
        raise RuntimeError("unexpected Q dimension declaration")
    text = text.replace(old_dims, new_dims, 1)
    old_rank = ".rank= 5,\n            .dimensions=dimensions__Reshape_5_output_0,"
    new_rank = ".rank= 4,\n            .dimensions=dimensions__Reshape_5_output_0,"
    if text.count(old_rank) != 1:
        raise RuntimeError("unexpected Q rank declaration")
    text = text.replace(old_rank, new_rank, 1)

    fused_node = r'''static ModelError_t addNode__MatMul_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  const char* inputs__MatMul_4[] = {
    "_Reshape_5_output_0",
    "_Concat_4_output_0",
    "_Concat_5_output_0"
  };
  uint32_t dimensions__Reshape_6_output_0[] = {1, 14, 1, 64};
  Qnn_Tensor_t outputs__MatMul_4[] = {
    (Qnn_Tensor_t) {
      .version=QNN_TENSOR_VERSION_2,
      {.v2={
        .id=0,
        .name="_Reshape_6_output_0",
        .type=QNN_TENSOR_TYPE_NATIVE,
        .dataFormat=QNN_TENSOR_DATA_FORMAT_DENSE,
        .dataType=QNN_DATATYPE_FLOAT_32,
        .quantizeParams={QNN_DEFINITION_UNDEFINED,
                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                         {.scaleOffsetEncoding={.scale=0.0f, .offset=0}}},
        .rank=4,
        .dimensions=dimensions__Reshape_6_output_0,
        .memType=QNN_TENSORMEMTYPE_RAW,
        {.clientBuf={.data=nullptr, .dataSize=0}},
        .isDynamicDimensions=nullptr,
        .sparseParams={QNN_SPARSE_LAYOUT_UNDEFINED,
                       .hybridCoo={.numSpecifiedElements=0,
                                   .numSparseDimensions=0}},
        .isProduced=0}}}
  };
  VALIDATE(model.addNode(
      QNN_OPCONFIG_VERSION_1,
      "_QwenGqaFlashAttention",
      "QwenFlashAttentionHvxOpPackage",
      "QwenGqaFlashAttentionFp32",
      nullptr, 0, inputs__MatMul_4, 3, outputs__MatMul_4, 1), err);
  return err;
}'''
    text, count = re.subn(
        r"static ModelError_t addNode__MatMul_4\(QnnModel& model\)\{.*?\n\}\n\n"
        r"(?=static ModelError_t addNode__Reshape_6)",
        fused_node + "\n\n",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"MatMul_4 replacement count={count}")

    removed_calls = [
        "addNode__Concat_4_output_0_nchw",
        "addNode__Unsqueeze_4",
        "addNode__MatMul_3",
        "addTensor__Constant_36_output_0",
        "addNode__Div_1",
        "addTensor_onnx__Add_269",
        "addNode__Add_6",
        "addNode__Softmax",
        "addNode__Unsqueeze_5",
        "addNode__Reshape_6",
    ]
    for function in removed_calls:
        pattern = (
            rf"^  VALIDATE\({re.escape(function)}"
            rf"\({MODEL_NEW}\), err\);\n"
        )
        text, count = re.subn(pattern, "", text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError(f"{function} removal count={count}")

    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
