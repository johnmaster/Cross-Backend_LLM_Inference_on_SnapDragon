#!/usr/bin/env python3
"""Patch the adopted past128 grouped-GQA decode source with fused attention."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "generated/qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv.cpp"
OUTPUT = ROOT / "generated/qwen2_0_5b_layer0_decode_past128_fused_attention.cpp"

MODEL_OLD = "qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv"
MODEL_NEW = "qwen2_0_5b_layer0_decode_past128_fused_attention"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


text = SOURCE.read_text()
text = text.replace(MODEL_OLD, MODEL_NEW)

# Present Q to QHPI as Flat4. This changes metadata only; the contiguous
# [2,7,1,64] and [14,1,64] element order is identical.
text = replace_once(
    text,
    "uint32_t dimensions__Reshape_5_output_0[] = {1, 2, 7, 1, 64};",
    "uint32_t dimensions__Reshape_5_output_0[] = {1, 14, 1, 64};",
    "Q dimensions",
)
text = replace_once(text, ".rank= 5,\n            .dimensions=dimensions__Reshape_5_output_0,",
                    ".rank= 4,\n            .dimensions=dimensions__Reshape_5_output_0,",
                    "Q rank")

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
      .version= QNN_TENSOR_VERSION_2,
      {.v2= {
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
      "_FusedDecodeAttentionPast128",
      "MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage",
      "Qwen2DecodeAttentionPast128Fp32",
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
    raise RuntimeError(f"MatMul_4 function: expected one replacement, found {count}")

# These nodes are now internal to the fused operator. Removing their graph
# construction calls guarantees that profiling measures the actual fusion,
# rather than relying on backend dead-code elimination.
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
        raise RuntimeError(f"{function} call: expected one removal, found {count}")

OUTPUT.write_text(text)
print(OUTPUT)
