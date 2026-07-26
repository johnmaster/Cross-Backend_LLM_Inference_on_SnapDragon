#!/usr/bin/env python3
"""Restore FP16 KV API tensors that QAIRT 2.47 normalizes to FP32."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "generated/qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv_fp16.cpp"
)
OUTPUT = (
    ROOT
    / "generated/qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv_fp16_boundary.cpp"
)


def function(text: str, name: str) -> tuple[int, int, str]:
    start = text.index(f"static ModelError_t {name}(")
    next_function = text.find("\nstatic ModelError_t ", start + 10)
    end = (
        next_function
        if next_function >= 0
        else text.index("\nQNN_API", start + 10)
    )
    return start, end, text[start:end]


def replace_function_dtype(text: str, name: str, dtype: str) -> str:
    start, end, body = function(text, name)
    old = ".dataType= QNN_DATATYPE_FLOAT_32,"
    if body.count(old) != 1:
        raise SystemExit(f"{name}: expected one FP32 tensor")
    body = body.replace(old, f".dataType= QNN_DATATYPE_{dtype},", 1)
    return text[:start] + body + text[end:]


def cast_node(name: str, source: str, output: str) -> str:
    return f"""static ModelError_t {name}(QnnModel& model){{
  ModelError_t err = MODEL_NO_ERROR;
  const char* inputs[] = {{"{source}"}};
  uint32_t dimensions[] = {{1, 128, 64, 2}};
  Qnn_Tensor_t outputs[] = {{
    (Qnn_Tensor_t) {{
      .version=QNN_TENSOR_VERSION_2,
      {{.v2={{
        .id=0,
        .name="{output}",
        .type=QNN_TENSOR_TYPE_NATIVE,
        .dataFormat=QNN_TENSOR_DATA_FORMAT_DENSE,
        .dataType=QNN_DATATYPE_FLOAT_32,
        .quantizeParams={{QNN_DEFINITION_UNDEFINED,
                         QNN_QUANTIZATION_ENCODING_UNDEFINED,
                         {{.scaleOffsetEncoding={{.scale=0.0f, .offset=0}}}}}},
        .rank=4,
        .dimensions=dimensions,
        .memType=QNN_TENSORMEMTYPE_RAW,
        {{.clientBuf={{.data=nullptr, .dataSize=0}}}},
        .isDynamicDimensions=nullptr,
        .sparseParams={{QNN_SPARSE_LAYOUT_UNDEFINED,
                       .hybridCoo={{.numSpecifiedElements=0,
                                   .numSparseDimensions=0}}}},
        .isProduced=0}}}}
    }}
  }};
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
                         "{output}",
                         "qti.aisw",
                         "Cast",
                         nullptr,
                         0,
                         inputs,
                         1,
                         outputs,
                         1), err);
  return err;
}}

"""


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    text = replace_function_dtype(text, "addTensor_past_key", "FLOAT_16")
    text = replace_function_dtype(text, "addTensor_past_value", "FLOAT_16")
    text = replace_function_dtype(text, "addNode__Cast_2", "FLOAT_16")
    text = replace_function_dtype(text, "addNode__Cast_3", "FLOAT_16")

    insertion = text.index("static ModelError_t addNode_hidden_states_ncf(")
    text = (
        text[:insertion]
        + cast_node("addNode_past_key_fp32", "past_key", "past_key_fp32")
        + cast_node("addNode_past_value_fp32", "past_value", "past_value_fp32")
        + text[insertion:]
    )
    text = text.replace(
        'const char*  inputs__Cast_1_output_0_nchw[] = {\n    "past_value"\n  };',
        'const char*  inputs__Cast_1_output_0_nchw[] = {\n'
        '    "past_value_fp32"\n  };',
        1,
    )
    text = text.replace(
        'const char*  inputs__Concat_4[] = {\n'
        '    "past_key",\n'
        '    "_Reshape_4_output_0_nhwc"\n'
        '  };',
        'const char*  inputs__Concat_4[] = {\n'
        '    "past_key_fp32",\n'
        '    "_Reshape_4_output_0_nhwc"\n'
        '  };',
        1,
    )

    marker = (
        "  VALIDATE(addTensor_past_value("
        "qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv_fp16), err);\n"
    )
    calls = (
        marker
        + "  VALIDATE(addNode_past_key_fp32("
        "qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv_fp16), err);\n"
        + "  VALIDATE(addNode_past_value_fp32("
        "qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv_fp16), err);\n"
    )
    if text.count(marker) != 1:
        raise SystemExit("could not locate graph input registration")
    text = text.replace(marker, calls, 1)

    OUTPUT.write_text(
        "/* Explicit FP16 KV API with FP32 attention-side Casts. */\n" + text,
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
