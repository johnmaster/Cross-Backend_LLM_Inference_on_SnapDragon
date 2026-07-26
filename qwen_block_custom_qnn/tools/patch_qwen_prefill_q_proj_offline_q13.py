#!/usr/bin/env python3
"""Use an offline Q13 static q_proj RHS and remove its runtime FP16 Cast."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "generated"
    / "qwen2_0_5b_layer0_prefill_seq16_q_proj_custom_multithread_tile_cache.cpp"
)
OUTPUT = (
    ROOT
    / "generated"
    / "qwen2_0_5b_layer0_prefill_seq16_q_proj_offline_q13.cpp"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one {label}, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")

    weight_start = text.index(
        "static ModelError_t addTensor_onnx__MatMul_227"
    )
    weight_end = text.index(
        "static ModelError_t addTensor_self_attn_q_proj_bias", weight_start
    )
    weight = text[weight_start:weight_end]
    weight = replace_once(
        weight,
        ".dataType= QNN_DATATYPE_FLOAT_32,",
        ".dataType= QNN_DATATYPE_INT_16,",
        "q_proj weight datatype",
    )
    weight = replace_once(
        weight,
        """.quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},""",
        """.quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},""",
        "q_proj Q13 encoding",
    )
    weight = replace_once(
        weight,
        "BINVARSTART(onnx__MatMul_227)",
        "BINVARSTART(onnx__MatMul_227_q13)",
        "q_proj Q13 bin start",
    )
    weight = replace_once(
        weight,
        "BINLEN(onnx__MatMul_227)",
        "BINLEN(onnx__MatMul_227_q13)",
        "q_proj Q13 bin length",
    )
    text = text[:weight_start] + weight + text[weight_end:]

    cast_start = text.index("  const char* inputs__MatMul_rhs_cast[] = {")
    custom_start = text.index("  const char* inputs__MatMul_custom[] = {", cast_start)
    text = text[:cast_start] + text[custom_start:]
    text = replace_once(
        text,
        '    "_MatMul_rhs_fp16"\n',
        '    "onnx__MatMul_227"\n',
        "q_proj custom RHS input",
    )
    text = replace_once(
        text,
        '"MatMulQhpiHvx8RowFp32StoreMultithread"',
        '"MatMulQhpiHvxOfflineQ13RhsFp32StoreMultithread"',
        "offline-Q13 op type",
    )

    OUTPUT.write_text(
        "/* Offline FP16-rounded Q13 q_proj RHS; runtime RHS Cast removed. */\n"
        + text,
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
