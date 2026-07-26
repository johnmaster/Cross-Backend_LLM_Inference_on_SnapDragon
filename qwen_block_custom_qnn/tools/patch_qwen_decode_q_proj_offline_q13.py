#!/usr/bin/env python3
"""Replace decode layer0 q_proj with the M=1 offline-Q13 custom HTP op."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "generated/qwen2_0_5b_layer0_decode_past16.cpp"
PREFILL_TEMPLATE = (
    ROOT / "generated/qwen2_0_5b_layer0_prefill_seq16_q_proj_offline_q13.cpp"
)
OUTPUT = (
    ROOT / "generated/qwen2_0_5b_layer0_decode_past16_q_proj_offline_q13.cpp"
)


def function(text: str, name: str) -> tuple[int, int, str]:
    start = text.index(f"static ModelError_t {name}(")
    end = text.index("\nstatic ModelError_t ", start + 10)
    return start, end, text[start:end]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one {label}, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    template = PREFILL_TEMPLATE.read_text(encoding="utf-8")

    # Change the static tensor to the custom op's rank-4 INT16 RHS.
    ws, we, weight = function(text, "addTensor_onnx__MatMul_230")
    weight = replace_once(
        weight, "{896, 896}", "{1, 1, 896, 896}", "weight dimensions"
    )
    weight = replace_once(weight, ".rank= 2,", ".rank= 4,", "weight rank")
    weight = replace_once(
        weight,
        ".dataType= QNN_DATATYPE_FLOAT_32,",
        ".dataType= QNN_DATATYPE_INT_16,",
        "weight datatype",
    )
    weight = replace_once(
        weight,
        "BINVARSTART(onnx__MatMul_230)",
        "BINVARSTART(onnx__MatMul_230_q13)",
        "weight bin start",
    )
    weight = replace_once(
        weight,
        "BINLEN(onnx__MatMul_230)",
        "BINLEN(onnx__MatMul_230_q13)",
        "weight bin length",
    )
    text = text[:ws] + weight + text[we:]

    # Reuse the validated prefill graph wrapper, specialized to one token.
    _, _, node = function(template, "addNode__MatMul")
    node = node.replace("onnx__MatMul_227", "onnx__MatMul_230")
    node = node.replace('"_MatMul_pre_reshape"', '"_Mul_1_output_0"')
    node = node.replace("{1, 1, 16, 896}", "{1, 1, 1, 896}")
    node = node.replace("{16, 896}", "{1, 896}")
    ns, ne, _ = function(text, "addNode__MatMul")
    text = text[:ns] + node + text[ne:]

    OUTPUT.write_text(
        "/* Decode M=1 q_proj with device-exact offline Q13 RHS. */\n" + text,
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
