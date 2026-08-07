#!/usr/bin/env python3
"""Create the past128 FlashAttention graph with head-contiguous K cache."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
QWEN = REPO / "qwen_block_custom_qnn"
BASE = QWEN / "generated/qwen2_0_5b_layer0_decode_past128_flash_attention.cpp"
OUTPUT = (
    QWEN
    / "generated/qwen2_0_5b_layer0_decode_past128_flash_attention_head_contiguous.cpp"
)
BASE_MODEL = "qwen2_0_5b_layer0_decode_past128_flash_attention"
MODEL = "qwen2_0_5b_layer0_decode_past128_flash_attention_head_contiguous"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected one match, got {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    # Regenerate the common fused graph first so this variant never depends on
    # hand-edited generated C++.
    spec = importlib.util.spec_from_file_location(
        "patch_qwen_decode_graph", HERE / "patch_qwen_decode_graph.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load base patcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()

    text = BASE.read_text(encoding="utf-8").replace(BASE_MODEL, MODEL)
    text = replace_once(
        text,
        "uint32_t dimensions_past_key[] = {1, 128, 64, 2};",
        "uint32_t dimensions_past_key[] = {1, 2, 128, 64};",
        "past-key dimensions",
    )
    text = replace_once(
        text,
        ".uint32Value = 1}}}}\n  };\n  const char*  inputs__Concat_4[] = {\n"
        '    "past_key",\n    "current_key_nhwc"\n'
        "  };\n  uint32_t dimensions__Concat_4_output_0[] = {1, 129, 64, 2};",
        ".uint32Value = 2}}}}\n  };\n  const char*  inputs__Concat_4[] = {\n"
        '    "past_key",\n    "current_key"\n'
        "  };\n  uint32_t dimensions__Concat_4_output_0[] = {1, 2, 129, 64};",
        "head-contiguous K concat",
    )
    call = f"  VALIDATE(addNode_current_key_nhwc({MODEL}), err);\n"
    text = replace_once(text, call, "", "current-key transpose call")
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
