#!/usr/bin/env python3
"""Remove the past-value NHWC-to-NCHW boundary transpose from decode C++."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "generated/qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv.cpp"
OUTPUT = ROOT / "generated/qwen2_0_5b_layer0_decode_past128_nchw_value_cache.cpp"
OLD_NAME = "qwen2_0_5b_layer0_decode_past128_grouped_gqa_delta_kv"
NEW_NAME = "qwen2_0_5b_layer0_decode_past128_nchw_value_cache"

text = SOURCE.read_text().replace(OLD_NAME, NEW_NAME)

old_dims = "uint32_t dimensions_past_value[] = {1, 128, 64, 2};"
new_dims = "uint32_t dimensions_past_value[] = {1, 2, 128, 64};"
if text.count(old_dims) != 1:
    raise RuntimeError("unexpected past_value dimension declaration")
text = text.replace(old_dims, new_dims, 1)

old_input = '    "past_value_nchw",\n    "current_value"'
new_input = '    "past_value",\n    "current_value"'
if text.count(old_input) != 1:
    raise RuntimeError("unexpected Concat_5 inputs")
text = text.replace(old_input, new_input, 1)

call = (
    rf"^  VALIDATE\(addNode_past_value_nchw"
    rf"\({NEW_NAME}\), err\);\n"
)
text, count = re.subn(call, "", text, count=1, flags=re.MULTILINE)
if count != 1:
    raise RuntimeError("past_value_nchw graph call was not removed")

OUTPUT.write_text(text)
print(OUTPUT)
