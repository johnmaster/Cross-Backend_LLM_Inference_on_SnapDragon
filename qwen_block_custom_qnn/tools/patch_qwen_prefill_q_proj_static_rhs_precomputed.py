#!/usr/bin/env python3
"""Retarget the verified Qwen q_proj patch to the static-RHS QHPI kernel."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "generated"
    / "qwen2_0_5b_layer0_prefill_seq16_q_proj_custom_multithread_tile_cache.cpp"
)
DEFAULT_OUTPUT = (
    ROOT
    / "generated"
    / "qwen2_0_5b_layer0_prefill_seq16_q_proj_static_rhs_precomputed.cpp"
)
OLD_QUOTED_OP = '"MatMulQhpiHvx8RowFp32StoreMultithread"'
NEW_QUOTED_OP = '"MatMulQhpiHvxStaticRhsPrecomputed"'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    occurrences = text.count(OLD_QUOTED_OP)
    if occurrences != 1:
        raise SystemExit(
            f"expected exactly one q_proj op type, found {occurrences}: "
            f"{OLD_QUOTED_OP}"
        )

    patched = text.replace(OLD_QUOTED_OP, NEW_QUOTED_OP, 1)
    old_inputs = (
        'const char* inputs__MatMul_custom[] = {\n'
        '    "_MatMul_lhs_fp16",\n'
        '    "_MatMul_rhs_fp16"\n'
        "  };"
    )
    new_inputs = (
        'const char* inputs__MatMul_custom[] = {\n'
        '    "_MatMul_lhs_fp16",\n'
        '    "_MatMul_rhs_fp16",\n'
        '    "onnx__MatMul_227"\n'
        "  };"
    )
    if patched.count(old_inputs) != 1:
        raise SystemExit("could not uniquely locate q_proj custom input array")
    patched = patched.replace(old_inputs, new_inputs, 1)
    old_count = (
        "                         inputs__MatMul_custom,\n"
        "                         2,\n"
        "                         outputs__MatMul_custom,"
    )
    new_count = (
        "                         inputs__MatMul_custom,\n"
        "                         3,\n"
        "                         outputs__MatMul_custom,"
    )
    if patched.count(old_count) != 1:
        raise SystemExit("could not uniquely locate q_proj custom input count")
    patched = patched.replace(old_count, new_count, 1)
    marker = (
        "/* Retargeted by "
        "qwen_block_custom_qnn/tools/"
        "patch_qwen_prefill_q_proj_static_rhs_precomputed.py.\n"
        " * The package is unchanged; only q_proj selects the graph-load "
        "static-RHS kernel.\n"
        " */\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(marker + patched, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
