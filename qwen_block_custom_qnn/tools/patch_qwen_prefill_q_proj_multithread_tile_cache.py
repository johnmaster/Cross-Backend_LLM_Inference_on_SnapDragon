#!/usr/bin/env python3
"""Retarget the validated Qwen q_proj patch to the multithread tile-cache op."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "generated" / "qwen2_0_5b_layer0_prefill_seq16_q_proj_custom.cpp"
)
DEFAULT_OUTPUT = (
    ROOT
    / "generated"
    / "qwen2_0_5b_layer0_prefill_seq16_q_proj_custom_multithread_tile_cache.cpp"
)

OLD_PACKAGE = "MatMulQhpiHvx8RowLhsTileCacheFp32StoreOpPackage"
OLD_OP = '"MatMulQhpiHvx8RowLhsTileCacheFp32Store"'
NEW_PACKAGE = "MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage"
NEW_OP = '"MatMulQhpiHvx8RowFp32StoreMultithread"'


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
    text = replace_once(text, OLD_PACKAGE, NEW_PACKAGE, "package name")
    text = replace_once(text, OLD_OP, NEW_OP, "op type")
    text = text.replace(
        "Replaces layer0 q_proj FullyConnected with "
        "MatMulQhpiHvx8RowLhsTileCacheFp32Store.",
        "Replaces layer0 q_proj FullyConnected with the multithread "
        "LHS-tile-cache MatMul.",
        1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
