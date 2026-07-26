#!/usr/bin/env python3
"""Retarget the verified q_proj node to the isolated precompute probe."""

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
    / "qwen2_0_5b_layer0_prefill_seq16_q_proj_precompute_probe.cpp"
)

REPLACEMENTS = {
    '"MatMulQhpiHvx8RowFp32StoreMultithreadOpPackage"':
        '"QhpiPrecomputeProbeOpPackage"',
    '"MatMulQhpiHvx8RowFp32StoreMultithread"':
        '"QhpiPrecomputeProbe"',
}


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS.items():
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"expected one occurrence of {old}, found {count}")
        text = text.replace(old, new, 1)
    OUTPUT.write_text(
        "/* Isolated QHPI precompute probe; output values are intentionally "
        "undefined. */\n" + text,
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
