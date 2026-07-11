#!/usr/bin/env python3
"""Summarize prefill profiling CSVs with a consistent median-after-warmup rule."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

PROFILES = {
    "builtin": REPO_ROOT / "tiny_llm_block" / "device_output" / "prefill" / "profile.csv",
    "custom_multithread": REPO_ROOT
    / "tiny_llm_block_custom_matmul"
    / "device_output"
    / "prefill_q_proj_custom"
    / "profile.csv",
    "custom_lhs_prepack": REPO_ROOT
    / "tiny_llm_block_custom_matmul"
    / "device_output"
    / "prefill_q_proj_lhs_prepack"
    / "profile.csv",
    "custom_lhs_prepack_mt": REPO_ROOT
    / "tiny_llm_block_custom_matmul"
    / "device_output"
    / "prefill_q_proj_lhs_prepack_multithread"
    / "profile.csv",
    "custom_both_prepack": REPO_ROOT
    / "tiny_llm_block_custom_matmul"
    / "device_output"
    / "prefill_q_proj_both_prepack"
    / "profile.csv",
    "custom_lhs_tile_cache": REPO_ROOT
    / "tiny_llm_block_custom_matmul"
    / "device_output"
    / "prefill_q_proj_lhs_tile_cache"
    / "profile.csv",
    "custom_gate_lhs_prepack": REPO_ROOT
    / "tiny_llm_block_custom_matmul"
    / "device_output"
    / "prefill_gate_proj_lhs_prepack"
    / "profile.csv",
    "custom_gate_lhs_prepack_mt": REPO_ROOT
    / "tiny_llm_block_custom_matmul"
    / "device_output"
    / "prefill_gate_proj_lhs_prepack_multithread"
    / "profile.csv",
}


def values(path: Path, unit: str, source: str, level: str, event: str | None = None) -> list[int]:
    result: list[int] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            row = [column.strip() for column in row]
            if (
                len(row) >= 7
                and row[1] == "EXECUTE"
                and row[3] == unit
                and row[4] == source
                and row[5] == level
                and (event is None or event in row[6])
            ):
                result.append(int(row[2]))
    return result


def median_after_warmup(items: list[int]) -> int:
    if len(items) < 2:
        return 0
    return int(statistics.median(items[1:]))


def summarize(name: str, path: Path) -> None:
    root_cycles = median_after_warmup(
        values(path, "CYCLES", "BACKEND", "ROOT", "Accelerator (execute) time (cycles)")
    )
    qnn_accelerator_us = median_after_warmup(
        values(path, "US", "BACKEND", "ROOT", "QNN accelerator (execute) time")
    )
    netrun_us = median_after_warmup(values(path, "US", "NETRUN", "ROOT"))
    q_proj_cycles = median_after_warmup(
        values(path, "CYCLES", "BACKEND", "SUB-EVENT", "_MatMul:")
    )
    pack_lhs_cycles = median_after_warmup(
        values(path, "CYCLES", "BACKEND", "SUB-EVENT", "_MatMul_pack_lhs_q13")
    )
    pack_rhs_cycles = median_after_warmup(
        values(path, "CYCLES", "BACKEND", "SUB-EVENT", "_MatMul_pack_rhs_q13")
    )
    gate_proj_cycles = median_after_warmup(
        values(path, "CYCLES", "BACKEND", "SUB-EVENT", "_MatMul_6:")
    )
    gate_pack_lhs_cycles = median_after_warmup(
        values(path, "CYCLES", "BACKEND", "SUB-EVENT", "_MatMul_6_pack_lhs_q13")
    )
    gate_weight_transpose_cycles = median_after_warmup(
        values(path, "CYCLES", "BACKEND", "SUB-EVENT", "_MatMul_6_weight_transpose")
    )

    print(
        f"{name:24s} "
        f"root_cycles={root_cycles:9d} "
        f"qnn_us={qnn_accelerator_us:5d} "
        f"netrun_us={netrun_us:5d} "
        f"q_proj_cycles={q_proj_cycles:9d} "
        f"pack_lhs_cycles={pack_lhs_cycles:6d} "
        f"pack_rhs_cycles={pack_rhs_cycles:6d} "
        f"gate_proj_cycles={gate_proj_cycles:9d} "
        f"gate_pack_lhs_cycles={gate_pack_lhs_cycles:6d} "
        f"gate_weight_transpose_cycles={gate_weight_transpose_cycles:6d}"
    )


def main() -> int:
    for name, path in PROFILES.items():
        summarize(name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
