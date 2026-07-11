#!/usr/bin/env python3
"""Summarize Qwen block QNN profiling CSVs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "device_output" / "builtin_layer0_prefill_seq16" / "profile.csv"


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path, nargs="?", default=DEFAULT_PROFILE)
    args = parser.parse_args()

    metrics = [
        ("root_cycles", "CYCLES", "BACKEND", "ROOT", "Accelerator (execute) time (cycles)"),
        ("qnn_accel_us", "US", "BACKEND", "ROOT", "QNN accelerator (execute) time"),
        ("qnn_us", "US", "BACKEND", "ROOT", "QNN (execute) time"),
        ("netrun_us", "US", "NETRUN", "ROOT", None),
        ("q_proj_rhs_cast_cycles", "CYCLES", "BACKEND", "SUB-EVENT", "_MatMul_rhs_cast_fp16"),
        ("q_proj_cycles", "CYCLES", "BACKEND", "SUB-EVENT", "_MatMul:"),
        ("q_proj_bias_add_cycles", "CYCLES", "BACKEND", "SUB-EVENT", "_MatMul_bias_add"),
        ("gate_proj_cycles", "CYCLES", "BACKEND", "SUB-EVENT", "_MatMul_6:"),
        ("up_proj_cycles", "CYCLES", "BACKEND", "SUB-EVENT", "_MatMul_7:"),
        ("down_proj_cycles", "CYCLES", "BACKEND", "SUB-EVENT", "_MatMul_8:"),
    ]

    for label, unit, source, level, event in metrics:
        print(f"{label:18s} {median_after_warmup(values(args.profile, unit, source, level, event))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
