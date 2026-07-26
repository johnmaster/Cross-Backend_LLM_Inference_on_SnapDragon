#!/usr/bin/env python3
"""Summarize fixed-past Qwen decode profiles using post-warmup medians."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


EVENTS = {
    "root_cycles": ("CYCLES", "BACKEND", "ROOT", "Accelerator (execute) time (cycles)"),
    "qnn_accel_us": ("US", "BACKEND", "ROOT", "QNN accelerator (execute) time"),
    "qnn_us": ("US", "BACKEND", "ROOT", "QNN (execute) time"),
    "netrun_us": ("US", "NETRUN", "ROOT", None),
    "qk_cycles": ("CYCLES", "BACKEND", "SUB-EVENT", "_MatMul_3:"),
    "softmax_cycles": ("CYCLES", "BACKEND", "SUB-EVENT", "_Softmax:"),
    "av_cycles": ("CYCLES", "BACKEND", "SUB-EVENT", "_MatMul_4:"),
    "key_layout_cycles": ("CYCLES", "BACKEND", "SUB-EVENT", "past_key_nhwc"),
    "key_concat_cycles": ("CYCLES", "BACKEND", "SUB-EVENT", "_Concat_4:"),
    "value_concat_cycles": ("CYCLES", "BACKEND", "SUB-EVENT", "_Concat_5:"),
    "k_tile_cycles": ("CYCLES", "BACKEND", "SUB-EVENT", "_Tile:"),
    "v_tile_cycles": ("CYCLES", "BACKEND", "SUB-EVENT", "_Tile_1:"),
}


def values(path: Path, spec: tuple[str, str, str, str | None]) -> list[int]:
    unit, source, level, event = spec
    result: list[int] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            row = [item.strip() for item in row]
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
    if not items:
        return 0
    if len(items) == 1:
        return items[0]
    return int(statistics.median(items[1:]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", type=Path, nargs="+")
    args = parser.parse_args()
    print("profile," + ",".join(EVENTS))
    for profile in args.profiles:
        metrics = [
            str(median_after_warmup(values(profile, spec)))
            for spec in EVENTS.values()
        ]
        print(profile.parent.name + "," + ",".join(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
