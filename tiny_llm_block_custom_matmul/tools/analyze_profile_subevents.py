#!/usr/bin/env python3
"""Print median-after-warmup profile subevents for selected prefill runs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROFILES = {
    "builtin": REPO_ROOT / "tiny_llm_block/device_output/prefill/profile.csv",
    "q_proj_lhs": REPO_ROOT
    / "tiny_llm_block_custom_matmul/device_output/prefill_q_proj_lhs_prepack/profile.csv",
    "gate_lhs": REPO_ROOT
    / "tiny_llm_block_custom_matmul/device_output/prefill_gate_proj_lhs_prepack/profile.csv",
    "gate_lhs_mt": REPO_ROOT
    / "tiny_llm_block_custom_matmul/device_output/prefill_gate_proj_lhs_prepack_multithread/profile.csv",
}


def median_after_warmup(items: list[int]) -> int:
    if len(items) < 2:
        return 0
    return int(statistics.median(items[1:]))


def load_execute_cycles(path: Path) -> tuple[int, dict[str, int]]:
    root_values: list[int] = []
    events: dict[str, list[int]] = {}

    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            row = [column.strip() for column in row]
            if len(row) < 7 or row[1] != "EXECUTE" or row[3] != "CYCLES":
                continue
            if row[4] != "BACKEND":
                continue
            value = int(row[2])
            event = row[6]
            if row[5] == "ROOT" and event == "Accelerator (execute) time (cycles)":
                root_values.append(value)
            elif row[5] == "SUB-EVENT":
                events.setdefault(event, []).append(value)

    return median_after_warmup(root_values), {
        event: median_after_warmup(values) for event, values in events.items()
    }


def interesting(events: dict[str, int]) -> dict[str, int]:
    wanted = {}
    for name, value in events.items():
        if (
            "_MatMul" in name
            or "FullyConnected" in name
            or "pack_lhs" in name
            or "weight_transpose" in name
        ):
            wanted[name] = value
    return wanted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    for name, path in DEFAULT_PROFILES.items():
        root, events = load_execute_cycles(path)
        print(f"\n== {name} ==")
        print(f"profile: {path.relative_to(REPO_ROOT)}")
        print(f"root_cycles: {root}")

        selected = interesting(events)
        for event, value in sorted(selected.items(), key=lambda item: item[1], reverse=True)[
            : args.top
        ]:
            print(f"{value:10d}  {event}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
