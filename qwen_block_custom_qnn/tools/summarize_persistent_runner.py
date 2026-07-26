#!/usr/bin/env python3
"""Summarize graphExecute wall times emitted by the persistent KV runner."""

import argparse
import csv
import statistics
from pathlib import Path


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


parser = argparse.ArgumentParser()
parser.add_argument("csv", type=Path)
parser.add_argument("--drop", type=int, default=1)
args = parser.parse_args()

with args.csv.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

if args.drop < 0 or args.drop >= len(rows):
    raise SystemExit(f"--drop must be in [0, {len(rows) - 1}]")

print(f"steps_total={len(rows)}")
print(f"dropped={args.drop}")
for column in ("graph_execute_us", "cache_update_us", "step_total_us"):
    if column not in rows[0]:
        continue
    values = [int(row[column]) for row in rows]
    kept = values[args.drop:]
    print(f"{column}_first={values[0]}")
    print(f"{column}_median={statistics.median(kept):.1f}")
    print(f"{column}_mean={statistics.fmean(kept):.2f}")
    print(f"{column}_p90={percentile(kept, 0.90)}")
    print(f"{column}_p95={percentile(kept, 0.95)}")
    print(f"{column}_min={min(kept)}")
    print(f"{column}_max={max(kept)}")
