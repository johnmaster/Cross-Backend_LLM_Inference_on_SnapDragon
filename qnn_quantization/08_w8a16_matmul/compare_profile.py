#!/usr/bin/env python3

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def cycles(path, event_name, level):
    values = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            row = [column.strip() for column in row]
            if (
                len(row) >= 7
                and row[1] == "EXECUTE"
                and row[3] == "CYCLES"
                and row[5] == level
                and event_name in row[6]
            ):
                values.append(int(row[2]))
    return values


def summarize(label, path):
    total = cycles(path, "Accelerator (execute) time (cycles)", "ROOT")
    matmul = cycles(path, "MatMul:", "SUB-EVENT")
    if len(total) < 2 or len(total) != len(matmul):
        raise RuntimeError(f"unexpected profiling events in {path}")

    # The first inference includes cold-start effects; compare steady-state runs.
    total_median = statistics.median(total[1:])
    matmul_median = statistics.median(matmul[1:])
    print(
        f"{label:8s} total={total} median_warm={total_median:.1f}  "
        f"matmul={matmul} median_warm={matmul_median:.1f}"
    )
    return total_median, matmul_median


def main():
    w8a8_total, w8a8_matmul = summarize(
        "W8A8", ROOT / "profile" / "w8a8.csv"
    )
    w8a16_total, w8a16_matmul = summarize(
        "W8A16", ROOT / "profile" / "w8a16.csv"
    )
    print(f"\nTotal cycle ratio W8A16/W8A8:  {w8a16_total / w8a8_total:.2f}x")
    print(f"MatMul cycle ratio W8A16/W8A8: {w8a16_matmul / w8a8_matmul:.2f}x")


if __name__ == "__main__":
    main()
