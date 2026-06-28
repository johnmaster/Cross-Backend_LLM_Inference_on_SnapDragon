#!/usr/bin/env python3

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def values(
    path: Path, unit: str, source: str, event: str | None = None
) -> list[int]:
    result = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            row = [column.strip() for column in row]
            if (
                len(row) >= 7
                and row[1] == "EXECUTE"
                and row[3] == unit
                and row[4] == source
                and row[5] == "ROOT"
                and (event is None or event in row[6])
            ):
                result.append(int(row[2]))
    return result


def summarize(mode: str) -> None:
    path = ROOT / "device_output" / mode / "profile.csv"
    cycles = values(
        path, "CYCLES", "BACKEND", "Accelerator (execute) time (cycles)"
    )
    accelerator_us = values(
        path, "US", "BACKEND", "QNN accelerator (execute) time"
    )
    netrun_us = values(path, "US", "NETRUN")
    if not (len(cycles) == len(accelerator_us) == len(netrun_us) == 10):
        raise RuntimeError(f"expected 10 execute events in {path}")
    print(
        f"{mode:7s} accelerator_cycles={statistics.median(cycles[1:]):.0f} "
        f"accelerator_us={statistics.median(accelerator_us[1:]):.0f} "
        f"netrun_us={statistics.median(netrun_us[1:]):.0f}"
    )


def main() -> None:
    summarize("prefill")
    summarize("decode")


if __name__ == "__main__":
    main()
