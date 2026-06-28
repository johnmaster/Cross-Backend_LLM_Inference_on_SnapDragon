#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "05_quantized_matmul" / "input" / "rhs_float.raw"
RHS_SHAPE = (1, 1, 256, 256)


def main():
    rhs = np.fromfile(SOURCE, np.float32).reshape(RHS_SHAPE)
    output_dir = ROOT / "overrides"
    output_dir.mkdir(parents=True, exist_ok=True)

    for block_size in (32, 64, 128):
        block_count = RHS_SHAPE[2] // block_size
        blocked = rhs.reshape(1, 1, block_count, block_size, RHS_SHAPE[3])
        # Signed INT4 uses [-8, 7]; symmetric scale uses the positive bound.
        scales = np.max(np.abs(blocked), axis=3) / np.float32(7.0)
        scales = np.maximum(scales, np.finfo(np.float32).tiny)
        zero_points = np.zeros_like(scales, dtype=np.int32)
        overrides = {
            "version": "2.0.0",
            "encodings": [
                {
                    "name": "rhs",
                    "output_dtype": "int4",
                    "y_scale": scales.tolist(),
                    "y_zero_point": zero_points.tolist(),
                    "axis": 2,
                    "block_size": block_size,
                }
            ],
        }
        path = output_dir / f"block_{block_size}.json"
        path.write_text(json.dumps(overrides, indent=2) + "\n", encoding="ascii")
        print(
            f"{path}: blocks={block_count}, encodings={scales.size}, "
            f"scale=[{scales.min():.8g}, {scales.max():.8g}]"
        )


if __name__ == "__main__":
    main()
