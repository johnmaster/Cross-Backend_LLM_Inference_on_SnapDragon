#!/usr/bin/env python3
"""Generate a standalone prefill q_proj fixture for the custom MatMul op."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_OUTPUT_DIR = ROOT / "test_data" / "q_proj_prefill"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tiny_llm_block.reference import TinyBlockConfig, TinyDecoderBlock, rms_norm


def _round_half_away_from_zero(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, value + 0.5, value - 0.5).astype(np.int64)


def _to_q13(value: np.ndarray) -> np.ndarray:
    scaled = value.astype(np.float32) * np.float32(8192.0)
    rounded = _round_half_away_from_zero(scaled)
    return np.clip(rounded, -32768, 32767).astype(np.int32)


def _q13_matmul(lhs_fp16: np.ndarray, rhs_fp16: np.ndarray) -> np.ndarray:
    lhs_q13 = _to_q13(lhs_fp16.astype(np.float32))
    rhs_q13 = _to_q13(rhs_fp16.astype(np.float32))
    acc = np.matmul(lhs_q13.astype(np.int64), rhs_q13.astype(np.int64))
    return (acc.astype(np.float32) * np.float32(2.0 ** -26)).astype(np.float32)


def save_raw(path: Path, value: np.ndarray, dtype: np.dtype) -> None:
    np.asarray(value, dtype=dtype).tofile(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = TinyBlockConfig()
    block = TinyDecoderBlock(config, seed=42)
    rng = np.random.default_rng(7)
    hidden = rng.normal(0.0, 0.2, (1, 32, config.hidden_size)).astype(np.float32)
    normalized = rms_norm(hidden, block.input_norm_weight, config.rms_norm_eps)

    lhs = normalized.reshape(1, 1, 32, 256).astype(np.float32)
    rhs = block.q_proj_weight.reshape(1, 1, 256, 256).astype(np.float32)
    lhs_fp16 = lhs.astype(np.float16)
    rhs_fp16 = rhs.astype(np.float16)

    expected_fp32_baseline = np.matmul(lhs, rhs).astype(np.float32)
    expected_fp16_input = np.matmul(
        lhs_fp16.astype(np.float32),
        rhs_fp16.astype(np.float32),
    ).astype(np.float32)
    expected_q13 = _q13_matmul(lhs_fp16, rhs_fp16)

    save_raw(args.output_dir / "lhs.raw", lhs, np.float32)
    save_raw(args.output_dir / "rhs.raw", rhs, np.float32)
    save_raw(args.output_dir / "lhs_fp16.raw", lhs_fp16, np.float16)
    save_raw(args.output_dir / "rhs_fp16.raw", rhs_fp16, np.float16)
    save_raw(args.output_dir / "expected_fp32_baseline.raw", expected_fp32_baseline, np.float32)
    save_raw(args.output_dir / "expected_fp16_input.raw", expected_fp16_input, np.float32)
    save_raw(args.output_dir / "expected_q13_kernel.raw", expected_q13, np.float32)

    (args.output_dir / "input_list.txt").write_text(
        f"lhs:={args.output_dir / 'lhs.raw'} rhs:={args.output_dir / 'rhs.raw'}\n",
        encoding="ascii",
    )
    (args.output_dir / "device_input_list.txt").write_text(
        "lhs:=input/lhs.raw rhs:=input/rhs.raw\n",
        encoding="ascii",
    )
    (args.output_dir / "sample_app_input_list.txt").write_text(
        "lhs:=tiny_llm_block_custom_matmul/input/lhs.raw "
        "rhs:=tiny_llm_block_custom_matmul/input/rhs.raw\n",
        encoding="ascii",
    )

    diff = np.abs(expected_fp16_input - expected_q13)
    print("fixture:", args.output_dir)
    print("lhs shape:", lhs.shape, "range:", float(lhs.min()), float(lhs.max()))
    print("rhs shape:", rhs.shape, "range:", float(rhs.min()), float(rhs.max()))
    print("expected shape:", expected_q13.shape)
    print("fp16_input_vs_q13 max_abs_error:", float(diff.max()))
    print("fp16_input_vs_q13 mean_abs_error:", float(diff.mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
