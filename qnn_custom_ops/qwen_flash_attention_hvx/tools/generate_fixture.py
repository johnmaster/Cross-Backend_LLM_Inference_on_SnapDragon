#!/usr/bin/env python3
"""Generate deterministic Qwen2.5 GQA decode-attention fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


HEAD_DIM = 64
KV_HEADS = 2
GROUPS = 7
SCALE = HEAD_DIM**-0.5


def attention_reference(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    batch, query_heads, query_length, head_dim = query.shape
    query_gqa = query.reshape(batch, KV_HEADS, GROUPS, query_length, head_dim)
    key_nchw = np.transpose(key, (0, 3, 1, 2))
    scores = np.einsum("bhgqd,bhkd->bhgqk", query_gqa, key_nchw, dtype=np.float32)
    scores *= np.float32(SCALE)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores).astype(np.float32)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    output = np.einsum("bhgqk,bhkd->bhgqd", probabilities, value, dtype=np.float32)
    return output.reshape(batch, query_heads, query_length, head_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-length", type=int, default=129)
    parser.add_argument("--query-length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qnn_custom_ops/qwen_flash_attention_hvx/test_data/past128"),
    )
    args = parser.parse_args()
    if args.kv_length <= 0 or args.query_length <= 0:
        raise ValueError("lengths must be positive")

    rng = np.random.default_rng(args.seed)
    query = rng.normal(0.0, 0.25, (1, KV_HEADS * GROUPS, args.query_length, HEAD_DIM)).astype(
        np.float32
    )
    key = rng.normal(0.0, 0.25, (1, args.kv_length, HEAD_DIM, KV_HEADS)).astype(np.float32)
    value = rng.normal(0.0, 0.25, (1, KV_HEADS, args.kv_length, HEAD_DIM)).astype(np.float32)
    expected = attention_reference(query, key, value)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    query.tofile(args.output_dir / "query.raw")
    key.tofile(args.output_dir / "key.raw")
    value.tofile(args.output_dir / "value.raw")
    expected.tofile(args.output_dir / "expected.raw")
    (args.output_dir / "input_list.txt").write_text(
        f"query:={args.output_dir / 'query.raw'} "
        f"key:={args.output_dir / 'key.raw'} "
        f"value:={args.output_dir / 'value.raw'}\n",
        encoding="utf-8",
    )
    print(f"query={query.shape} key={key.shape} value={value.shape} output={expected.shape}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
