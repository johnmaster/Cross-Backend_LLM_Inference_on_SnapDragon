#!/usr/bin/env python3
"""Validate blockwise online softmax against the materialized reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from generate_fixture import GROUPS, HEAD_DIM, KV_HEADS, SCALE, attention_reference


def online_attention(
    query: np.ndarray, key: np.ndarray, value: np.ndarray, block_size: int
) -> np.ndarray:
    batch, query_heads, query_length, head_dim = query.shape
    kv_heads = KV_HEADS
    kv_length = key.shape[1]
    output = np.empty_like(query)
    for b in range(batch):
        for query_head in range(query_heads):
            h = query_head // GROUPS
            for q_index in range(query_length):
                    q = query[b, query_head, q_index]
                    running_max = np.float32(-np.inf)
                    running_sum = np.float32(0.0)
                    accumulator = np.zeros(head_dim, dtype=np.float32)
                    for begin in range(0, kv_length, block_size):
                        end = min(begin + block_size, kv_length)
                        scores = key[b, begin:end, :, h] @ q
                        scores = scores.astype(np.float32) * np.float32(SCALE)
                        block_max = np.max(scores)
                        next_max = max(running_max, block_max)
                        old_scale = np.exp(running_max - next_max).astype(np.float32)
                        weights = np.exp(scores - next_max).astype(np.float32)
                        accumulator *= old_scale
                        accumulator += weights @ value[b, h, begin:end]
                        running_sum = running_sum * old_scale + np.sum(weights)
                        running_max = next_max
                    output[b, query_head, q_index] = accumulator / running_sum
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-length", type=int, default=129)
    parser.add_argument("--query-length", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device-output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    query = rng.normal(0.0, 0.25, (1, KV_HEADS * GROUPS, args.query_length, HEAD_DIM)).astype(
        np.float32
    )
    key = rng.normal(0.0, 0.25, (1, args.kv_length, HEAD_DIM, KV_HEADS)).astype(np.float32)
    value = rng.normal(0.0, 0.25, (1, KV_HEADS, args.kv_length, HEAD_DIM)).astype(np.float32)
    expected = attention_reference(query, key, value)
    actual = online_attention(query, key, value, args.block_size)
    difference = np.abs(actual - expected)
    print(f"online max_abs_error={difference.max():.9g}")
    print(f"online mean_abs_error={difference.mean():.9g}")
    print(f"online cosine={np.dot(actual.ravel(), expected.ravel()) / (np.linalg.norm(actual) * np.linalg.norm(expected)):.9g}")
    if not np.allclose(actual, expected, atol=2e-6, rtol=2e-5):
        raise SystemExit("online reference mismatch")

    if args.device_output:
        device = np.fromfile(args.device_output, dtype=np.float32).reshape(expected.shape)
        device_difference = np.abs(device - expected)
        print(f"device max_abs_error={device_difference.max():.9g}")
        print(f"device mean_abs_error={device_difference.mean():.9g}")
        if not np.allclose(device, expected, atol=2e-5, rtol=2e-4):
            raise SystemExit("device output mismatch")


if __name__ == "__main__":
    main()
