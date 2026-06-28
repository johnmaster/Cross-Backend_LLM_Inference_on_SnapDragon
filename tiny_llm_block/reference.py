#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


Array = np.ndarray
KvCache = Tuple[Array, Array]


@dataclass(frozen=True)
class TinyBlockConfig:
    hidden_size: int = 256
    intermediate_size: int = 768
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 32
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")


def rms_norm(x: Array, weight: Array, eps: float) -> Array:
    variance = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True)
    return (x * np.reciprocal(np.sqrt(variance + eps)) * weight).astype(
        np.float32
    )


def silu(x: Array) -> Array:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def softmax(x: Array, axis: int = -1) -> Array:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(shifted)
    return (numerator / np.sum(numerator, axis=axis, keepdims=True)).astype(
        np.float32
    )


class TinyDecoderBlock:
    """A deterministic FP32 Qwen-style decoder block implemented in NumPy."""

    def __init__(self, config: TinyBlockConfig, seed: int = 42) -> None:
        self.config = config
        rng = np.random.default_rng(seed)

        def weight(input_size: int, output_size: int) -> Array:
            return rng.normal(0.0, 0.02, (input_size, output_size)).astype(
                np.float32
            )

        hidden = config.hidden_size
        kv_size = config.num_kv_heads * config.head_dim
        intermediate = config.intermediate_size

        self.input_norm_weight = np.ones(hidden, dtype=np.float32)
        self.post_attention_norm_weight = np.ones(hidden, dtype=np.float32)
        self.q_proj_weight = weight(hidden, hidden)
        self.k_proj_weight = weight(hidden, kv_size)
        self.v_proj_weight = weight(hidden, kv_size)
        self.o_proj_weight = weight(hidden, hidden)
        self.q_proj_bias = np.zeros(hidden, dtype=np.float32)
        self.k_proj_bias = np.zeros(kv_size, dtype=np.float32)
        self.v_proj_bias = np.zeros(kv_size, dtype=np.float32)
        self.gate_proj_weight = weight(hidden, intermediate)
        self.up_proj_weight = weight(hidden, intermediate)
        self.down_proj_weight = weight(intermediate, hidden)

    def _split_heads(self, x: Array, num_heads: int) -> Array:
        batch, sequence, _ = x.shape
        return x.reshape(
            batch, sequence, num_heads, self.config.head_dim
        ).transpose(0, 2, 1, 3)

    def _apply_rope(self, x: Array, positions: Array) -> Array:
        half = self.config.head_dim // 2
        frequencies = np.power(
            np.float32(self.config.rope_theta),
            -np.arange(0, half, dtype=np.float32) / np.float32(half),
        )
        angles = positions.astype(np.float32)[:, None] * frequencies[None, :]
        cos = np.cos(angles)[None, None, :, :]
        sin = np.sin(angles)[None, None, :, :]
        first, second = x[..., :half], x[..., half:]
        return np.concatenate(
            (first * cos - second * sin, second * cos + first * sin), axis=-1
        ).astype(np.float32)

    def _attention(
        self, hidden_states: Array, cache: Optional[KvCache]
    ) -> Tuple[Array, KvCache]:
        config = self.config
        batch, sequence, _ = hidden_states.shape
        past_length = 0 if cache is None else cache[0].shape[2]
        positions = np.arange(
            past_length, past_length + sequence, dtype=np.int64
        )

        query = self._split_heads(
            hidden_states @ self.q_proj_weight + self.q_proj_bias,
            config.num_heads,
        )
        key = self._split_heads(
            hidden_states @ self.k_proj_weight + self.k_proj_bias,
            config.num_kv_heads,
        )
        value = self._split_heads(
            hidden_states @ self.v_proj_weight + self.v_proj_bias,
            config.num_kv_heads,
        )
        query = self._apply_rope(query, positions)
        key = self._apply_rope(key, positions)

        if cache is not None:
            key = np.concatenate((cache[0], key), axis=2)
            value = np.concatenate((cache[1], value), axis=2)
        new_cache = (key, value)

        repeats = config.num_heads // config.num_kv_heads
        expanded_key = np.repeat(key, repeats, axis=1)
        expanded_value = np.repeat(value, repeats, axis=1)
        scores = query @ expanded_key.transpose(0, 1, 3, 2)
        scores *= np.float32(1.0 / np.sqrt(config.head_dim))

        key_positions = np.arange(key.shape[2], dtype=np.int64)
        causal_mask = key_positions[None, :] <= positions[:, None]
        scores = np.where(causal_mask[None, None, :, :], scores, -np.inf)
        context = softmax(scores) @ expanded_value
        context = context.transpose(0, 2, 1, 3).reshape(
            batch, sequence, config.hidden_size
        )
        return (context @ self.o_proj_weight).astype(np.float32), new_cache

    def forward(
        self, hidden_states: Array, cache: Optional[KvCache] = None
    ) -> Tuple[Array, KvCache]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden_states has the wrong hidden dimension")

        normalized = rms_norm(
            hidden_states,
            self.input_norm_weight,
            self.config.rms_norm_eps,
        )
        attention_output, new_cache = self._attention(normalized, cache)
        residual = (hidden_states + attention_output).astype(np.float32)

        normalized = rms_norm(
            residual,
            self.post_attention_norm_weight,
            self.config.rms_norm_eps,
        )
        gate = silu(normalized @ self.gate_proj_weight)
        up = normalized @ self.up_proj_weight
        mlp_output = (gate * up) @ self.down_proj_weight
        return (residual + mlp_output).astype(np.float32), new_cache


def main() -> None:
    config = TinyBlockConfig()
    block = TinyDecoderBlock(config)
    rng = np.random.default_rng(7)
    hidden_states = rng.normal(
        0.0, 0.2, (1, 32, config.hidden_size)
    ).astype(np.float32)

    prefill_output, cache = block.forward(hidden_states)
    decode_output, cache = block.forward(hidden_states[:, :1], None)
    for position in range(1, hidden_states.shape[1]):
        decode_output, cache = block.forward(
            hidden_states[:, position : position + 1], cache
        )

    difference = np.abs(prefill_output[:, -1:] - decode_output)
    print(f"prefill output: {prefill_output.shape}")
    print(f"KV cache key:   {cache[0].shape}")
    print(f"KV cache value: {cache[1].shape}")
    print(f"last-token max error:  {np.max(difference):.8e}")
    print(f"last-token mean error: {np.mean(difference):.8e}")


if __name__ == "__main__":
    main()
