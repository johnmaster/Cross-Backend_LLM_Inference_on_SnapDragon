#!/usr/bin/env python3
"""Export one real Qwen2 decoder layer to fixed-shape ONNX.

This script intentionally avoids `transformers` model loading. The current
host environment has a torch/torchvision mismatch that breaks Qwen2 import, and
for this case study we only need layer weights plus the Qwen2 block math.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "model" / "data" / "models" / "Qwen2.5-0.5B-Instruct"
DEFAULT_ONNX = ROOT / "model" / "qwen2_0_5b_layer0_prefill_seq16.onnx"
DEFAULT_TEST_DATA = ROOT / "test_data" / "layer0_prefill_seq16"


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return F.linear(x, weight, bias)


class Qwen2DecoderLayer0(torch.nn.Module):
    def __init__(
        self,
        model_dir: Path,
        seq_len: int,
        past_len: int = 0,
        grouped_gqa: bool = False,
        delta_kv_output: bool = False,
        fp16_kv_cache: bool = False,
    ) -> None:
        super().__init__()
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.hidden_size = int(config["hidden_size"])
        self.intermediate_size = int(config["intermediate_size"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.rms_norm_eps = float(config["rms_norm_eps"])
        self.seq_len = seq_len
        self.past_len = past_len
        self.grouped_gqa = grouped_gqa
        self.delta_kv_output = delta_kv_output
        self.fp16_kv_cache = fp16_kv_cache

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        tensor_path = model_dir / "model.safetensors"
        prefix = "model.layers.0."
        with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
            for name in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.weight",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ]:
                tensor = tensors.get_tensor(prefix + name).to(torch.float32).contiguous()
                self.register_buffer(name.replace(".", "_"), tensor)

        inv_freq = 1.0 / (
            float(config["rope_theta"])
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(
            past_len, past_len + seq_len, dtype=torch.float32
        )
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("rope_cos", torch.cos(freqs).contiguous())
        self.register_buffer("rope_sin", torch.sin(freqs).contiguous())

        total_len = past_len + seq_len
        causal_mask = torch.full(
            (seq_len, total_len), float("-inf"), dtype=torch.float32
        )
        causal_mask = torch.triu(causal_mask, diagonal=past_len + 1)
        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, seq_len, total_len).contiguous(),
        )

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.rms_norm_eps) * weight

    def apply_rotary(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, heads, seq, head_dim]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        cos = self.rope_cos.view(1, 1, self.seq_len, self.head_dim // 2)
        sin = self.rope_sin.view(1, 1, self.seq_len, self.head_dim // 2)
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normed = self.rms_norm(hidden_states, self.input_layernorm_weight)

        batch = normed.shape[0]
        q = linear(normed, self.self_attn_q_proj_weight, self.self_attn_q_proj_bias)
        k = linear(normed, self.self_attn_k_proj_weight, self.self_attn_k_proj_bias)
        v = linear(normed, self.self_attn_v_proj_weight, self.self_attn_v_proj_bias)

        q = q.view(batch, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, self.seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, self.seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.apply_rotary(q)
        current_key = self.apply_rotary(k)
        current_value = v
        if past_key is None:
            present_key = current_key
            present_value = current_value
        else:
            # Keep attention math in FP32 while allowing a compact FP16 cache
            # at the graph boundary.
            if self.fp16_kv_cache:
                past_key = past_key.to(torch.float32)
                past_value = past_value.to(torch.float32)
            present_key = torch.cat((past_key, current_key), dim=2)
            present_value = torch.cat((past_value, current_value), dim=2)

        if self.grouped_gqa:
            # Express GQA as [kv_head, query_group] broadcasting. This avoids
            # materializing a seven-times-larger K/V tensor with Tile.
            grouped_q = q.reshape(
                batch,
                self.num_kv_heads,
                self.num_kv_groups,
                self.seq_len,
                self.head_dim,
            )
            grouped_key = present_key.transpose(-2, -1).unsqueeze(2)
            scores = torch.matmul(grouped_q, grouped_key) / math.sqrt(
                self.head_dim
            )
            grouped_mask = self.causal_mask.unsqueeze(2)
            probs = torch.softmax(scores + grouped_mask, dim=-1)
            grouped_value = present_value.unsqueeze(2)
            attn_output = torch.matmul(probs, grouped_value).reshape(
                batch, self.num_heads, self.seq_len, self.head_dim
            )
        else:
            key_for_attention = present_key.repeat_interleave(
                self.num_kv_groups, dim=1
            )
            value_for_attention = present_value.repeat_interleave(
                self.num_kv_groups, dim=1
            )
            scores = torch.matmul(
                q, key_for_attention.transpose(-2, -1)
            ) / math.sqrt(self.head_dim)
            scores = scores + self.causal_mask
            probs = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(probs, value_for_attention)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, self.seq_len, self.hidden_size)
        hidden_states = residual + linear(attn_output, self.self_attn_o_proj_weight)

        residual = hidden_states
        normed = self.rms_norm(hidden_states, self.post_attention_layernorm_weight)
        gate = linear(normed, self.mlp_gate_proj_weight)
        up = linear(normed, self.mlp_up_proj_weight)
        mlp_output = linear(F.silu(gate) * up, self.mlp_down_proj_weight)
        hidden_states = residual + mlp_output

        if self.delta_kv_output:
            if self.fp16_kv_cache:
                return hidden_states, current_key.to(torch.float16), current_value.to(
                    torch.float16
                )
            return hidden_states, current_key, current_value
        return hidden_states, present_key, present_value


def export(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    model = Qwen2DecoderLayer0(
        args.model_dir,
        args.seq_len,
        args.past_len,
        args.grouped_gqa,
        args.delta_kv_output,
        args.fp16_kv_cache,
    ).eval()
    if args.past_len:
        prefill_model = Qwen2DecoderLayer0(
            args.model_dir, args.past_len
        ).eval()
        prefill_hidden = torch.randn(
            1, args.past_len, model.hidden_size, dtype=torch.float32
        )
        with torch.no_grad():
            _, past_key, past_value = prefill_model(prefill_hidden)
        if args.fp16_kv_cache:
            past_key = past_key.to(torch.float16)
            past_value = past_value.to(torch.float16)
        hidden_states = torch.randn(
            1, args.seq_len, model.hidden_size, dtype=torch.float32
        )
        model_inputs = (hidden_states, past_key, past_value)
        input_names = ["hidden_states", "past_key", "past_value"]
    else:
        past_key = None
        past_value = None
        hidden_states = torch.randn(
            1, args.seq_len, model.hidden_size, dtype=torch.float32
        )
        model_inputs = (hidden_states,)
        input_names = ["hidden_states"]

    with torch.no_grad():
        outputs = model(*model_inputs)

    args.test_data_dir.mkdir(parents=True, exist_ok=True)
    hidden_states.numpy().astype(np.float32).tofile(args.test_data_dir / "hidden_states.raw")
    if past_key is not None:
        cache_dtype = np.float16 if args.fp16_kv_cache else np.float32
        past_key.numpy().astype(cache_dtype).tofile(
            args.test_data_dir / "past_key.raw"
        )
        past_value.numpy().astype(cache_dtype).tofile(
            args.test_data_dir / "past_value.raw"
        )
    kv_output_names = (
        ["current_key", "current_value"]
        if args.delta_kv_output
        else ["present_key", "present_value"]
    )
    output_names = ["hidden_out", *kv_output_names]
    for name, tensor in zip(output_names, outputs):
        dtype = (
            np.float16
            if args.fp16_kv_cache and name in kv_output_names
            else np.float32
        )
        tensor.numpy().astype(dtype).tofile(args.test_data_dir / f"{name}.raw")
    input_entries = [
        f"hidden_states:={args.test_data_dir / 'hidden_states.raw'}"
    ]
    if past_key is not None:
        input_entries.extend(
            [
                f"past_key:={args.test_data_dir / 'past_key.raw'}",
                f"past_value:={args.test_data_dir / 'past_value.raw'}",
            ]
        )
    (args.test_data_dir / "input_list.txt").write_text(
        " ".join(input_entries) + "\n", encoding="utf-8"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        model_inputs,
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )

    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        ort_inputs = {"hidden_states": hidden_states.numpy()}
        if past_key is not None:
            ort_inputs["past_key"] = past_key.numpy()
            ort_inputs["past_value"] = past_value.numpy()
        ort_outputs = session.run(None, ort_inputs)
        for name, expected, actual in zip(output_names, outputs, ort_outputs):
            diff = np.abs(expected.numpy() - actual)
            print(
                name,
                "max_abs",
                float(diff.max()),
                "mean_abs",
                float(diff.mean()),
                "allclose",
                bool(np.allclose(expected.numpy(), actual, atol=1e-4, rtol=1e-4)),
            )
    except ImportError:
        print("onnx/onnxruntime not available; skipped ONNX validation")

    print(args.output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--test-data-dir", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--past-len", type=int, default=0)
    parser.add_argument(
        "--grouped-gqa",
        action="store_true",
        help="broadcast over KV groups instead of explicitly repeating K/V",
    )
    parser.add_argument(
        "--delta-kv-output",
        action="store_true",
        help="output only the current token K/V; the host owns persistent cache",
    )
    parser.add_argument(
        "--fp16-kv-cache",
        action="store_true",
        help="use FP16 past/current KV at graph boundaries; attention stays FP32",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
