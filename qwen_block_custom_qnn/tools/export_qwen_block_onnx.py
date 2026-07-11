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
    def __init__(self, model_dir: Path, seq_len: int) -> None:
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
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("rope_cos", torch.cos(freqs).contiguous())
        self.register_buffer("rope_sin", torch.sin(freqs).contiguous())

        causal_mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        self.register_buffer("causal_mask", causal_mask.view(1, 1, seq_len, seq_len).contiguous())

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

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        present_key = self.apply_rotary(k)
        present_value = v

        key_for_attention = present_key.repeat_interleave(self.num_kv_groups, dim=1)
        value_for_attention = present_value.repeat_interleave(self.num_kv_groups, dim=1)
        scores = torch.matmul(q, key_for_attention.transpose(-2, -1)) / math.sqrt(self.head_dim)
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

        return hidden_states, present_key, present_value


def export(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    model = Qwen2DecoderLayer0(args.model_dir, args.seq_len).eval()
    hidden_states = torch.randn(1, args.seq_len, model.hidden_size, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(hidden_states)

    args.test_data_dir.mkdir(parents=True, exist_ok=True)
    hidden_states.numpy().astype(np.float32).tofile(args.test_data_dir / "hidden_states.raw")
    for name, tensor in zip(["hidden_out", "present_key", "present_value"], outputs):
        tensor.numpy().astype(np.float32).tofile(args.test_data_dir / f"{name}.raw")
    (args.test_data_dir / "input_list.txt").write_text(
        f"hidden_states:={args.test_data_dir / 'hidden_states.raw'}\n",
        encoding="utf-8",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (hidden_states,),
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["hidden_states"],
        output_names=["hidden_out", "present_key", "present_value"],
    )

    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        ort_outputs = session.run(None, {"hidden_states": hidden_states.numpy()})
        for name, expected, actual in zip(["hidden_out", "present_key", "present_value"], outputs, ort_outputs):
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
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
