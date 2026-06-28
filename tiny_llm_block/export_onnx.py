#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from tiny_llm_block.reference import TinyBlockConfig, TinyDecoderBlock


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "model"
DEFAULT_DATA_DIR = ROOT / "test_data"


class TorchTinyDecoderBlock(nn.Module):
    def __init__(self, reference: TinyDecoderBlock) -> None:
        super().__init__()
        self.config = reference.config
        for name, value in vars(reference).items():
            if isinstance(value, np.ndarray):
                self.register_buffer(name, torch.from_numpy(value.copy()))

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.config.rms_norm_eps) * weight

    def _split_heads(
        self, x: torch.Tensor, num_heads: int
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape
        return x.reshape(
            batch, sequence, num_heads, self.config.head_dim
        ).permute(0, 2, 1, 3)

    def _rope(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        half = self.config.head_dim // 2
        dimensions = torch.arange(half, dtype=torch.float32, device=x.device)
        frequencies = self.config.rope_theta ** (-dimensions / half)
        angles = positions.to(torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        first, second = x[..., :half], x[..., half:]
        return torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self.config
        past_length = 0 if past_key is None else past_key.shape[2]
        sequence = hidden_states.shape[1]
        positions = torch.arange(
            past_length,
            past_length + sequence,
            dtype=torch.int64,
            device=hidden_states.device,
        )

        normalized = self._rms_norm(
            hidden_states, self.input_norm_weight
        )
        query = self._split_heads(
            normalized @ self.q_proj_weight + self.q_proj_bias,
            config.num_heads,
        )
        key = self._split_heads(
            normalized @ self.k_proj_weight + self.k_proj_bias,
            config.num_kv_heads,
        )
        value = self._split_heads(
            normalized @ self.v_proj_weight + self.v_proj_bias,
            config.num_kv_heads,
        )
        query = self._rope(query, positions)
        key = self._rope(key, positions)

        if past_key is not None:
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        repeats = config.num_heads // config.num_kv_heads
        expanded_key = torch.repeat_interleave(key, repeats, dim=1)
        expanded_value = torch.repeat_interleave(value, repeats, dim=1)
        scores = query @ expanded_key.transpose(-1, -2)
        scores = scores * (config.head_dim**-0.5)
        key_positions = torch.arange(
            key.shape[2], dtype=torch.int64, device=hidden_states.device
        )
        causal_mask = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
        scores = torch.where(
            causal_mask.unsqueeze(0).unsqueeze(0),
            scores,
            torch.full_like(scores, -1.0e4),
        )
        context = torch.softmax(scores, dim=-1) @ expanded_value
        context = context.permute(0, 2, 1, 3).reshape(
            hidden_states.shape[0], sequence, config.hidden_size
        )
        attention_output = context @ self.o_proj_weight
        residual = hidden_states + attention_output

        normalized = self._rms_norm(
            residual, self.post_attention_norm_weight
        )
        gate_input = normalized @ self.gate_proj_weight
        gate = gate_input * torch.sigmoid(gate_input)
        up = normalized @ self.up_proj_weight
        output = residual + (gate * up) @ self.down_proj_weight
        return output, key, value


def export_model(
    model: TorchTinyDecoderBlock,
    inputs: Tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
) -> None:
    torch.onnx.export(
        model,
        inputs,
        path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["hidden_out", "present_key", "present_value"],
    )
    loaded = onnx.load(path)
    onnx.checker.check_model(loaded)
    loaded.ir_version = min(loaded.ir_version, 10)
    onnx.save(loaded, path)


def save_raw(path: Path, value: np.ndarray) -> None:
    np.asarray(value, dtype=np.float32).tofile(path)


def validate(
    path: Path,
    inputs: dict[str, np.ndarray],
    expected: Tuple[np.ndarray, ...],
) -> None:
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, inputs)
    print(path.name)
    for name, reference, result in zip(
        ("hidden_out", "present_key", "present_value"), expected, actual
    ):
        difference = np.abs(reference - result)
        print(
            f"  {name:13s} max={np.max(difference):.8e} "
            f"mean={np.mean(difference):.8e}"
        )
        np.testing.assert_allclose(reference, result, rtol=2e-5, atol=2e-6)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export fixed-shape tiny decoder block ONNX models."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    config = TinyBlockConfig()
    reference = TinyDecoderBlock(config, seed=42)
    model = TorchTinyDecoderBlock(reference).eval()
    rng = np.random.default_rng(7)
    hidden = rng.normal(0.0, 0.2, (1, 32, config.hidden_size)).astype(
        np.float32
    )
    next_token = rng.normal(0.0, 0.2, (1, 1, config.hidden_size)).astype(
        np.float32
    )

    prefill_expected = reference.forward(hidden)
    decode_expected = reference.forward(next_token, prefill_expected[1])
    prefill_path = args.model_dir / "tiny_block_prefill_seq32.onnx"
    decode_path = args.model_dir / "tiny_block_decode_past32.onnx"

    with torch.no_grad():
        export_model(
            model,
            (torch.from_numpy(hidden),),
            prefill_path,
            ["hidden_states"],
        )
        export_model(
            model,
            tuple(
                torch.from_numpy(value)
                for value in (
                    next_token,
                    prefill_expected[1][0],
                    prefill_expected[1][1],
                )
            ),
            decode_path,
            ["hidden_states", "past_key", "past_value"],
        )

    save_raw(args.data_dir / "prefill_hidden.raw", hidden)
    save_raw(args.data_dir / "prefill_expected.raw", prefill_expected[0])
    save_raw(args.data_dir / "prefill_expected_key.raw", prefill_expected[1][0])
    save_raw(
        args.data_dir / "prefill_expected_value.raw", prefill_expected[1][1]
    )
    save_raw(args.data_dir / "decode_hidden.raw", next_token)
    save_raw(args.data_dir / "decode_past_key.raw", prefill_expected[1][0])
    save_raw(args.data_dir / "decode_past_value.raw", prefill_expected[1][1])
    save_raw(args.data_dir / "decode_expected.raw", decode_expected[0])
    save_raw(args.data_dir / "decode_expected_key.raw", decode_expected[1][0])
    save_raw(args.data_dir / "decode_expected_value.raw", decode_expected[1][1])

    (args.data_dir / "prefill_input_list.txt").write_text(
        f"hidden_states:={args.data_dir / 'prefill_hidden.raw'}\n",
        encoding="ascii",
    )
    (args.data_dir / "decode_input_list.txt").write_text(
        " ".join(
            (
                f"hidden_states:={args.data_dir / 'decode_hidden.raw'}",
                f"past_key:={args.data_dir / 'decode_past_key.raw'}",
                f"past_value:={args.data_dir / 'decode_past_value.raw'}",
            )
        )
        + "\n",
        encoding="ascii",
    )

    validate(
        prefill_path,
        {"hidden_states": hidden},
        (prefill_expected[0], *prefill_expected[1]),
    )
    validate(
        decode_path,
        {
            "hidden_states": next_token,
            "past_key": prefill_expected[1][0],
            "past_value": prefill_expected[1][1],
        },
        (decode_expected[0], *decode_expected[1]),
    )


if __name__ == "__main__":
    main()
