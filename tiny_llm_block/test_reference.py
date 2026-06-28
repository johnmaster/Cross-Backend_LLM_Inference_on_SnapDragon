#!/usr/bin/env python3

import unittest

import numpy as np

from tiny_llm_block.reference import TinyBlockConfig, TinyDecoderBlock


class TinyDecoderBlockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TinyBlockConfig(
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
        )
        self.block = TinyDecoderBlock(self.config, seed=11)
        rng = np.random.default_rng(23)
        self.inputs = rng.normal(0.0, 0.2, (2, 6, 32)).astype(np.float32)

    def test_shapes_and_cache(self) -> None:
        output, cache = self.block.forward(self.inputs)
        self.assertEqual(output.shape, self.inputs.shape)
        self.assertEqual(cache[0].shape, (2, 2, 6, 8))
        self.assertEqual(cache[1].shape, (2, 2, 6, 8))
        self.assertEqual(output.dtype, np.float32)

    def test_prefill_matches_token_by_token_decode(self) -> None:
        prefill_output, _ = self.block.forward(self.inputs)
        cache = None
        decode_outputs = []
        for position in range(self.inputs.shape[1]):
            output, cache = self.block.forward(
                self.inputs[:, position : position + 1], cache
            )
            decode_outputs.append(output)
        decode_output = np.concatenate(decode_outputs, axis=1)
        np.testing.assert_allclose(
            prefill_output, decode_output, rtol=2e-5, atol=2e-6
        )

    def test_future_tokens_do_not_change_past_outputs(self) -> None:
        original, _ = self.block.forward(self.inputs)
        changed_inputs = self.inputs.copy()
        changed_inputs[:, 3:] += np.float32(10.0)
        changed, _ = self.block.forward(changed_inputs)
        np.testing.assert_allclose(
            original[:, :3], changed[:, :3], rtol=0.0, atol=0.0
        )

    def test_rejects_invalid_hidden_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "wrong hidden dimension"):
            self.block.forward(np.zeros((1, 1, 31), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
