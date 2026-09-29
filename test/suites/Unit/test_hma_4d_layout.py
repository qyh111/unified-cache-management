"""FAWA KVCacheGroupLayout 4-D page classification tests (requires torch).

Covers the DeepseekV4 vLLM 0.30 MLA pages [num_blocks, H=1, states, bytes]
alongside the legacy combined-KV and plain 4-D layouts. Addresses are checked
against torch's own strided-view data_ptr as the oracle. Run with an
interpreter that has torch and vllm installed (e.g. the 0.30 CPU venv):
python test/suites/Unit/test_hma_4d_layout.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ucm.integration.vllm.hma_connector import KVCacheGroupLayout


def as_strided_page(shape, strides, dtype=torch.uint8):
    """Build a dense buffer and view it with the given element strides."""

    span = 1 + max(
        sum((d - 1) * s for d, s in zip(shape, strides) if d > 0), 0
    )
    flat = torch.zeros(span, dtype=dtype)
    return torch.as_strided(flat, tuple(shape), tuple(strides))


class UnifiedMLAPageTests(unittest.TestCase):
    def test_compressed_page_rows_scale_with_scheduler_block(self):
        # fp8 MLA page: block_size=256 tokens, compress ratio 4 -> 64 states,
        # 584B per state (448 NoPE + 128 RoPE + 8 scale).
        tensor = as_strided_page(
            (16, 1, 64, 584), (1002240, 37376, 584, 1)
        )
        layout = KVCacheGroupLayout(
            {"model.layers.2.attn": tensor}, expected_block_size=256
        )
        self.assertEqual(layout.tensor_block_sizes.tolist(), [64])
        self.assertEqual(layout.tensor_token_strides.tolist(), [584])
        self.assertEqual(layout.tensor_sizes_per_token.tolist(), [584])
        self.assertEqual(layout.block_strides.tolist(), [1002240])

        # High compression ratio layers store 2 states per 256-token block.
        sparse = as_strided_page((16, 1, 2, 584), (1002240, 1168, 584, 1))
        layout = KVCacheGroupLayout(
            {"model.layers.3.attn": sparse}, expected_block_size=256
        )
        self.assertEqual(layout.tensor_block_sizes.tolist(), [2])
        self.assertEqual(
            layout.segment_tensor_size_list(256, 256), [584 * 2]
        )
        # A half-block request covers half the states (ratio-4 alignment).
        self.assertEqual(
            layout.segment_tensor_size_list(128, 256), [584]
        )

    def test_block_addresses_match_torch_views(self):
        tensor = as_strided_page((16, 1, 64, 584), (1002240, 37376, 584, 1))
        layout = KVCacheGroupLayout(
            {"model.layers.2.attn": tensor}, expected_block_size=256
        )
        for block_id in (0, 1, 7, 15):
            parsed = int(layout.extract_addrs(np.array([block_id]))[0, 0])
            self.assertEqual(parsed, tensor[block_id, 0].data_ptr())
        # With a token offset of one full block the address advances by the
        # state count times the state stride.
        parsed = int(
            layout.extract_addrs_with_offsets(
                np.array([3]), 256, np.array([128])
            )[0, 0]
        )
        self.assertEqual(parsed, tensor[3, 0, 32].data_ptr())

    def test_state_pages_and_swa_pages(self):
        # Compressor state page: float32 [B, H=1, 4, 512] over 4-state blocks.
        state = as_strided_page(
            (16, 1, 4, 512), (250560, 2048, 512, 1), dtype=torch.float32
        )
        layout = KVCacheGroupLayout(
            {"model.layers.2.attn.compressor.state_cache": state},
            expected_block_size=4,
        )
        self.assertEqual(layout.tensor_block_sizes.tolist(), [4])
        self.assertEqual(layout.tensor_token_strides.tolist(), [2048])
        self.assertEqual(layout.tensor_sizes_per_token.tolist(), [2048])
        self.assertEqual(layout.block_strides.tolist(), [1002240])

        # SWA page: H=1 with one state per token (64-token scheduler block).
        swa = as_strided_page((16, 1, 64, 584), (1002240, 37376, 584, 1))
        layout = KVCacheGroupLayout(
            {"model.layers.0.attn.swa_cache": swa}, expected_block_size=64
        )
        self.assertEqual(layout.tensor_block_sizes.tolist(), [64])
        self.assertEqual(layout.segment_tensor_size_list(64, 64), [64 * 584])

    def test_mixed_compression_views_share_one_group(self):
        dense = as_strided_page((16, 1, 64, 584), (1002240, 37376, 584, 1))
        sparse = as_strided_page((16, 1, 2, 584), (1002240, 1168, 584, 1))
        layout = KVCacheGroupLayout(
            {
                "model.layers.2.attn": dense,
                "model.layers.3.attn": sparse,
            },
            expected_block_size=256,
        )
        self.assertEqual(layout.tensor_block_sizes.tolist(), [64, 2])
        sizes = layout.segment_tensor_size_list(256, 256)
        self.assertEqual(sizes, [64 * 584, 2 * 584])


class LegacyPageTests(unittest.TestCase):
    def test_combined_kv_pair_stays_split(self):
        tensor = as_strided_page((16, 2, 64, 584), (74752, 37376, 584, 1))
        layout = KVCacheGroupLayout(
            {"model.layers.0.attn": tensor}, expected_block_size=64
        )
        self.assertEqual(layout.tensor_block_sizes.tolist(), [64, 64])
        k_base, v_base = layout.base_ptrs.tolist()
        self.assertEqual(v_base - k_base, 37376)
        for block_id in (0, 5):
            addresses = layout.extract_addrs(np.array([block_id]))
            self.assertEqual(
                int(addresses[0, 0]), tensor[block_id, 0].data_ptr()
            )
            self.assertEqual(
                int(addresses[0, 1]), tensor[block_id, 1].data_ptr()
            )

    def test_plain_and_tiny_block_pages(self):
        # Legacy [B, block_size, heads, dim] with the token axis at 1.
        tensor = as_strided_page(
            (16, 64, 8, 128), (65536, 1024, 128, 1), dtype=torch.float16
        )
        layout = KVCacheGroupLayout(
            {"model.layers.0.attn": tensor}, expected_block_size=64
        )
        self.assertEqual(layout.tensor_block_sizes.tolist(), [64])
        self.assertEqual(layout.tensor_token_strides.tolist(), [2048])
        self.assertEqual(layout.tensor_sizes_per_token.tolist(), [2048])

        # block_size=2 must not be mistaken for the K/V axis.
        tiny = as_strided_page(
            (16, 2, 8, 128), (2048, 1024, 128, 1), dtype=torch.float16
        )
        layout = KVCacheGroupLayout(
            {"model.layers.0.attn": tiny}, expected_block_size=2
        )
        self.assertEqual(layout.tensor_block_sizes.tolist(), [2])
        self.assertEqual(layout.tensor_sizes_per_token.tolist(), [2048])

    def test_ascend_layout_check_stays_strict(self):
        tensor = as_strided_page(
            (16, 64, 8, 128), (65536, 1024, 128, 1), dtype=torch.float16
        )
        layout = KVCacheGroupLayout(
            {"model.layers.0.attn": tensor},
            is_ascend_layout=True,
            expected_block_size=64,
        )
        self.assertEqual(layout.tensor_block_sizes.tolist(), [64])

        with self.assertRaisesRegex(ValueError, "block size mismatch"):
            KVCacheGroupLayout(
                {"model.layers.0.attn": tensor},
                is_ascend_layout=True,
                expected_block_size=256,
            )

    def test_non_divisor_state_count_is_rejected(self):
        # 200 states do not divide a 256-token scheduler block.
        tensor = as_strided_page((16, 1, 200, 584), (116800, 116800, 584, 1))
        with self.assertRaisesRegex(ValueError, "block size mismatch"):
            KVCacheGroupLayout(
                {"model.layers.2.attn": tensor}, expected_block_size=256
            )


if __name__ == "__main__":
    unittest.main()
