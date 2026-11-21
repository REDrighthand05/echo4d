"""Unit tests for compute policy and heterogeneous resource management."""

import os
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
import sys
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from echo_engine.compute_policy import (
    ComputePolicy,
    ExecutionReceipt,
    ExecutionTier,
    calculate_batch_size,
    probe_hardware,
    select_providers,
)


class TestComputePolicy(unittest.TestCase):
    def test_default_policy(self):
        policy = ComputePolicy()
        self.assertEqual(policy.preference, "auto")
        self.assertTrue(policy.use_dynamic_batch)
        self.assertTrue(policy.retain_ui_reserve)
        self.assertTrue(policy.fallback_on_gpu_oom)

        d = policy.to_dict()
        reconstructed = ComputePolicy.from_dict(d)
        self.assertEqual(reconstructed.preference, policy.preference)
        self.assertEqual(reconstructed.use_dynamic_batch, policy.use_dynamic_batch)

    def test_hardware_probe(self):
        info = probe_hardware()
        self.assertIn("gpu_vendor", info)
        self.assertIn("available_providers", info)
        self.assertIsInstance(info["available_providers"], list)
        self.assertGreaterEqual(len(info["available_providers"]), 1)

    def test_select_providers_cpu(self):
        policy = ComputePolicy(preference="cpu")
        providers = select_providers(policy)
        self.assertEqual(providers, ["CPUExecutionProvider"])

    def test_select_providers_env_override(self):
        old_env = os.environ.get("ECHO_DEPTH_PROVIDER")
        try:
            os.environ["ECHO_DEPTH_PROVIDER"] = "CPUExecutionProvider"
            providers = select_providers()
            self.assertEqual(providers, ["CPUExecutionProvider"])
        finally:
            if old_env is None:
                os.environ.pop("ECHO_DEPTH_PROVIDER", None)
            else:
                os.environ["ECHO_DEPTH_PROVIDER"] = old_env

    def test_calculate_batch_size(self):
        # High VRAM free (5000 MB - 1024 MB reserve = 3976 MB usable)
        # Small model (400 MB / frame) -> 3976 // 400 = 9 -> capped at 8
        bs_small = calculate_batch_size(5000, is_large_model=False)
        self.assertEqual(bs_small, 8)

        # Base model (1200 MB / frame) -> 3976 // 1200 = 3 -> capped at 4
        bs_base = calculate_batch_size(5000, is_large_model=True)
        self.assertEqual(bs_base, 3)

        # Zero VRAM -> fallback to 1
        self.assertEqual(calculate_batch_size(0), 1)

        # User override
        self.assertEqual(calculate_batch_size(5000, user_override=2), 2)

    def test_execution_receipt(self):
        receipt = ExecutionReceipt(
            tier=ExecutionTier.GPU_EXECUTED,
            actual_provider="DmlExecutionProvider",
            model_name="depth_anything_v2_base.onnx",
            batch_size=2,
            frame_count=32,
            warmup_seconds=0.22,
            total_seconds=7.15,
            avg_frame_ms=223.4,
            peak_vram_mb=1850,
        )
        d = receipt.to_dict()
        self.assertEqual(d["tier"], "gpu_executed")
        self.assertEqual(d["actual_provider"], "DmlExecutionProvider")
        self.assertEqual(d["frame_count"], 32)


if __name__ == "__main__":
    unittest.main()
