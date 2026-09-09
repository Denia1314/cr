import os
import unittest
from unittest.mock import patch

import numpy as np

from crbot import gpu
from crbot.imitation import _knn_predict
from crbot.replay_learning import _balanced_knn_predict, _plain_knn_predict, _local_predict, CONTEXT_FEATURE_COUNT


class DeviceTests(unittest.TestCase):
    def test_cpu_does_not_import_torch(self):
        with patch.dict(os.environ, CRBOT_COMPUTE_DEVICE="cpu"), patch.dict("sys.modules", torch=None):
            self.assertEqual(gpu.resolve_device("cpu"), "cpu")
            self.assertIsNone(gpu.predict(None, None, None, 1))

    def test_auto_falls_back_but_explicit_cuda_fails(self):
        gpu.resolve_device.cache_clear()
        try:
            with patch.dict("sys.modules", torch=None):
                self.assertEqual(gpu.resolve_device("auto"), "cpu")
                with self.assertRaises(RuntimeError):
                    gpu.resolve_device("cuda")
        finally:
            gpu.resolve_device.cache_clear()

    def test_visual_device_mapping(self):
        with patch.object(gpu, "resolve_device", return_value="cuda:0") as resolve:
            self.assertEqual(gpu.vision_device(0), "cuda:0")
            resolve.assert_called_once_with("cuda:0")


class CudaParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            gpu.resolve_device("cuda:0")
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc))

    def test_predictions_match_cpu(self):
        rng = np.random.default_rng(21)
        x = rng.random((71, 32), dtype=np.float32)
        y = rng.random(71, dtype=np.float32)
        weights = rng.random(71, dtype=np.float32) + 0.1
        query = rng.random(32, dtype=np.float32)
        for k in (1, 7, 100):
            for f, args in (
                (_plain_knn_predict, (x, y, query, k)),
                (_knn_predict, (x, y, query, k)),
                (_balanced_knn_predict, (x, y, query, k, 1.7, 0.6, weights)),
                (_local_predict, ((x, y, weights), query, k)),
            ):
                with self.subTest(function=f.__name__, k=k):
                    with patch.dict(os.environ, CRBOT_COMPUTE_DEVICE="cpu"):
                        expected = f(*args)
                    with patch.dict(os.environ, CRBOT_COMPUTE_DEVICE="cuda:0"):
                        actual = f(*args)
                    self.assertAlmostEqual(actual, expected, places=6)

    def test_ties_empty_phase_and_zero_weights(self):
        x = np.zeros((9, 32), dtype=np.float32)
        y = np.arange(9, dtype=np.float32) / 9
        query = x[0].copy()
        with patch.dict(os.environ, CRBOT_COMPUTE_DEVICE="cpu"):
            expected = _plain_knn_predict(x, y, query, 3)
        with patch.dict(os.environ, CRBOT_COMPUTE_DEVICE="cuda:0"):
            self.assertAlmostEqual(_plain_knn_predict(x, y, query, 3), expected, places=6)
            self.assertEqual(_balanced_knn_predict(x, y, query, 3, 1, 1, np.zeros(9)), 0.5)
            query[-CONTEXT_FEATURE_COUNT] = 1
            self.assertEqual(_local_predict((x, y, np.ones(9)), query, 3), 0.0)

    def test_yolo_forward_backward_on_cuda(self):
        try:
            from ultralytics import YOLO
            from ultralytics.cfg import get_cfg
        except ImportError:
            self.skipTest("Visual training dependencies not installed")
        import torch
        device = gpu.vision_device("auto")
        self.assertTrue(device.startswith("cuda:"))
        detector = YOLO("yolo11n.yaml")
        model = detector.model.to(device).train()
        model.args = get_cfg()
        # Synthetic data checks the actual detection loss without downloading weights.
        batch = {
            "img": torch.rand(2, 3, 64, 64, device=device),
            "batch_idx": torch.tensor([0, 1], device=device),
            "cls": torch.tensor([[0.0], [1.0]], device=device),
            "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]] * 2, device=device),
        }
        loss = model(batch)[0].sum()
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradient = next(p.grad for p in model.parameters() if p.grad is not None)
        self.assertTrue(gradient.is_cuda)
        self.assertTrue(torch.isfinite(gradient).all().item())
        result = detector.predict(np.zeros((64, 64, 3), dtype=np.uint8),
                                  device=device, imgsz=64, verbose=False)[0]
        self.assertTrue(result.boxes.data.is_cuda)


if __name__ == "__main__":
    unittest.main()
