from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from crbot.model_status import model_status


class ModelStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write_registry(self, kind: str, candidates: list, champion=None) -> Path:
        path = self.root / "models" / kind / "registry.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"candidates": candidates, "champion": champion}), encoding="utf-8")
        return path

    def test_latest_rejected_download_is_visible_without_changing_champion(self) -> None:
        old = {"version": "old", "created_at_unix": 1, "promoted": True}
        new = {"version": "new", "created_at_unix": 2, "quality_passed": False,
               "sync_publication": "download", "rejection_reasons": ["样本不足"]}
        path = self.write_registry("replay_policy", [new, old], old)
        before = path.read_bytes()
        summary, detail = model_status(self.root)
        self.assertIn("最新模型：回放 new（未通过验证）", summary)
        self.assertIn("本次加载：未启动", summary)
        self.assertIn("当前冠军：old", detail)
        self.assertIn("同步接收", detail)
        self.assertIn("样本不足", detail)
        self.assertEqual(path.read_bytes(), before)

    def test_live_loaded_version_does_not_follow_new_disk_champion(self) -> None:
        new = {"version": "new", "created_at_unix": 2, "promoted": True}
        self.write_registry("replay_policy", [new], new)
        policy = SimpleNamespace(replay_model=SimpleNamespace(
            available=True, champion={"version": "old"}, influence_scale=0.0))
        summary, detail = model_status(self.root, running=True, policy=policy)
        self.assertIn("最新模型：回放 new", summary)
        self.assertIn("本次加载：回放 old", summary)
        self.assertIn("当前影响权重 0", detail)
        stopped, _ = model_status(self.root, running=False, policy=policy)
        self.assertIn("本次加载：未启动", stopped)

    def test_incompatible_champion_is_not_reported_loaded(self) -> None:
        champion = {"version": "incompatible", "promoted": True}
        self.write_registry("replay_policy", [champion], champion)
        policy = SimpleNamespace(replay_model=SimpleNamespace(
            available=False, champion=champion, load_error="champion_incompatible"))
        summary, detail = model_status(self.root, running=True, policy=policy)
        self.assertIn("本次加载：未加载模型", summary)
        self.assertIn("champion_incompatible", detail)

    def test_corrupt_registry_does_not_hide_other_models_and_recovers(self) -> None:
        path = self.write_registry("replay_policy", [])
        path.write_text("{", encoding="utf-8")
        self.write_registry("imitation", [None, {"version": "demo", "created_at_unix": "bad"}])
        summary, _ = model_status(self.root)
        self.assertIn("示范 demo", summary)
        self.assertIn("回放目录读取失败", summary)
        self.write_registry("replay_policy", [{"version": "recovered", "created_at_unix": 3}])
        summary, _ = model_status(self.root)
        self.assertIn("回放 recovered", summary)
        self.assertNotIn("读取失败", summary)

    def test_all_types_missing_files_and_champion_only_registry(self) -> None:
        for kind in ("replay_policy", "imitation", "battlefield"):
            self.write_registry(kind, [], {"version": kind, "model_path": "missing.npz"})
        _, detail = model_status(self.root)
        for kind in ("replay_policy", "imitation", "battlefield"):
            self.assertIn(kind, detail)
        self.assertEqual(detail.count("文件缺失"), 3)
