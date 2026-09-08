from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crbot.runtime_model import (
    DEFAULT_RUNTIME_MODEL,
    apply_runtime_model,
    load_runtime_model,
    normalize_runtime_model,
    runtime_model_allows,
    save_runtime_model,
)


class RuntimeModelTests(unittest.TestCase):
    def test_model_capabilities_are_isolated(self) -> None:
        self.assertTrue(runtime_model_allows("hybrid", "imitation"))
        self.assertTrue(runtime_model_allows("hybrid", "replay"))
        self.assertTrue(runtime_model_allows("imitation", "imitation"))
        self.assertFalse(runtime_model_allows("imitation", "replay"))
        self.assertFalse(runtime_model_allows("rules", "imitation"))
        self.assertFalse(runtime_model_allows("rules", "replay"))

    def test_local_selection_round_trip_and_invalid_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            self.assertEqual(load_runtime_model(config_path), DEFAULT_RUNTIME_MODEL)
            self.assertEqual(save_runtime_model(config_path, "replay"), "replay")
            self.assertEqual(load_runtime_model(config_path), "replay")
            settings = config_path.parent / ".royal-lab.json"
            settings.write_text(
                json.dumps({"runtime_model": "not-a-model"}), encoding="utf-8"
            )
            self.assertEqual(load_runtime_model(config_path), DEFAULT_RUNTIME_MODEL)

    def test_runtime_overlay_does_not_mutate_tracked_config(self) -> None:
        original = {"policy": {"mode": "reactive_catalog"}}
        effective = apply_runtime_model(original, "rules")
        self.assertEqual(effective["policy"]["runtime_model"], "rules")
        self.assertNotIn("runtime_model", original["policy"])
        self.assertEqual(normalize_runtime_model("unknown"), DEFAULT_RUNTIME_MODEL)


if __name__ == "__main__":
    unittest.main()
