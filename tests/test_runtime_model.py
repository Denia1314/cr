from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crbot.runtime_model import (
    DEFAULT_RUNTIME_MODEL,
    RUNTIME_MODEL_LABELS,
    apply_runtime_model,
    load_runtime_model,
    normalize_runtime_model,
    save_runtime_model,
)


class RuntimeModelTests(unittest.TestCase):
    def test_all_existing_strategy_versions_are_selectable(self) -> None:
        self.assertEqual(
            set(RUNTIME_MODEL_LABELS),
            {"v5", "m1", "m2", "m2_1", "m3_c1", "m3_c2", "m3_c3"},
        )

    def test_local_selection_round_trip_and_invalid_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            self.assertEqual(load_runtime_model(config_path), DEFAULT_RUNTIME_MODEL)
            self.assertEqual(save_runtime_model(config_path, "m2_1"), "m2_1")
            self.assertEqual(load_runtime_model(config_path), "m2_1")
            settings = config_path.parent / ".royal-lab.json"
            settings.write_text(
                json.dumps({"runtime_model": "not-a-model"}), encoding="utf-8"
            )
            self.assertEqual(load_runtime_model(config_path), DEFAULT_RUNTIME_MODEL)

    def test_runtime_overlay_does_not_mutate_tracked_config(self) -> None:
        original = {"policy": {"mode": "reactive_catalog"}}
        effective = apply_runtime_model(original, "m3_c1")
        self.assertEqual(effective["policy"]["runtime_model"], "m3_c1")
        self.assertEqual(
            effective["policy"]["version"], "adaptive_counterpush_v5_m3_c1"
        )
        self.assertTrue(effective["policy"]["dual_lane_elixir_enabled"])
        self.assertFalse(effective["policy"]["role_aware_placement_enabled"])
        self.assertFalse(effective["policy"]["counterpush_evidence_enabled"])
        self.assertEqual(
            effective["replay"]["training_policy_version"],
            "adaptive_counterpush_v5_m3_c1",
        )
        self.assertNotIn("runtime_model", original["policy"])
        self.assertEqual(normalize_runtime_model("unknown"), DEFAULT_RUNTIME_MODEL)

    def test_profiles_enable_milestones_incrementally(self) -> None:
        source = {"policy": {}, "replay": {}}
        v5 = apply_runtime_model(source, "v5")
        m1 = apply_runtime_model(source, "m1")
        c2 = apply_runtime_model(source, "m3_c2")
        c3 = apply_runtime_model(source, "m3_c3")
        self.assertFalse(v5["replay"]["require_action_confirmation"])
        self.assertFalse(v5["policy"]["temporal_perception_enabled"])
        self.assertTrue(m1["replay"]["require_action_confirmation"])
        self.assertFalse(m1["policy"]["temporal_perception_enabled"])
        self.assertTrue(apply_runtime_model(source, "m2")["policy"]["temporal_perception_enabled"])
        self.assertTrue(c2["policy"]["role_aware_placement_enabled"])
        self.assertFalse(c2["policy"]["counterpush_evidence_enabled"])
        self.assertTrue(c3["policy"]["counterpush_evidence_enabled"])


if __name__ == "__main__":
    unittest.main()
