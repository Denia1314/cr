from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image

from crbot.battle_perception import LaneThreat, detect_lane_threats
from crbot.policy import BattlePolicy, LaneFormation
from crbot.temporal import HandHistory, TimingStats
from crbot.battle_perception import HandCardMatch


class M2TemporalTests(unittest.TestCase):
    def test_hand_history_holds_one_occlusion_then_expires(self) -> None:
        history = HandHistory(max_age_s=1.0, min_confidence=0.4)
        card = HandCardMatch(0, "knight", 0.9, 80, 20, 400, False)
        unknown = HandCardMatch(0, None, 0.1, 0, 0, 120, False)
        history.update([card], 0.0)
        self.assertEqual(history.matches_for_decision([card], 0.1)[0].card_id, "knight")
        history.update([unknown], 0.2)
        self.assertEqual(history.matches_for_decision([unknown], 0.2)[0].card_id, "knight")
        history.update([unknown], 1.3)
        self.assertIsNone(history.matches_for_decision([unknown], 1.3)[0].card_id)

    def test_confirmed_action_blocks_stale_slot_until_replacement(self) -> None:
        history = HandHistory(max_age_s=2.0, min_confidence=0.4)
        old = HandCardMatch(1, "archers", 0.95, 80, 20, 400, False)
        replacement = HandCardMatch(1, "arrows", 0.92, 80, 20, 400, False)
        history.update([old], 0.0)
        history.consume(1, "archers", 0.1)
        self.assertIsNone(history.matches_for_decision([old], 0.2)[0].card_id)
        history.update([replacement], 0.3)
        self.assertEqual(history.matches_for_decision([replacement], 0.3)[0].card_id, "arrows")

    def test_lane_speed_is_stable_when_frame_interval_changes(self) -> None:
        image = Image.new("RGB", (10, 10), "black")
        with patch(
            "crbot.battle_perception._level_badge_candidates",
            side_effect=[[(0.72, 0.37, 20)], [(0.72, 0.34, 20)],
                          [(0.72, 0.355, 20)], [(0.72, 0.34, 20)]],
        ):
            one_second = detect_lane_threats(image, image, frame_dt_s=1.0)
            half_second = detect_lane_threats(image, image, frame_dt_s=0.5)
        self.assertAlmostEqual(
            one_second["right"].approach_rate,
            half_second["right"].approach_rate,
            places=6,
        )

    def test_formation_confidence_decays_before_lifetime_expiry(self) -> None:
        formation = LaneFormation(
            frontline_until=20.0,
            frontline_at=0.0,
            frontline_point=(0.3, 0.7),
            frontline_confidence=0.8,
        )
        self.assertTrue(formation.has_frontline(2.0, min_confidence=0.35, decay_s=10.0))
        self.assertFalse(formation.has_frontline(8.5, min_confidence=0.35, decay_s=10.0))

    def test_double_elixir_phase_only_changes_timer_rate(self) -> None:
        config = {
            "timing": {"battle_action_cooldown_s": [1.0, 1.0]},
            "vision": {"elixir_roi": [0.0, 0.9, 1.0, 1.0], "card_slot_centers": []},
            "policy": {
                "initial_elixir": 0.0,
                "seconds_per_elixir": 2.0,
                "double_elixir_after_s": 10.0,
                "double_elixir_multiplier": 2.0,
            },
        }
        policy = BattlePolicy(config)
        policy.reset_battle(now=0.0)
        policy._update_virtual_elixir(10.0)
        self.assertEqual(policy._elixir_phase, "double")
        policy.virtual_elixir = 0.0
        policy.last_update = 10.0
        before = policy.virtual_elixir
        policy._update_virtual_elixir(11.0)
        self.assertAlmostEqual(policy.virtual_elixir - before, 1.0, places=5)

    def test_timing_stats_reports_p50_and_p95(self) -> None:
        stats = TimingStats()
        for value in (0.1, 0.2, 0.3, 0.4, 0.5):
            stats.record("decision", value)
        summary = stats.summary()["decision"]
        self.assertEqual(summary["count"], 5)
        self.assertAlmostEqual(float(summary["p50_s"]), 0.3, places=6)
        self.assertAlmostEqual(float(summary["p95_s"]), 0.48, places=6)


if __name__ == "__main__":
    unittest.main()
