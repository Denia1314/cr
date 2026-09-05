from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from crbot.battle_perception import HandCardMatch, LaneThreat
from crbot.cards import CardCatalog
from crbot.policy import BattlePolicy
from crbot.tactics import card_tactics


def formation_config() -> dict:
    return {
        "timing": {"battle_action_cooldown_s": [1.0, 1.0]},
        "vision": {
            "elixir_roi": [0.0, 0.9, 1.0, 1.0],
            "friendly_left_roi": [0.0, 0.4, 0.5, 0.8],
            "friendly_right_roi": [0.5, 0.4, 1.0, 0.8],
            "card_slot_centers": [
                [0.2, 0.9],
                [0.4, 0.9],
                [0.6, 0.9],
                [0.8, 0.9],
            ],
        },
        "policy": {
            "mode": "reactive_catalog",
            "fallback_mode": "hold",
            "initial_elixir": 10,
            "seconds_per_elixir": 2.8,
            "unknown_card_cost": 3,
            "enemy_pressure_threshold": 0.2,
            "defense_min_card_score": 1.0,
            "push_min_elixir": 6,
            "support_min_elixir": 4,
            "formation_min_card_score": 1.0,
            "maximum_push_supports": 2,
            "overflow_elixir": 9,
            "left_x": [0.22, 0.4],
            "right_x": [0.6, 0.78],
            "intercept_y": [0.52, 0.67],
            "ranged_defense_y": [0.56, 0.69],
            "building_defense_y": [0.55, 0.63],
            "seed": 9,
        },
    }


class FakeRecognizer:
    available = True
    templates = {"ready": object()}

    def __init__(self, card_ids: list[str]):
        self.card_ids = card_ids

    def recognize(self, _image: Image.Image) -> list[HandCardMatch]:
        return [
            HandCardMatch(index, card_id, 0.95, 80, 20, 400, False)
            for index, card_id in enumerate(self.card_ids)
        ]


class CardTacticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.catalog = CardCatalog.load(root / "data" / "cards.json")

    def profile(self, card_id: str):
        card = self.catalog.get(card_id)
        assert card is not None
        return card_tactics(card)

    def test_catalog_text_derives_formation_roles_without_deck_rules(self) -> None:
        for card_id in ("executioner", "musketeer", "wizard", "flying_machine"):
            self.assertEqual(self.profile(card_id).formation_role, "backline")
        for card_id in ("guards", "knight", "valkyrie", "golem", "mini_pekka"):
            self.assertEqual(self.profile(card_id).formation_role, "frontline")
        rascals = self.profile("rascals")
        self.assertEqual(rascals.formation_role, "hybrid")
        self.assertGreater(rascals.frontline_score, 2.0)
        self.assertGreater(rascals.backline_score, 2.0)

    def test_catalog_text_understands_building_targets_and_splash_quality(self) -> None:
        goblin_giant = self.profile("goblin_giant")
        ice_golem = self.profile("ice_golem")
        baby_dragon = self.profile("baby_dragon")

        self.assertTrue(goblin_giant.is_win_condition)
        self.assertGreater(goblin_giant.offensive_commitment, 0.8)
        self.assertLess(ice_golem.splash_strength, baby_dragon.splash_strength)

    def test_primary_area_damage_beats_death_splash_against_a_swarm(self) -> None:
        policy = BattlePolicy(formation_config())
        policy.catalog = self.catalog
        policy.hand_recognizer = FakeRecognizer(["ice_golem", "baby_dragon"])
        policy.imitation_model = None
        policy.replay_model = None
        policy.reset_battle(now=0.0)
        policy.next_action_at = 0.0
        pressure = {
            "left": LaneThreat("left", 0.8, 4, 0.52, "swarm", ((0.30, 0.52),)),
            "right": LaneThreat("right", 0.0, 0, 0.0, "none", ()),
        }

        with patch("crbot.policy.detect_lane_threats", return_value=pressure):
            decision = policy.decide(
                Image.new("RGB", (600, 1000), "black"), None, now=1.0
            )

        assert decision is not None
        self.assertEqual(decision.card_id, "baby_dragon")

    def test_frontline_is_played_before_executioner_and_executioner_follows(self) -> None:
        policy = BattlePolicy(formation_config())
        policy.catalog = self.catalog
        policy.hand_recognizer = FakeRecognizer(["executioner", "guards"])
        policy.imitation_model = None
        policy.reset_battle(now=0.0)
        policy.next_action_at = 0.0
        empty = {
            "left": LaneThreat("left", 0.0, 0, 0.0, "none", ()),
            "right": LaneThreat("right", 0.0, 0, 0.0, "none", ()),
        }
        image = Image.new("RGB", (600, 1000), "black")

        with patch("crbot.policy.detect_lane_threats", return_value=empty):
            front = policy.decide(image, None, now=1.0)
            rear = policy.decide(image, None, now=3.0)

        assert front is not None and rear is not None
        self.assertEqual(front.card_id, "guards")
        self.assertEqual(front.formation_phase, "form_frontline")
        self.assertEqual(rear.card_id, "executioner")
        self.assertEqual(rear.formation_phase, "support_frontline")
        self.assertEqual(front.lane, rear.lane)
        self.assertGreater(rear.deploy_point[1], front.deploy_point[1])

    def test_defensive_backline_is_then_protected_by_a_frontline(self) -> None:
        policy = BattlePolicy(formation_config())
        policy.catalog = self.catalog
        policy.hand_recognizer = FakeRecognizer(["executioner", "guards"])
        policy.imitation_model = None
        policy.reset_battle(now=0.0)
        policy.next_action_at = 0.0
        pressure = {
            "left": LaneThreat("left", 0.9, 4, 0.55, "swarm", ((0.30, 0.55),)),
            "right": LaneThreat("right", 0.0, 0, 0.0, "none", ()),
        }
        empty = {
            "left": LaneThreat("left", 0.0, 0, 0.0, "none", ()),
            "right": LaneThreat("right", 0.0, 0, 0.0, "none", ()),
        }
        image = Image.new("RGB", (600, 1000), "black")

        with patch("crbot.policy.detect_lane_threats", return_value=pressure):
            defender = policy.decide(image, None, now=1.0)
        with patch("crbot.policy.detect_lane_threats", return_value=empty):
            protector = policy.decide(image, None, now=3.0)

        assert defender is not None and protector is not None
        self.assertEqual(defender.card_id, "executioner")
        self.assertEqual(defender.formation_phase, "direct_defense")
        self.assertEqual(protector.card_id, "guards")
        self.assertEqual(protector.formation_phase, "protect_surviving_backline")
        self.assertEqual(defender.lane, protector.lane)
        self.assertLess(protector.deploy_point[1], defender.deploy_point[1])

    def test_surviving_defender_can_counterpush_before_four_elixir(self) -> None:
        config = formation_config()
        config["policy"].update(
            {"counterpush_min_elixir": 3.0, "counterpush_window_s": 11.0}
        )
        policy = BattlePolicy(config)
        policy.catalog = self.catalog
        policy.hand_recognizer = FakeRecognizer(["guards"])
        policy.imitation_model = None
        policy.replay_model = None
        defender = self.catalog.get("executioner")
        assert defender is not None
        policy.reset_battle(now=0.0)
        policy._remember_formation(
            "left", defender, [0.30, 0.62], now=1.0, source="defense"
        )
        policy.virtual_elixir = 3.2
        policy.last_update = 2.0
        policy.next_action_at = 0.0
        empty = {
            "left": LaneThreat("left", 0.0, 0, 0.0, "none", ()),
            "right": LaneThreat("right", 0.0, 0, 0.0, "none", ()),
        }

        with patch("crbot.policy.detect_lane_threats", return_value=empty):
            decision = policy.decide(
                Image.new("RGB", (600, 1000), "black"), None, now=2.0
            )

        assert decision is not None
        self.assertEqual(decision.card_id, "guards")
        self.assertEqual(decision.formation_phase, "protect_surviving_backline")


if __name__ == "__main__":
    unittest.main()
