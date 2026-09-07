from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

from PIL import Image

from crbot.battle_perception import HandCardMatch, LaneThreat
from crbot.cards import CardCatalog, CardDefinition
from crbot.policy import BattlePolicy


def _card(
    card_id: str,
    cost: int,
    kind: str,
    targets: tuple[str, ...],
    roles: tuple[str, ...],
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        official_id=None,
        name_zh="",
        name_en=card_id,
        elixir=cost,
        rarity="",
        max_level=None,
        icon_url="",
        icon_path="",
        icon_variants=(),
        kind=kind,
        targets=targets,
        roles=roles,
        counters=(),
        synergies=(),
    )


SCENARIO_CARDS = CardCatalog(
    [
        _card("ground_guard", 3, "troop", ("ground",), ("tank_killer",)),
        _card(
            "anti_air_ranger",
            4,
            "troop",
            ("ground", "air"),
            ("ranged", "support"),
        ),
        _card("ground_splash", 3, "troop", ("ground",), ("splash",)),
        _card(
            "air_splash",
            4,
            "troop",
            ("ground", "air"),
            ("ranged", "support", "splash"),
        ),
        _card(
            "ground_building",
            3,
            "building",
            ("ground",),
            ("building", "tank_killer"),
        ),
        _card(
            "air_building",
            4,
            "building",
            ("ground", "air"),
            ("building", "tank_killer"),
        ),
        _card(
            "building_runner",
            5,
            "troop",
            ("buildings",),
            ("building_target", "tank", "win_condition"),
        ),
        _card(
            "dual_spell",
            3,
            "spell",
            ("ground", "air"),
            ("spell", "splash"),
        ),
        _card("ground_spell", 3, "spell", ("ground",), ("spell", "splash")),
        _card("cheap_ground", 1, "troop", ("ground",), ("cheap", "swarm")),
        _card("unknown_ranger", 4, "troop", (), ("ranged", "support")),
        _card("front_tank", 5, "troop", ("ground",), ("tank",)),
    ]
)


def _threat(
    lane: str,
    score: float,
    count: int,
    proximity: float,
    kind: str,
    layers: tuple[str, ...] = (),
    approach_rate: float = 0.0,
    centers: tuple[tuple[float, float], ...] = (),
) -> LaneThreat:
    if not centers and count:
        centers = ((0.30 if lane == "left" else 0.70, proximity),)
    return LaneThreat(
        lane,
        score,
        count,
        proximity,
        kind,
        centers,
        approach_rate=approach_rate,
        unit_layers=layers,
        layer_confidence=0.9 if layers else 0.0,
    )


QUIET_LEFT = _threat("left", 0.0, 0, 0.0, "none")
QUIET_RIGHT = _threat("right", 0.0, 0, 0.0, "none")


@dataclass(frozen=True)
class TacticalScenario:
    scenario_id: str
    category: str
    hand: tuple[str, ...]
    left: LaneThreat
    right: LaneThreat
    acceptable_cards: frozenset[str]
    forbidden_cards: frozenset[str]
    acceptable_lanes: frozenset[str]
    deploy_y_range: tuple[float, float]
    acceptable_phases: frozenset[str] = frozenset({"direct_defense"})
    formation_cards: tuple[tuple[str, str], ...] = ()
    elixir: float = 10.0


# Fixed, deck-independent contracts.  They intentionally specify sets and
# coordinate regions instead of treating one card or one pixel as ground truth.
SCENARIOS = (
    TacticalScenario("air_single_left", "anti_air", ("ground_guard", "anti_air_ranger"), _threat("left", 0.78, 1, 0.50, "single", ("air",)), QUIET_RIGHT, frozenset({"anti_air_ranger"}), frozenset({"ground_guard"}), frozenset({"left"}), (0.49, 0.72)),
    TacticalScenario("air_single_right", "anti_air", ("ground_guard", "anti_air_ranger"), QUIET_LEFT, _threat("right", 0.78, 1, 0.50, "single", ("air",)), frozenset({"anti_air_ranger"}), frozenset({"ground_guard"}), frozenset({"right"}), (0.49, 0.72)),
    TacticalScenario("air_swarm_left", "anti_air", ("ground_splash", "air_splash"), _threat("left", 0.86, 4, 0.54, "swarm", ("air",)), QUIET_RIGHT, frozenset({"air_splash"}), frozenset({"ground_splash"}), frozenset({"left"}), (0.49, 0.72)),
    TacticalScenario("air_swarm_right", "anti_air", ("ground_splash", "air_splash"), QUIET_LEFT, _threat("right", 0.86, 4, 0.54, "swarm", ("air",)), frozenset({"air_splash"}), frozenset({"ground_splash"}), frozenset({"right"}), (0.49, 0.72)),
    TacticalScenario("air_heavy_left", "anti_air", ("ground_building", "air_building"), _threat("left", 0.90, 1, 0.63, "heavy", ("air",)), QUIET_RIGHT, frozenset({"air_building"}), frozenset({"ground_building"}), frozenset({"left"}), (0.55, 0.63)),
    TacticalScenario("air_heavy_right", "anti_air", ("ground_building", "air_building"), QUIET_LEFT, _threat("right", 0.90, 1, 0.63, "heavy", ("air",)), frozenset({"air_building"}), frozenset({"ground_building"}), frozenset({"right"}), (0.55, 0.63)),
    TacticalScenario("mixed_air_ground", "anti_air", ("ground_splash", "air_splash"), _threat("left", 0.88, 3, 0.57, "swarm", ("ground", "air")), QUIET_RIGHT, frozenset({"air_splash"}), frozenset({"ground_splash"}), frozenset({"left"}), (0.49, 0.72)),
    TacticalScenario("air_fast_spell", "anti_air", ("building_runner", "dual_spell"), _threat("right", 0.88, 2, 0.56, "single", ("air",), 0.12), QUIET_LEFT, frozenset({"dual_spell"}), frozenset({"building_runner"}), frozenset({"right"}), (0.45, 0.65)),
    TacticalScenario("ground_swarm_troop", "swarm", ("ground_guard", "ground_splash", "building_runner"), _threat("left", 0.84, 4, 0.54, "swarm", ("ground",)), QUIET_RIGHT, frozenset({"ground_splash"}), frozenset({"ground_guard", "building_runner"}), frozenset({"left"}), (0.49, 0.72)),
    TacticalScenario("ground_swarm_spell", "swarm", ("ground_splash", "dual_spell", "building_runner"), QUIET_LEFT, _threat("right", 0.88, 5, 0.58, "swarm", ("ground",)), frozenset({"ground_splash", "dual_spell"}), frozenset({"building_runner"}), frozenset({"right"}), (0.22, 0.72)),
    TacticalScenario("mixed_swarm_splash", "swarm", ("ground_splash", "air_splash"), _threat("right", 0.90, 5, 0.60, "swarm", ("ground", "air")), QUIET_LEFT, frozenset({"air_splash"}), frozenset({"ground_splash"}), frozenset({"right"}), (0.49, 0.72)),
    TacticalScenario("unknown_layer_swarm", "swarm", ("ground_guard", "ground_splash"), _threat("left", 0.82, 4, 0.52, "swarm"), QUIET_RIGHT, frozenset({"ground_splash"}), frozenset({"ground_guard"}), frozenset({"left"}), (0.49, 0.72)),
    TacticalScenario("dual_left_air_stronger", "dual_lane", ("ground_guard", "anti_air_ranger", "building_runner"), _threat("left", 0.88, 2, 0.55, "single", ("air",)), _threat("right", 0.62, 1, 0.48, "single", ("ground",)), frozenset({"anti_air_ranger"}), frozenset({"ground_guard", "building_runner"}), frozenset({"left"}), (0.49, 0.72)),
    TacticalScenario("dual_right_ground_stronger", "dual_lane", ("ground_guard", "anti_air_ranger", "building_runner"), _threat("left", 0.60, 1, 0.47, "single", ("air",)), _threat("right", 0.91, 1, 0.64, "heavy", ("ground",)), frozenset({"ground_guard"}), frozenset({"building_runner"}), frozenset({"right"}), (0.49, 0.72)),
    TacticalScenario("dual_equal_tie_left", "dual_lane", ("ground_guard", "building_runner"), _threat("left", 0.72, 1, 0.52, "single", ("ground",)), _threat("right", 0.72, 1, 0.52, "single", ("ground",)), frozenset({"ground_guard"}), frozenset({"building_runner"}), frozenset({"left"}), (0.49, 0.72)),
    TacticalScenario("dual_left_mixed", "dual_lane", ("ground_splash", "air_splash"), _threat("left", 0.86, 3, 0.56, "swarm", ("ground", "air")), _threat("right", 0.74, 1, 0.53, "single", ("ground",)), frozenset({"air_splash"}), frozenset({"ground_splash"}), frozenset({"left"}), (0.49, 0.72)),
    TacticalScenario("dual_right_air", "dual_lane", ("ground_building", "air_building"), _threat("left", 0.66, 1, 0.50, "single", ("ground",)), _threat("right", 0.89, 1, 0.62, "heavy", ("air",)), frozenset({"air_building"}), frozenset({"ground_building"}), frozenset({"right"}), (0.55, 0.63)),
    TacticalScenario("fast_early_melee", "fast_approach", ("ground_guard", "building_runner"), _threat("left", 0.68, 1, 0.35, "single", ("ground",), 0.10), QUIET_RIGHT, frozenset({"ground_guard"}), frozenset({"building_runner"}), frozenset({"left"}), (0.52, 0.59)),
    TacticalScenario("fast_engaged_ranged", "fast_approach", ("anti_air_ranger", "building_runner"), QUIET_LEFT, _threat("right", 0.78, 2, 0.50, "single", ("ground",), 0.15), frozenset({"anti_air_ranger"}), frozenset({"building_runner"}), frozenset({"right"}), (0.56, 0.69)),
    TacticalScenario("fast_close_building", "fast_approach", ("ground_building", "building_runner"), _threat("left", 0.92, 1, 0.66, "heavy", ("ground",), 0.18), QUIET_RIGHT, frozenset({"ground_building"}), frozenset({"building_runner"}), frozenset({"left"}), (0.55, 0.63)),
    TacticalScenario("fast_air_spell", "fast_approach", ("ground_spell", "dual_spell"), QUIET_LEFT, _threat("right", 0.90, 2, 0.48, "single", ("air",), 0.16, ((0.69, 0.44), (0.72, 0.50))), frozenset({"dual_spell"}), frozenset({"ground_spell"}), frozenset({"right"}), (0.42, 0.53)),
    TacticalScenario("counter_backline_left", "counterpush", ("front_tank", "anti_air_ranger"), QUIET_LEFT, QUIET_RIGHT, frozenset({"front_tank"}), frozenset({"anti_air_ranger"}), frozenset({"left"}), (0.49, 0.58), frozenset({"protect_surviving_backline"}), (("left", "anti_air_ranger"),), 6.0),
    TacticalScenario("counter_frontline_left", "counterpush", ("anti_air_ranger", "front_tank"), QUIET_LEFT, QUIET_RIGHT, frozenset({"anti_air_ranger"}), frozenset({"front_tank"}), frozenset({"left"}), (0.62, 0.75), frozenset({"support_counterpush"}), (("left", "front_tank"),), 6.0),
    TacticalScenario("counter_backline_right", "counterpush", ("front_tank", "air_splash"), QUIET_LEFT, QUIET_RIGHT, frozenset({"front_tank"}), frozenset({"air_splash"}), frozenset({"right"}), (0.49, 0.58), frozenset({"protect_surviving_backline"}), (("right", "air_splash"),), 6.0),
    TacticalScenario("counter_complete_right", "counterpush", ("anti_air_ranger", "ground_building"), QUIET_LEFT, QUIET_RIGHT, frozenset({"anti_air_ranger"}), frozenset({"ground_building"}), frozenset({"right"}), (0.62, 0.75), frozenset({"support_counterpush"}), (("right", "front_tank"), ("right", "anti_air_ranger")), 6.0),
    TacticalScenario("survivor_protect_left", "surviving_backline", ("ground_guard", "unknown_ranger"), QUIET_LEFT, QUIET_RIGHT, frozenset({"ground_guard"}), frozenset({"unknown_ranger"}), frozenset({"left"}), (0.49, 0.58), frozenset({"protect_surviving_backline"}), (("left", "unknown_ranger"),), 5.0),
    TacticalScenario("survivor_protect_right", "surviving_backline", ("front_tank", "anti_air_ranger"), QUIET_LEFT, QUIET_RIGHT, frozenset({"front_tank"}), frozenset({"anti_air_ranger"}), frozenset({"right"}), (0.49, 0.58), frozenset({"protect_surviving_backline"}), (("right", "anti_air_ranger"),), 6.0),
    TacticalScenario("survivor_support_left", "surviving_backline", ("air_splash", "front_tank"), QUIET_LEFT, QUIET_RIGHT, frozenset({"air_splash"}), frozenset({"front_tank"}), frozenset({"left"}), (0.62, 0.75), frozenset({"support_counterpush"}), (("left", "front_tank"),), 6.0),
    TacticalScenario("spell_cluster_left", "spell_motion", ("dual_spell", "building_runner"), _threat("left", 0.86, 3, 0.49, "swarm", ("ground",), 0.08, ((0.27, 0.47), (0.30, 0.49), (0.33, 0.51))), QUIET_RIGHT, frozenset({"dual_spell"}), frozenset({"building_runner"}), frozenset({"left"}), (0.45, 0.53)),
    TacticalScenario("spell_cluster_right", "spell_motion", ("ground_spell", "building_runner"), QUIET_LEFT, _threat("right", 0.86, 3, 0.54, "swarm", ("ground",), 0.10, ((0.67, 0.51), (0.70, 0.54), (0.73, 0.56))), frozenset({"ground_spell"}), frozenset({"building_runner"}), frozenset({"right"}), (0.49, 0.58)),
    TacticalScenario("spell_air_moving", "spell_motion", ("ground_spell", "dual_spell"), _threat("left", 0.90, 3, 0.58, "swarm", ("air",), 0.18, ((0.26, 0.54), (0.30, 0.58), (0.34, 0.61))), QUIET_RIGHT, frozenset({"dual_spell"}), frozenset({"ground_spell"}), frozenset({"left"}), (0.53, 0.62)),
    TacticalScenario("lost_layer_single", "observation_loss", ("ground_guard", "unknown_ranger", "building_runner"), _threat("left", 0.72, 1, 0.51, "single"), QUIET_RIGHT, frozenset({"ground_guard", "unknown_ranger"}), frozenset({"building_runner"}), frozenset({"left"}), (0.49, 0.72)),
    TacticalScenario("lost_centers_heavy", "observation_loss", ("ground_guard", "ground_splash"), QUIET_LEFT, _threat("right", 0.88, 1, 0.63, "heavy", ("ground",), centers=()), frozenset({"ground_guard"}), frozenset({"ground_splash"}), frozenset({"right"}), (0.49, 0.72)),
    TacticalScenario("lost_count_ground", "observation_loss", ("ground_guard", "building_runner"), _threat("left", 0.82, 0, 0.57, "single", ("ground",), 0.05), QUIET_RIGHT, frozenset({"ground_guard"}), frozenset({"building_runner"}), frozenset({"left"}), (0.49, 0.72)),
)


def _config(initial_elixir: float) -> dict:
    return {
        "dataset": {"card_catalog": "__m3_scenario_catalog_only__.json"},
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
            "initial_elixir": initial_elixir,
            "seconds_per_elixir": 2.8,
            "unknown_card_cost": 3,
            "enemy_pressure_threshold": 0.2,
            "defense_min_card_score": 1.0,
            "push_min_elixir": 6,
            "support_min_elixir": 4,
            "counterpush_min_elixir": 3,
            "formation_min_card_score": 1.0,
            "maximum_push_supports": 2,
            "overflow_elixir": 9,
            "left_x": [0.22, 0.4],
            "right_x": [0.6, 0.78],
            "intercept_y": [0.52, 0.67],
            "ranged_defense_y": [0.56, 0.69],
            "building_defense_y": [0.55, 0.63],
            "seed": 19,
        },
    }


class ScenarioRecognizer:
    available = True
    templates = {"scenario": object()}

    def __init__(self, hand: tuple[str, ...]) -> None:
        self.hand = hand

    def recognize(self, _image: Image.Image) -> list[HandCardMatch]:
        return [
            HandCardMatch(index, card_id, 0.95, 80, 20, 400, False)
            for index, card_id in enumerate(self.hand)
        ]


class M3FixedScenarioTests(unittest.TestCase):
    def test_fixed_scenario_contracts_are_complete(self) -> None:
        required_categories = {
            "anti_air",
            "dual_lane",
            "fast_approach",
            "swarm",
            "counterpush",
            "surviving_backline",
            "spell_motion",
            "observation_loss",
        }
        self.assertGreaterEqual(len(SCENARIOS), 30)
        self.assertEqual(len({scenario.scenario_id for scenario in SCENARIOS}), len(SCENARIOS))
        self.assertTrue(required_categories.issubset({value.category for value in SCENARIOS}))
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario.scenario_id):
                self.assertTrue(scenario.acceptable_cards)
                self.assertTrue(scenario.forbidden_cards)
                self.assertTrue(scenario.acceptable_lanes)
                self.assertTrue(set(scenario.acceptable_cards).issubset(scenario.hand))
                self.assertTrue(set(scenario.forbidden_cards).issubset(scenario.hand))
                self.assertTrue(scenario.acceptable_cards.isdisjoint(scenario.forbidden_cards))

    def test_policy_satisfies_fixed_scenario_action_sets(self) -> None:
        image = Image.new("RGB", (600, 1000), "black")
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario.scenario_id):
                policy = BattlePolicy(_config(scenario.elixir))
                policy.catalog = SCENARIO_CARDS
                policy.hand_recognizer = ScenarioRecognizer(scenario.hand)
                policy.learned_detector = None
                policy.imitation_model = None
                policy.replay_model = None
                policy.reset_battle(now=0.0)
                policy.next_action_at = 0.0
                for lane, card_id in scenario.formation_cards:
                    card = SCENARIO_CARDS.get(card_id)
                    assert card is not None
                    policy._remember_formation(
                        lane,
                        card,
                        [0.30 if lane == "left" else 0.70, 0.62],
                        now=0.2,
                        source="defense",
                    )

                threats = {"left": scenario.left, "right": scenario.right}
                with (
                    patch("crbot.policy.detect_lane_threats", return_value=threats),
                    patch("crbot.policy.estimate_elixir", return_value=(None, 0.0)),
                ):
                    decision = policy.decide(image, None, now=1.0)

                self.assertIsNotNone(decision)
                assert decision is not None
                self.assertIn(decision.card_id, scenario.acceptable_cards)
                self.assertNotIn(decision.card_id, scenario.forbidden_cards)
                self.assertIn(decision.lane, scenario.acceptable_lanes)
                self.assertIn(decision.formation_phase, scenario.acceptable_phases)
                self.assertGreaterEqual(decision.deploy_point[1], scenario.deploy_y_range[0])
                self.assertLessEqual(decision.deploy_point[1], scenario.deploy_y_range[1])


if __name__ == "__main__":
    unittest.main()
