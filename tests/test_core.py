from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from crbot.battle_perception import (
    HandCardMatch,
    LaneThreat,
    UniversalHandRecognizer,
    detect_lane_threats,
)
from crbot.cards import CardCatalog, CardDefinition
from crbot.card_sync import merge_community_cards, merge_official_cards
from crbot.dataset import DatasetStore
from crbot.policy import BattlePolicy
from crbot.engine import BotEngine
from crbot.config import load_config
from crbot.recorder import TrainingRecorder
from crbot.vision import (
    ProbeMatch,
    WorkflowRecognizer,
    battle_ui_score,
    estimate_elixir,
    find_chest_open_screen,
    find_result_confirm_button,
    motion_score,
    normalized_box,
    patch_similarity,
)


def base_config() -> dict:
    return {
        "timing": {"battle_action_cooldown_s": [1.0, 1.0]},
        "vision": {
            "min_probe_score": 0.78,
            "motion_threshold": 0.045,
            "elixir_roi": [0.0, 0.9, 1.0, 1.0],
            "friendly_left_roi": [0.0, 0.4, 0.5, 0.8],
            "friendly_right_roi": [0.5, 0.4, 1.0, 0.8],
            "card_slot_centers": [[0.2, 0.9], [0.4, 0.9], [0.6, 0.9], [0.8, 0.9]],
        },
        "policy": {
            "min_elixir_to_act": 4,
            "unknown_card_cost": 3,
            "initial_elixir": 6,
            "seconds_per_elixir": 2.8,
            "left_x": [0.25, 0.35],
            "right_x": [0.65, 0.75],
            "defense_y": [0.7, 0.75],
            "push_y": [0.58, 0.64],
            "seed": 7,
        },
    }


def paste_confirm_text(
    image: Image.Image,
    center: tuple[float, float],
    *,
    scale_multiplier: float = 1.0,
) -> Image.Image:
    template_path = Path(__file__).resolve().parents[1] / "templates" / "result_continue.png"
    with Image.open(template_path) as source:
        scale = image.width / 1080.0 * scale_multiplier
        size = (
            max(12, round(source.width * scale)),
            max(8, round(source.height * scale)),
        )
        template = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    output = image.copy()
    left = round(center[0] * image.width - template.width / 2)
    top = round(center[1] * image.height - template.height / 2)
    output.paste(template, (left, top))
    return output


def make_chest_open_screen(
    *,
    include_chest: bool = True,
    include_star: bool = True,
    coin_count: int = 4,
    purple_theme: bool = False,
    orange_theme: bool = False,
    star_count: int = 1,
) -> Image.Image:
    width, height = 600, 1000
    if orange_theme:
        background = (225, 115, 18)
    elif purple_theme:
        background = (82, 20, 132)
    else:
        background = (15, 72, 150)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    if include_star:
        centers = {
            1: (0.5,),
            2: (0.44, 0.56),
            3: (0.38, 0.50, 0.62),
        }.get(star_count, (0.5,))
        for center_ratio in centers:
            center_x, center_y = center_ratio * width, 0.29 * height
            outer, inner = 0.045 * width, 0.020 * width
            points: list[tuple[float, float]] = []
            for index in range(10):
                radius = outer if index % 2 == 0 else inner
                angle = -math.pi / 2 + index * math.pi / 5
                points.append(
                    (
                        center_x + radius * math.cos(angle),
                        center_y + radius * math.sin(angle),
                    )
                )
            draw.polygon(points, fill=(255, 240, 90))
    if include_chest:
        draw.rounded_rectangle(
            (0.28 * width, 0.44 * height, 0.72 * width, 0.62 * height),
            radius=24,
            fill=(125, 35, 170),
            outline=(235, 160, 35),
            width=22,
        )
        draw.rectangle(
            (0.27 * width, 0.51 * height, 0.73 * width, 0.55 * height),
            fill=(235, 160, 35),
        )
        draw.rounded_rectangle(
            (0.44 * width, 0.48 * height, 0.56 * width, 0.58 * height),
            radius=14,
            fill=(125, 35, 170),
            outline=(235, 160, 35),
            width=10,
        )
    coin_y = 0.90 * height
    for coin_x in (0.29, 0.43, 0.57, 0.71)[:coin_count]:
        radius = 0.047 * width
        center_x = coin_x * width
        draw.ellipse(
            (
                center_x - radius,
                coin_y - radius,
                center_x + radius,
                coin_y + radius,
            ),
            fill=(235, 160, 35),
        )
        question_color = (105, 58, 25)
        stroke = max(3, round(radius * 0.20))
        draw.arc(
            (
                center_x - radius * 0.34,
                coin_y - radius * 0.42,
                center_x + radius * 0.34,
                coin_y + radius * 0.18,
            ),
            start=205,
            end=520,
            fill=question_color,
            width=stroke,
        )
        draw.line(
            (
                center_x,
                coin_y + radius * 0.08,
                center_x,
                coin_y + radius * 0.28,
            ),
            fill=question_color,
            width=stroke,
        )
        draw.ellipse(
            (
                center_x - stroke / 2,
                coin_y + radius * 0.48 - stroke / 2,
                center_x + stroke / 2,
                coin_y + radius * 0.48 + stroke / 2,
            ),
            fill=question_color,
        )
    return image


def make_direct_open_chest_screen(
    *, include_chest: bool = True, include_footer: bool = True
) -> Image.Image:
    width, height = 600, 1000
    top = np.array([145, 45, 125], dtype=np.float32)
    middle = np.array([195, 100, 35], dtype=np.float32)
    bottom = np.array([20, 175, 170], dtype=np.float32)
    data = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        if y < height // 2:
            amount = y / (height / 2)
            color = top * (1.0 - amount) + middle * amount
        else:
            amount = (y - height / 2) / (height / 2)
            color = middle * (1.0 - amount) + bottom * amount
        data[y, :, :] = np.clip(color, 0, 255).astype(np.uint8)
    image = Image.fromarray(data)
    draw = ImageDraw.Draw(image)

    for center_x, center_y in (
        (0.50, 0.255),
        (0.38, 0.29),
        (0.62, 0.29),
        (0.50, 0.325),
    ):
        points: list[tuple[float, float]] = []
        for index in range(10):
            radius = (0.045 if index % 2 == 0 else 0.021) * width
            angle = -math.pi / 2 + index * math.pi / 5
            points.append(
                (
                    center_x * width + radius * math.cos(angle),
                    center_y * height + radius * math.sin(angle),
                )
            )
        draw.polygon(
            points,
            fill=(255, 250, 100),
        )
        draw.line(points + [points[0]], fill=(50, 20, 80), width=5, joint="curve")

    if include_chest:
        draw.rounded_rectangle(
            (0.27 * width, 0.43 * height, 0.73 * width, 0.63 * height),
            radius=26,
            fill=(125, 35, 175),
            outline=(240, 165, 55),
            width=18,
        )
        draw.rounded_rectangle(
            (0.43 * width, 0.48 * height, 0.57 * width, 0.59 * height),
            radius=12,
            fill=(145, 40, 185),
            outline=(245, 175, 65),
            width=8,
        )

    if include_footer:
        for left, component_width in ((0.41, 0.025), (0.46, 0.03), (0.515, 0.02), (0.56, 0.032)):
            draw.rectangle(
                (
                    left * width,
                    0.91 * height,
                    (left + component_width) * width,
                    0.94 * height,
                ),
                fill=(245, 245, 245),
            )
    return image


class VisionTests(unittest.TestCase):
    def test_normalized_box(self) -> None:
        self.assertEqual(normalized_box([0.1, 0.2, 0.9, 0.8], (1000, 500)), (100, 100, 900, 400))

    def test_patch_similarity(self) -> None:
        first = Image.new("RGB", (100, 60), (40, 80, 120))
        same = first.copy()
        different = Image.new("RGB", (100, 60), (220, 30, 10))
        self.assertGreater(patch_similarity(first, same), 0.999)
        self.assertLess(patch_similarity(first, different), 0.7)

    def test_elixir_estimate_synthetic(self) -> None:
        data = np.zeros((100, 200, 3), dtype=np.uint8)
        data[90:100, :120] = [180, 40, 220]
        image = Image.fromarray(data)
        value, confidence = estimate_elixir(image, [0.0, 0.9, 1.0, 1.0])
        self.assertIsNotNone(value)
        self.assertGreater(confidence, 0.1)
        self.assertAlmostEqual(value or 0, 6.0, delta=0.6)

    def test_motion_score(self) -> None:
        previous = Image.new("RGB", (200, 200), "black")
        data = np.zeros((200, 200, 3), dtype=np.uint8)
        data[80:160, :100] = 255
        current = Image.fromarray(data)
        left = motion_score(current, previous, [0.0, 0.4, 0.5, 0.8])
        right = motion_score(current, previous, [0.5, 0.4, 1.0, 0.8])
        self.assertGreater(left, right + 0.5)

    def test_battle_ui_score(self) -> None:
        data = np.zeros((200, 200, 3), dtype=np.uint8)
        data[188:200, 20:180] = [180, 40, 220]
        battle = Image.fromarray(data)
        menu = Image.new("RGB", (200, 200), (20, 70, 110))
        roi = [0.1, 0.94, 0.99, 0.999]
        self.assertGreater(battle_ui_score(battle, roi), 0.72)
        self.assertLess(battle_ui_score(menu, roi), 0.1)

    def test_find_result_confirm_button(self) -> None:
        image = paste_confirm_text(
            Image.new("RGB", (600, 1000), (18, 25, 38)),
            (0.5, 0.89),
        )
        point, confidence = find_result_confirm_button(image)
        self.assertIsNotNone(point)
        assert point is not None
        self.assertAlmostEqual(point[0], 0.5, delta=0.02)
        self.assertAlmostEqual(point[1], 0.89, delta=0.02)
        self.assertGreater(confidence, 0.7)

    def test_find_reward_confirm_button_at_bottom_edge(self) -> None:
        image = paste_confirm_text(
            Image.new("RGB", (843, 1368), (18, 25, 38)),
            (0.5, 0.966),
            scale_multiplier=0.9,
        )

        point, confidence = find_result_confirm_button(image)

        self.assertIsNotNone(point)
        assert point is not None
        self.assertAlmostEqual(point[0], 0.5, delta=0.02)
        self.assertAlmostEqual(point[1], 0.966, delta=0.02)
        self.assertGreater(confidence, 0.7)

    def test_no_result_confirm_button(self) -> None:
        image = Image.new("RGB", (600, 1000), (25, 65, 100))
        point, confidence = find_result_confirm_button(image)
        self.assertIsNone(point)
        self.assertEqual(confidence, 0.0)

    def test_rejects_blue_button_without_confirm_text(self) -> None:
        data = np.zeros((1000, 600, 3), dtype=np.uint8)
        data[860:920, 220:380] = [55, 155, 235]

        point, confidence = find_result_confirm_button(Image.fromarray(data))

        self.assertIsNone(point)
        self.assertLess(confidence, 0.78)

    def test_rejects_off_center_blue_control(self) -> None:
        image = paste_confirm_text(
            Image.new("RGB", (600, 1000), (18, 25, 38)),
            (0.82, 0.84),
        )
        point, confidence = find_result_confirm_button(image)
        self.assertIsNone(point)
        self.assertEqual(confidence, 0.0)

    def test_find_exact_chest_open_screen(self) -> None:
        point, confidence = find_chest_open_screen(make_chest_open_screen())

        self.assertIsNotNone(point)
        assert point is not None
        self.assertAlmostEqual(point[0], 0.5, delta=0.03)
        self.assertAlmostEqual(point[1], 0.53, delta=0.04)
        self.assertGreater(confidence, 0.75)

    def test_accepts_three_question_marks_during_animation(self) -> None:
        point, confidence = find_chest_open_screen(
            make_chest_open_screen(coin_count=3)
        )

        self.assertIsNotNone(point)
        self.assertGreater(confidence, 0.7)

    def test_accepts_purple_three_star_chest_theme(self) -> None:
        image = make_chest_open_screen(purple_theme=True, star_count=3)

        self.assertGreater(battle_ui_score(image, [0.1, 0.94, 0.99, 0.999]), 0.72)
        point, confidence = find_chest_open_screen(image)

        self.assertIsNotNone(point)
        self.assertGreater(confidence, 0.7)

    def test_accepts_orange_two_star_chest_theme(self) -> None:
        image = make_chest_open_screen(
            orange_theme=True,
            star_count=2,
            coin_count=3,
        )

        point, confidence = find_chest_open_screen(image)

        self.assertIsNotNone(point)
        self.assertGreater(confidence, 0.7)

    def test_accepts_four_star_direct_open_chest_screen(self) -> None:
        point, confidence = find_chest_open_screen(make_direct_open_chest_screen())

        self.assertIsNotNone(point)
        assert point is not None
        self.assertAlmostEqual(point[0], 0.5, delta=0.04)
        self.assertAlmostEqual(point[1], 0.53, delta=0.05)
        self.assertGreater(confidence, 0.8)

    def test_rejects_direct_open_layout_without_footer_text(self) -> None:
        point, confidence = find_chest_open_screen(
            make_direct_open_chest_screen(include_footer=False)
        )

        self.assertIsNone(point)
        self.assertEqual(confidence, 0.0)

    def test_rejects_direct_open_layout_without_center_chest(self) -> None:
        point, confidence = find_chest_open_screen(
            make_direct_open_chest_screen(include_chest=False)
        )

        self.assertIsNone(point)
        self.assertEqual(confidence, 0.0)

    def test_rejects_chest_screen_without_question_mark_row(self) -> None:
        point, confidence = find_chest_open_screen(
            make_chest_open_screen(coin_count=2)
        )

        self.assertIsNone(point)
        self.assertEqual(confidence, 0.0)

    def test_rejects_four_coins_without_center_chest(self) -> None:
        point, confidence = find_chest_open_screen(
            make_chest_open_screen(include_chest=False)
        )

        self.assertIsNone(point)
        self.assertEqual(confidence, 0.0)

    def test_rejects_chest_screen_without_yellow_star(self) -> None:
        point, confidence = find_chest_open_screen(
            make_chest_open_screen(include_star=False)
        )

        self.assertIsNone(point)
        self.assertEqual(confidence, 0.0)


class PolicyTests(unittest.TestCase):
    def test_policy_defends_busier_lane(self) -> None:
        config = base_config()
        policy = BattlePolicy(config)
        previous = Image.new("RGB", (200, 200), "black")
        data = np.zeros((200, 200, 3), dtype=np.uint8)
        data[80:160, :100] = 255
        current = Image.fromarray(data)
        policy.reset_battle(now=0.0)
        policy.next_action_at = 0.0

        decision = policy.decide(current, previous, now=1.0)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.lane, "left")
        self.assertEqual(decision.reason, "defend_left")

    @staticmethod
    def _card(
        card_id: str,
        *,
        elixir: int,
        kind: str,
        roles: tuple[str, ...],
    ) -> CardDefinition:
        return CardDefinition(
            card_id=card_id,
            official_id=None,
            name_zh="",
            name_en=card_id,
            elixir=elixir,
            rarity="",
            max_level=None,
            icon_url="",
            icon_path="",
            icon_variants=(),
            kind=kind,
            targets=(),
            roles=roles,
            counters=(),
            synergies=(),
        )

    def test_reactive_policy_uses_splash_against_swarm(self) -> None:
        config = base_config()
        config["policy"].update(
            {
                "mode": "reactive_catalog",
                "fallback_mode": "hold",
                "initial_elixir": 10,
                "enemy_pressure_threshold": 0.2,
            }
        )
        policy = BattlePolicy(config)
        cards = [
            self._card("arrows", elixir=3, kind="spell", roles=("spell", "splash")),
            self._card("guards", elixir=3, kind="troop", roles=("swarm",)),
        ]
        policy.catalog = CardCatalog(cards)

        class FakeRecognizer:
            available = True
            templates = {"ready": object()}

            @staticmethod
            def recognize(_image: Image.Image) -> list[HandCardMatch]:
                return [
                    HandCardMatch(0, "guards", 0.9, 80, 20, 400, False),
                    HandCardMatch(1, "arrows", 0.9, 80, 20, 400, False),
                ]

        policy.hand_recognizer = FakeRecognizer()  # type: ignore[assignment]
        policy.imitation_model = None
        policy.reset_battle(now=0.0)
        policy.next_action_at = 0.0
        left = LaneThreat("left", 0.9, 4, 0.62, "swarm", ((0.3, 0.62),))
        right = LaneThreat("right", 0.0, 0, 0.0, "none", ())
        with patch("crbot.policy.detect_lane_threats", return_value={"left": left, "right": right}):
            decision = policy.decide(Image.new("RGB", (600, 1000), "black"), None, now=1.0)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.card_id, "arrows")
        self.assertEqual(decision.lane, "left")
        self.assertEqual(decision.reason, "counter_swarm_left")

    def test_reactive_policy_holds_when_hand_is_unknown(self) -> None:
        config = base_config()
        config["policy"].update(
            {"mode": "reactive_catalog", "fallback_mode": "hold", "initial_elixir": 10}
        )
        policy = BattlePolicy(config)
        policy.catalog = CardCatalog(
            [self._card("guards", elixir=3, kind="troop", roles=("swarm",))]
        )

        class UnknownRecognizer:
            available = True
            templates = {"ready": object()}

            @staticmethod
            def recognize(_image: Image.Image) -> list[HandCardMatch]:
                return [HandCardMatch(0, None, 0.2, 20, 18, 400, False)]

        policy.hand_recognizer = UnknownRecognizer()  # type: ignore[assignment]
        policy.reset_battle(now=0.0)
        policy.next_action_at = 0.0
        decision = policy.decide(Image.new("RGB", (600, 1000), "black"), None, now=1.0)

        self.assertIsNone(decision)

    def test_reactive_policy_can_open_with_an_ordinary_troop(self) -> None:
        config = base_config()
        config["policy"].update(
            {
                "mode": "reactive_catalog",
                "fallback_mode": "hold",
                "initial_elixir": 10,
                "push_min_elixir": 6,
                "push_min_card_score": 2.5,
            }
        )
        policy = BattlePolicy(config)
        policy.catalog = CardCatalog(
            [self._card("zappies", elixir=4, kind="troop", roles=("troop",))]
        )

        class FakeRecognizer:
            available = True
            templates = {"ready": object()}

            @staticmethod
            def recognize(_image: Image.Image) -> list[HandCardMatch]:
                return [HandCardMatch(0, "zappies", 0.9, 80, 20, 400, False)]

        policy.hand_recognizer = FakeRecognizer()  # type: ignore[assignment]
        policy.imitation_model = None
        policy.reset_battle(now=0.0)
        policy.next_action_at = 0.0
        empty_left = LaneThreat("left", 0.0, 0, 0.0, "none", ())
        empty_right = LaneThreat("right", 0.0, 0, 0.0, "none", ())
        with patch(
            "crbot.policy.detect_lane_threats",
            return_value={"left": empty_left, "right": empty_right},
        ):
            decision = policy.decide(
                Image.new("RGB", (600, 1000), "black"), None, now=1.0
            )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.card_id, "zappies")
        self.assertTrue(decision.reason.startswith("form_frontline_"))
        self.assertLess(decision.deploy_point[1], 0.60)

    def test_early_threat_uses_forward_interception_point(self) -> None:
        config = base_config()
        config["policy"].update(
            {
                "mode": "reactive_catalog",
                "fallback_mode": "hold",
                "initial_elixir": 10,
                "enemy_pressure_threshold": 0.2,
            }
        )
        policy = BattlePolicy(config)
        policy.catalog = CardCatalog(
            [
                self._card(
                    "lumberjack",
                    elixir=4,
                    kind="troop",
                    roles=("tank_killer", "troop"),
                )
            ]
        )

        class FakeRecognizer:
            available = True
            templates = {"ready": object()}

            @staticmethod
            def recognize(_image: Image.Image) -> list[HandCardMatch]:
                return [HandCardMatch(0, "lumberjack", 0.9, 80, 20, 400, False)]

        policy.hand_recognizer = FakeRecognizer()  # type: ignore[assignment]
        policy.imitation_model = None
        policy.reset_battle(now=0.0)
        policy.next_action_at = 0.0
        left = LaneThreat("left", 0.0, 0, 0.0, "none", ())
        early = LaneThreat(
            "right",
            0.4,
            1,
            0.34,
            "single",
            ((0.72, 0.34),),
            approach_rate=0.03,
        )
        with patch(
            "crbot.policy.detect_lane_threats",
            return_value={"left": left, "right": early},
        ):
            decision = policy.decide(
                Image.new("RGB", (600, 1000), "black"), None, now=1.0
            )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.lane, "right")
        self.assertLess(decision.deploy_point[1], 0.61)
        self.assertAlmostEqual(decision.deploy_point[0], 0.72, delta=0.04)

    def test_same_threat_does_not_trigger_repeated_overcommitment(self) -> None:
        policy = BattlePolicy(base_config())
        threat = LaneThreat(
            "right", 0.5, 1, 0.42, "single", ((0.72, 0.42),)
        )
        policy.last_defense_snapshots["right"] = (1.0, 0.5, 1, 0.42)

        self.assertFalse(policy._defense_response_due(threat, 2.5))
        escalated = LaneThreat(
            "right", 0.72, 2, 0.52, "single", ((0.72, 0.52),)
        )
        self.assertTrue(policy._defense_response_due(escalated, 2.5))
        self.assertTrue(policy._defense_response_due(threat, 6.0))

    def test_validated_imitation_model_can_prefer_demonstrated_counter(self) -> None:
        config = base_config()
        config["policy"].update(
            {
                "mode": "reactive_catalog",
                "fallback_mode": "hold",
                "initial_elixir": 10,
                "enemy_pressure_threshold": 0.2,
            }
        )
        config["demonstration"] = {
            "policy_card_weight": 4.0,
            "policy_deploy_weight": 0.5,
        }
        policy = BattlePolicy(config)
        cards = [
            self._card("arrows", elixir=3, kind="spell", roles=("spell", "splash")),
            self._card("guards", elixir=3, kind="troop", roles=("swarm",)),
        ]
        policy.catalog = CardCatalog(cards)

        class FakeRecognizer:
            available = True
            templates = {"ready": object()}

            @staticmethod
            def recognize(_image: Image.Image) -> list[HandCardMatch]:
                return [
                    HandCardMatch(0, "guards", 0.9, 80, 20, 400, False),
                    HandCardMatch(1, "arrows", 0.9, 80, 20, 400, False),
                ]

        class FakeImitation:
            available = True
            champion = {"version": "test"}

            @staticmethod
            def card_score(card: CardDefinition, _elixir: float, _threats: dict) -> float:
                return 1.0 if card.card_id == "arrows" else 0.0

            @staticmethod
            def deploy_point(
                _card: CardDefinition, _elixir: float, _threats: dict
            ) -> list[float]:
                return [0.3, 0.62]

        policy.hand_recognizer = FakeRecognizer()  # type: ignore[assignment]
        policy.imitation_model = FakeImitation()  # type: ignore[assignment]
        policy.reset_battle(now=0.0)
        policy.next_action_at = 0.0
        left = LaneThreat("left", 0.7, 1, 0.55, "single", ((0.3, 0.55),))
        right = LaneThreat("right", 0.0, 0, 0.0, "none", ())
        with patch("crbot.policy.detect_lane_threats", return_value={"left": left, "right": right}):
            decision = policy.decide(Image.new("RGB", (600, 1000), "black"), None, now=1.0)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.card_id, "arrows")
        self.assertTrue(decision.imitation_used)


class BattlePerceptionTests(unittest.TestCase):
    def test_universal_hand_recognizer_uses_catalog_not_deck(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        catalog = CardCatalog.load(root / "data" / "cards.json")
        recognizer = UniversalHandRecognizer(catalog, config["vision"])
        expected = ["clone", "golem", "hog_rider", "guards"]
        screen = Image.new("RGB", (1080, 1920), (20, 40, 70))
        top = round(float(config["vision"]["card_roi_top"]) * screen.height)
        bottom = round(float(config["vision"]["card_roi_bottom"]) * screen.height)
        half_width = float(config["vision"]["card_roi_half_width"])
        for center, card_id in zip(config["vision"]["card_slot_centers"], expected):
            left = round((float(center[0]) - half_width) * screen.width)
            right = round((float(center[0]) + half_width) * screen.width)
            card = catalog.get(card_id)
            assert card is not None
            icon_path = catalog.local_icon_path(card)
            assert icon_path is not None
            icon = Image.open(icon_path).convert("RGBA").resize((right - left, bottom - top))
            screen.paste(icon, (left, top), icon)

        matches = recognizer.recognize(screen)

        self.assertEqual([match.card_id for match in matches], expected)

    def test_lane_threat_ignores_hero_button_region(self) -> None:
        image = Image.new("RGB", (600, 1000), "black")
        data = np.asarray(image).copy()
        data[520:551, 150:191] = [210, 20, 45]
        data[527:543, 160:169] = 255
        data[527:543, 175:184] = 255
        data[710:741, 500:541] = [210, 20, 45]
        data[717:733, 510:519] = 255
        data[717:733, 525:534] = 255

        threats = detect_lane_threats(Image.fromarray(data))

        self.assertEqual(threats["left"].unit_count, 1)
        self.assertGreater(threats["left"].score, 0.2)
        self.assertEqual(threats["right"].unit_count, 0)

    def test_lane_threat_is_seen_before_crossing_bridge(self) -> None:
        previous = np.zeros((1000, 600, 3), dtype=np.uint8)
        current = previous.copy()
        previous[270:301, 320:361] = [210, 20, 45]
        previous[277:293, 330:339] = 255
        previous[277:293, 345:354] = 255
        current[300:331, 320:361] = [210, 20, 45]
        current[307:323, 330:339] = 255
        current[307:323, 345:354] = 255

        threats = detect_lane_threats(
            Image.fromarray(current), Image.fromarray(previous)
        )

        self.assertEqual(threats["right"].unit_count, 1)
        self.assertLess(threats["right"].proximity, 0.4)
        self.assertGreaterEqual(threats["right"].score, 0.2)
        self.assertGreater(threats["right"].approach_rate, 0.02)

    def test_enemy_tower_skin_is_not_treated_as_a_troop(self) -> None:
        data = np.zeros((1000, 600, 3), dtype=np.uint8)
        data[195:226, 150:211] = [210, 20, 45]
        data[202:218, 160:169] = 255
        data[202:218, 180:189] = 255

        threats = detect_lane_threats(Image.fromarray(data))

        self.assertEqual(threats["left"].unit_count, 0)

    def test_own_deploy_cost_indicator_can_be_suppressed(self) -> None:
        data = np.zeros((1000, 600, 3), dtype=np.uint8)
        data[630:661, 220:251] = [210, 20, 45]
        data[637:653, 230:239] = 255

        threats = detect_lane_threats(
            Image.fromarray(data),
            ignore_points=((0.39, 0.68),),
        )

        self.assertEqual(threats["left"].unit_count, 0)


class DatasetTests(unittest.TestCase):
    def test_continuous_samples_are_unlabeled_and_throttled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = TrainingRecorder(
                root,
                {
                    "enabled": True,
                    "battle_frame_interval_s": 1.0,
                    "jpeg_quality": 80,
                },
            )
            image = Image.new("RGB", (120, 200), (20, 50, 90))

            first = recorder.record_battle_sample(
                image,
                battle_index=1,
                observed_at_monotonic=10.0,
            )
            skipped = recorder.record_battle_sample(
                image,
                battle_index=1,
                observed_at_monotonic=10.5,
            )
            second = recorder.record_battle_sample(
                image,
                battle_index=1,
                observed_at_monotonic=11.0,
            )

            self.assertIsNotNone(first)
            self.assertIsNone(skipped)
            self.assertIsNotNone(second)
            store = DatasetStore(recorder.run_dir)
            samples = store.samples()
            self.assertEqual(len(samples), 2)
            self.assertTrue(all(sample.label_status == "unlabeled" for sample in samples))
            self.assertTrue(
                all(not sample.bot_action_is_ground_truth for sample in samples)
            )

    def test_recorded_images_are_resized_and_duplicate_action_frames_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = TrainingRecorder(
                root,
                {
                    "battle_frame_interval_s": 0.1,
                    "max_long_edge": 640,
                    "event_frame_policy": "key_events",
                },
            )
            image = Image.new("RGB", (1080, 1920), (20, 50, 90))

            action = recorder.record("battle_action", image)
            started = recorder.record("battle_started_auto", image)
            sample = recorder.record_battle_sample(
                image,
                battle_index=1,
                observed_at_monotonic=1.0,
            )

            self.assertIsNone(action["frame"])
            self.assertEqual(action["frame_capture"], "policy_skipped")
            self.assertIsNotNone(started["frame"])
            self.assertIsNotNone(sample)
            assert sample is not None
            with Image.open(recorder.run_dir / sample.frame) as saved:
                self.assertEqual(max(saved.size), 640)
                self.assertEqual(tuple(sample.screen_size), saved.size)

    def test_image_quota_stops_images_but_keeps_event_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = TrainingRecorder(
                root,
                {
                    "max_run_image_mb": 0.00001,
                    "event_frame_policy": "all",
                },
            )
            image = Image.new("RGB", (120, 200), "black")

            event = recorder.record("battle_started_auto", image, {"battle": 1})

            self.assertIsNone(event["frame"])
            self.assertEqual(event["frame_capture"], "quota_exceeded")
            self.assertEqual(event["battle"], 1)
            self.assertTrue(recorder.events_path.exists())
            self.assertIsNone(
                recorder.record_battle_sample(
                    image,
                    battle_index=1,
                    observed_at_monotonic=1.0,
                )
            )

    def test_human_annotation_revisions_are_kept_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = TrainingRecorder(root, {"battle_frame_interval_s": 0.1})
            sample = recorder.record_battle_sample(
                Image.new("RGB", (120, 200), "black"),
                battle_index=2,
                observed_at_monotonic=1.0,
            )
            assert sample is not None
            store = DatasetStore(recorder.run_dir)
            annotation = {
                "sample_id": sample.sample_id,
                "source": "human",
                "usable": True,
                "battle_phase": "defense",
                "hand_cards": ["card_a", "card_b", None, None],
                "objects": [
                    {
                        "card_id": "enemy_unit",
                        "team": "enemy",
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
                "notes": "人工确认",
            }

            first = store.save_annotation(annotation)
            annotation["notes"] = "第二次确认"
            second = store.save_annotation(annotation)

            self.assertEqual(first["revision"], 1)
            self.assertEqual(second["revision"], 2)
            self.assertEqual(
                store.latest_annotations()[sample.sample_id]["notes"],
                "第二次确认",
            )
            with self.assertRaises(ValueError):
                store.save_annotation({**annotation, "source": "bot"})

    def test_card_catalog_is_deck_independent_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cards.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source": "test",
                        "cards": [
                            {
                                "id": "sample_card",
                                "name_zh": "示例卡",
                                "name_en": "Sample Card",
                                "elixir": 3,
                                "kind": "troop",
                                "targets": ["ground"],
                                "roles": ["support"],
                                "counters": [],
                                "synergies": [],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            catalog = CardCatalog.load(path)

            self.assertEqual(catalog.ids(), ["sample_card"])
            self.assertEqual(catalog.search("示例")[0].card_id, "sample_card")

    def test_official_card_sync_preserves_manual_strategy_fields(self) -> None:
        existing = {
            "schema_version": 1,
            "cards": [
                {
                    "id": "sample_card",
                    "official_id": 26000001,
                    "name_en": "Sample Card",
                    "name_zh": "示例卡",
                    "kind": "troop",
                    "targets": ["ground"],
                    "roles": ["support"],
                    "counters": ["swarm"],
                    "synergies": ["tank"],
                }
            ],
        }
        official = {
            "items": [
                {
                    "id": 26000001,
                    "name": "Sample Card",
                    "elixirCost": 4,
                    "rarity": "rare",
                    "maxLevel": 14,
                    "iconUrls": {"medium": "https://example.invalid/card.png"},
                }
            ]
        }

        merged = merge_official_cards(existing, official)
        card = merged["cards"][0]

        self.assertEqual(card["elixir"], 4)
        self.assertEqual(card["roles"], ["support"])
        self.assertEqual(card["counters"], ["swarm"])
        self.assertEqual(card["name_zh"], "示例卡")

    def test_community_sync_derives_roles_but_preserves_verified_strategy(self) -> None:
        community = [
            {
                "id": 26000001,
                "key": "verified-card",
                "name": "Verified Card",
                "elixir": 4,
                "rarity": "Rare",
                "type": "Troop",
                "description": "Deals area damage and only attacks buildings.",
            },
            {
                "id": 26000002,
                "key": "derived-card",
                "name": "Derived Card",
                "elixir": 2,
                "rarity": "Common",
                "type": "Spell",
                "description": "Explodes and deals area damage.",
            },
        ]
        existing = {
            "schema_version": 1,
            "cards": [
                {
                    "id": "verified_card",
                    "official_id": 26000001,
                    "strategy_verified": True,
                    "targets": ["air"],
                    "roles": ["tank_killer"],
                }
            ],
        }

        merged = merge_community_cards(existing, community)
        verified, derived = merged["cards"]

        self.assertEqual(verified["roles"], ["tank_killer"])
        self.assertEqual(verified["targets"], ["air"])
        self.assertIn("spell", derived["roles"])
        self.assertIn("splash", derived["roles"])
        self.assertIn("cheap", derived["roles"])

    def test_community_sync_derives_only_unambiguous_target_capabilities(self) -> None:
        community = [
            {
                "id": 26000101,
                "key": "ground-cannon",
                "name": "Ground Cannon",
                "elixir": 3,
                "rarity": "Common",
                "type": "Building",
                "description": "Cannot target flying troops.",
            },
            {
                "id": 26000102,
                "key": "melee-guard",
                "name": "Melee Guard",
                "elixir": 3,
                "rarity": "Common",
                "type": "Troop",
                "description": "A tough melee fighter.",
            },
            {
                "id": 26000103,
                "key": "ambiguous-shooter",
                "name": "Ambiguous Shooter",
                "elixir": 4,
                "rarity": "Rare",
                "type": "Troop",
                "description": "Shoots a powerful projectile.",
            },
        ]

        cards = merge_community_cards({"schema_version": 1, "cards": []}, community)[
            "cards"
        ]
        by_id = {card["id"]: card for card in cards}

        self.assertEqual(by_id["ground_cannon"]["targets"], ["ground"])
        self.assertEqual(by_id["melee_guard"]["targets"], ["ground"])
        self.assertEqual(by_id["ambiguous_shooter"]["targets"], [])


class WorkflowTests(unittest.TestCase):
    def test_workflow_recognition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = Image.new("RGB", (50, 40), (20, 160, 80))
            (root / "templates").mkdir()
            template.save(root / "templates" / "gate.png")
            config = {
                "vision": {"min_probe_score": 0.9},
                "workflow": [
                    {
                        "name": "gate",
                        "kind": "offline_gate",
                        "priority": 10,
                        "template": "templates/gate.png",
                        "roi": [0.25, 0.3, 0.75, 0.7],
                        "click": None,
                    }
                ],
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            screen = Image.new("RGB", (100, 100), "black")
            screen.paste(template.resize((50, 40)), (25, 30))
            matches = WorkflowRecognizer(config, config_path).match_all(screen)
            self.assertEqual([match.name for match in matches], ["gate"])


class EngineControlTests(unittest.TestCase):
    def test_global_confirm_is_clicked_outside_post_battle_state(self) -> None:
        screen = paste_confirm_text(
            Image.new("RGB", (600, 1000), (18, 25, 38)),
            (0.5, 0.89),
        )
        stop_event = Event()

        class FakeDevice:
            def __init__(self) -> None:
                self.taps: list[list[float]] = []

            def foreground_package(self) -> str:
                return "example.game"

            def screenshot(self) -> Image.Image:
                return screen

            def tap_normalized(
                self, point: list[float], _size: tuple[int, int]
            ) -> tuple[int, int]:
                self.taps.append(list(point))
                stop_event.set()
                return 300, 890

        class EmptyRecognizer:
            @staticmethod
            def match_all(_image: Image.Image) -> list[ProbeMatch]:
                return []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = base_config()
            config.update(
                {
                    "game": {"package": "example.game"},
                    "automation": {
                        "start_battle_point": [0.5, 0.75],
                        "battle_ui_roi": [0.1, 0.94, 0.9, 0.999],
                        "battle_ui_threshold": 1.0,
                        "post_battle_confirm_interval_s": 0,
                        "chest_screen_enabled": False,
                    },
                    "workflow": [],
                }
            )
            device = FakeDevice()
            engine = BotEngine(
                device,  # type: ignore[arg-type]
                config,
                root / "config.json",
                stop_event=stop_event,
            )
            engine.recognizer = EmptyRecognizer()  # type: ignore[assignment]

            engine._run_single_marker("example.game")

            self.assertEqual(len(device.taps), 1)
            self.assertAlmostEqual(device.taps[0][0], 0.5, delta=0.02)
            self.assertAlmostEqual(device.taps[0][1], 0.89, delta=0.02)

    def test_global_chest_screen_uses_five_taps_then_delayed_followup(self) -> None:
        # Purple background deliberately scores as battle UI. Chest recognition
        # must run first and override that color-only false positive.
        screen = make_chest_open_screen(purple_theme=True, star_count=3)
        stop_event = Event()

        class FakeDevice:
            def __init__(self) -> None:
                self.taps: list[list[float]] = []

            def foreground_package(self) -> str:
                return "example.game"

            def screenshot(self) -> Image.Image:
                return screen

            def tap_normalized(
                self, point: list[float], _size: tuple[int, int]
            ) -> tuple[int, int]:
                self.taps.append(list(point))
                if len(self.taps) >= 6:
                    stop_event.set()
                return 300, 530

        class EmptyRecognizer:
            @staticmethod
            def match_all(_image: Image.Image) -> list[ProbeMatch]:
                return []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = base_config()
            config.update(
                {
                    "game": {"package": "example.game"},
                    "automation": {
                        "start_battle_point": [0.5, 0.75],
                        "battle_ui_roi": [0.1, 0.94, 0.9, 0.999],
                        "battle_ui_threshold": 1.0,
                        "chest_screen_enabled": True,
                        "chest_screen_tap_interval_s": 0.5,
                        "chest_screen_taps_per_detection": 5,
                        "chest_screen_tap_burst_delay_s": 0.12,
                        "chest_screen_followup_tap_delay_s": 3.0,
                    },
                    "workflow": [],
                }
            )
            device = FakeDevice()
            engine = BotEngine(
                device,  # type: ignore[arg-type]
                config,
                root / "config.json",
                stop_event=stop_event,
            )
            engine.recognizer = EmptyRecognizer()  # type: ignore[assignment]
            sleeps: list[float] = []

            def record_sleep(seconds: float) -> bool:
                sleeps.append(seconds)
                return False

            engine._sleep = record_sleep  # type: ignore[method-assign]

            engine._run_single_marker("example.game")

            self.assertEqual(len(device.taps), 6)
            self.assertEqual(sleeps[:4], [0.12, 0.12, 0.12, 0.12])
            self.assertEqual(sleeps[4], 3.0)
            self.assertAlmostEqual(device.taps[0][0], 0.5, delta=0.03)
            self.assertAlmostEqual(device.taps[0][1], 0.53, delta=0.04)

    def test_long_matchmaking_keeps_waiting_without_repeated_taps(self) -> None:
        screen = Image.new("RGB", (600, 1000), (18, 25, 38))
        stop_event = Event()

        class FakeDevice:
            def __init__(self) -> None:
                self.taps: list[list[float]] = []
                self.screenshot_count = 0

            def foreground_package(self) -> str:
                return "example.game"

            def screenshot(self) -> Image.Image:
                self.screenshot_count += 1
                if self.screenshot_count >= 3:
                    stop_event.set()
                return screen

            def tap_normalized(
                self, point: list[float], _size: tuple[int, int]
            ) -> tuple[int, int]:
                self.taps.append(list(point))
                return 300, 750

        gate = ProbeMatch(
            name="offline_ai_marker",
            kind="offline_gate",
            priority=100,
            score=0.99,
            click=None,
            requires_offline_gate=False,
            requires_battle_seen=False,
        )

        class GateRecognizer:
            @staticmethod
            def match_all(_image: Image.Image) -> list[ProbeMatch]:
                return [gate]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = base_config()
            config["timing"]["poll_interval_s"] = 0
            config.update(
                {
                    "game": {"package": "example.game"},
                    "safety": {"offline_gate_consecutive_frames": 1},
                    "automation": {
                        "start_battle_point": [0.5, 0.75],
                        "start_retry_s": 0,
                        "matchmaking_status_interval_s": 10,
                        "battle_ui_roi": [0.1, 0.94, 0.9, 0.999],
                        "battle_ui_threshold": 1.0,
                        "chest_screen_enabled": False,
                    },
                    "workflow": [],
                }
            )
            device = FakeDevice()
            engine = BotEngine(
                device,  # type: ignore[arg-type]
                config,
                root / "config.json",
                stop_event=stop_event,
            )
            engine.recognizer = GateRecognizer()  # type: ignore[assignment]
            clock = [-61.0]

            def advance_clock() -> float:
                clock[0] += 61.0
                return clock[0]

            with patch("crbot.engine.time.monotonic", side_effect=advance_clock):
                engine._run_single_marker("example.game")

            self.assertEqual(device.taps, [[0.5, 0.75]])
            events = [
                json.loads(line)
                for line in engine.recorder.events_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            event_names = [event["event"] for event in events]
            self.assertIn("matchmaking_waiting", event_names)
            self.assertNotIn("matchmaking_timeout", event_names)

    def test_pre_requested_stop_does_not_launch_game(self) -> None:
        class DeviceThatMustNotLaunch:
            def launch_package(self, _package: str) -> None:
                raise AssertionError("game launch must not occur after a stop request")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = base_config()
            config.update(
                {
                    "game": {"package": "example.offline.game"},
                    "automation": {"single_marker_mode": True},
                    "workflow": [],
                }
            )
            stop_event = Event()
            stop_event.set()
            engine = BotEngine(
                DeviceThatMustNotLaunch(),  # type: ignore[arg-type]
                config,
                root / "config.json",
                stop_event=stop_event,
            )
            engine.run()


class ConfigSafetyTests(unittest.TestCase):
    def test_friendly_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"game": {"allowed_mode": "offline_ai_and_friendly"}}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_config(path)

    def test_unrestricted_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"game": {"allowed_mode": "all_online_modes"}}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
