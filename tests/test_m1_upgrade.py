from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from threading import Event
from unittest.mock import patch

from PIL import Image

from crbot.action_confirmation import (
    ActionConfirmationTracker,
    assess_action_evidence,
)
from crbot.battle_perception import HandCardMatch
from crbot.cards import CardCatalog, CardDefinition
from crbot.engine import BotEngine
from crbot.policy import BattleDecision, BattlePolicy
from crbot.replay import ExperienceReplayRecorder
from crbot.replay_evaluation import (
    champion_frozen_exposure,
    content_fingerprint,
    experiment_protocol_fingerprint,
    reserve_frozen_groups,
)
from crbot.replay_learning import collect_replay_learning_actions
from crbot.temporal import TimingStats
from crbot.vision import BattleResult


def _decision() -> BattleDecision:
    return BattleDecision(
        slot_index=0,
        card_point=[0.2, 0.89],
        deploy_point=[0.3, 0.7],
        lane="left",
        reason="test",
        elixir=7.0,
        elixir_source="vision",
        left_motion=0.0,
        right_motion=0.0,
        card_id="test_card",
        card_cost=3,
        hand=("test_card", None, None, None),
    )


class M1UpgradeTests(unittest.TestCase):
    def test_confirmation_requires_two_stable_independent_signals(self) -> None:
        pre = [HandCardMatch(0, "test_card", 0.9, 80, 20, 400, False)]
        post = [HandCardMatch(0, "replacement", 0.9, 80, 20, 400, False)]
        before = Image.new("RGB", (600, 1000), (30, 40, 50))
        after = Image.new("RGB", (600, 1000), (30, 40, 50))
        evidence = assess_action_evidence(
            pre_image=before,
            post_image=after,
            pre_matches=pre,
            post_matches=post,
            slot_index=0,
            pre_elixir=7.0,
            post_elixir=4.0,
            card_cost=3,
            slot_center=[0.2, 0.89],
            vision_config={
                "card_roi_half_width": 0.09,
                "card_roi_top": 0.825,
                "card_roi_bottom": 0.955,
            },
        )
        self.assertEqual(evidence["independent_signals"], 2)
        tracker = ActionConfirmationTracker(timeout_s=2.0, stable_frames=2)
        tracker.register("a", sent_at=0.0)
        self.assertEqual(tracker.observe("a", evidence, now=0.5).status, "sent")
        confirmed = tracker.observe(
            "a", {**evidence, "frame_fingerprint": "next-frame"}, now=0.7
        )
        self.assertEqual(confirmed.status, "confirmed")
        self.assertEqual(tracker.observe("a", evidence, now=0.8), confirmed)

    def test_single_empty_slot_is_unknown_and_no_evidence_is_rejected(self) -> None:
        pre = [HandCardMatch(0, "test_card", 0.9, 80, 20, 400, False)]
        empty = [HandCardMatch(0, None, 1.0, 0, 0, 20, True)]
        evidence = assess_action_evidence(
            pre_image=None,
            post_image=None,
            pre_matches=pre,
            post_matches=empty,
            slot_index=0,
            pre_elixir=7.0,
            post_elixir=7.0,
            card_cost=3,
        )
        tracker = ActionConfirmationTracker(timeout_s=1.0, stable_frames=2)
        tracker.register("empty", sent_at=0.0)
        result = tracker.observe("empty", evidence, now=1.1)
        self.assertEqual(result.status, "unknown")

        tracker.register("none", sent_at=0.0)
        result = tracker.observe("none", {"independent_signals": 0}, now=1.1)
        self.assertEqual(result.status, "rejected")

    def test_timeout_preserves_partial_evidence_without_adding_a_frame(self) -> None:
        tracker = ActionConfirmationTracker(timeout_s=1.0, stable_frames=2)
        tracker.register("partial", sent_at=0.0)
        partial = {
            "independent_signals": 1,
            "hand_change": True,
            "confidence": 0.35,
        }
        waiting = tracker.observe("partial", partial, now=0.4)
        self.assertEqual(waiting.status, "sent")
        self.assertEqual(waiting.observed_frames, 1)

        result = tracker.timeout("partial", now=1.1)

        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.observed_frames, 1)
        self.assertTrue(result.evidence["hand_change"])
        self.assertTrue(result.evidence["partial_evidence_seen"])
        self.assertEqual(result.evidence["best_independent_signals"], 1)

    def test_timeout_preserves_one_unstable_positive_frame(self) -> None:
        tracker = ActionConfirmationTracker(timeout_s=1.0, stable_frames=2)
        tracker.register("unstable", sent_at=0.0)
        positive = {
            "independent_signals": 2,
            "hand_change": True,
            "elixir_change": True,
            "confidence": 0.78,
        }
        self.assertEqual(
            tracker.observe("unstable", positive, now=0.4).status,
            "sent",
        )

        result = tracker.timeout("unstable", now=1.1)

        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.observed_frames, 1)
        self.assertEqual(result.evidence["independent_signals"], 2)
        self.assertAlmostEqual(result.confidence, 0.78)

    def test_timeout_summarizes_different_partial_signals_across_frames(self) -> None:
        tracker = ActionConfirmationTracker(timeout_s=1.0, stable_frames=2)
        tracker.register("cross-frame", sent_at=0.0)
        tracker.observe(
            "cross-frame",
            {
                "independent_signals": 1,
                "hand_change": True,
                "confidence": 0.35,
            },
            now=0.3,
        )
        tracker.observe(
            "cross-frame",
            {
                "independent_signals": 1,
                "elixir_change": True,
                "confidence": 0.4,
            },
            now=0.6,
        )

        result = tracker.timeout("cross-frame", now=1.1)

        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.observed_frames, 2)
        self.assertEqual(
            set(result.evidence["observed_signals"]),
            {"hand_change", "elixir_change"},
        )
        self.assertEqual(result.evidence["cumulative_independent_signals"], 2)

    def test_timeout_without_observations_is_rejected_without_synthetic_frame(self) -> None:
        tracker = ActionConfirmationTracker(timeout_s=1.0, stable_frames=2)
        tracker.register("none-direct", sent_at=0.0)

        result = tracker.timeout("none-direct", now=1.1)

        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.observed_frames, 0)
        self.assertEqual(result.evidence["evidence_frames"], 0)

    def test_confirmation_summary_classifies_missing_second_frame(self) -> None:
        tracker = ActionConfirmationTracker(timeout_s=1.0, stable_frames=2)
        tracker.register("late", sent_at=0.0)
        tracker.observe(
            "late",
            {"independent_signals": 1, "visual_change": True, "confidence": 0.35},
            now=0.7,
        )

        result = tracker.timeout("late", now=1.1)
        summary = tracker.summary()

        self.assertEqual(result.failure_code, "second_frame_unavailable")
        self.assertEqual(summary["outcome_counts"], {"unknown": 1})
        self.assertEqual(summary["failure_counts"], {"second_frame_unavailable": 1})
        self.assertEqual(summary["elapsed_s"]["count"], 1)

    def test_duplicate_confirmation_frame_does_not_confirm(self) -> None:
        tracker = ActionConfirmationTracker(timeout_s=1.0, stable_frames=2)
        tracker.register("duplicate", sent_at=0.0)
        evidence = {
            "independent_signals": 2,
            "hand_change": True,
            "elixir_change": True,
            "confidence": 0.8,
            "frame_fingerprint": "same-frame",
        }
        self.assertEqual(tracker.observe("duplicate", evidence, now=0.3).status, "sent")
        self.assertEqual(tracker.observe("duplicate", evidence, now=0.6).status, "sent")

        result = tracker.timeout("duplicate", now=1.1)

        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.failure_code, "stale_frame")
        self.assertEqual(result.evidence["duplicate_frames"], 1)

    def test_confirmation_window_starts_after_slow_precheck_and_deploy_tap(self) -> None:
        image = Image.new("RGB", (600, 1000), (30, 40, 50))

        class Clock:
            def __init__(self) -> None:
                self.value = 0.0

            def monotonic(self) -> float:
                return self.value

            def advance(self, seconds: float) -> None:
                self.value += float(seconds)

        clock = Clock()
        old = HandCardMatch(0, "test_card", 0.9, 80, 20, 400, False)
        replacement = HandCardMatch(0, "replacement", 0.9, 80, 20, 400, False)

        class SlowHandRecognizer:
            def __init__(self) -> None:
                self.calls = 0

            def recognize(self, _image: Image.Image) -> list[HandCardMatch]:
                clock.advance(0.4)
                self.calls += 1
                return [old] if self.calls == 1 else [replacement]

        class FakePolicy:
            def __init__(self) -> None:
                self.hand_recognizer = SlowHandRecognizer()
                self.last_hand_image = None
                self.last_hand_matches: list[HandCardMatch] = []
                self.last_elixir_image = None
                self.last_elixir_estimate_value = None
                self.last_elixir_confidence = 0.0
                self.last_elixir_estimate_source = "timer"
                self.resolved_status: str | None = None

            def prepare_action(self, decision, _snapshot):
                return replace(decision, action_id="slow-action")

            def resolve_action(self, _action_id: str, status: str, *, now: float) -> bool:
                self.resolved_status = status
                return True

        class FakeDevice:
            def __init__(self) -> None:
                self.screenshots = 0

            def tap_normalized(self, point, _size):
                clock.advance(0.25)
                return [round(point[0] * 600), round(point[1] * 1000)]

            def screenshot(self) -> Image.Image:
                clock.advance(0.1)
                self.screenshots += 1
                return Image.new(
                    "RGB", image.size, (30 + 10 * self.screenshots, 40, 50)
                )

        class FakeRecorder:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict]] = []

            def record(self, event_type, _image, payload):
                self.events.append((event_type, dict(payload)))
                return {"frame": None}

        class RecordingTracker(ActionConfirmationTracker):
            registered_at: float | None = None

            def register(self, action_id: str, *, sent_at: float | None = None):
                self.registered_at = sent_at
                return super().register(action_id, sent_at=sent_at)

        tracker = RecordingTracker(timeout_s=1.35, poll_interval_s=0.18, stable_frames=2)
        engine = object.__new__(BotEngine)
        engine.policy = FakePolicy()
        engine.action_confirmation = tracker
        engine.recorder = FakeRecorder()
        engine.device = FakeDevice()
        engine.dry_run = False
        engine.stop_event = Event()
        engine.response_timing = TimingStats()
        engine.config = {
            "vision": {
                "elixir_roi": [0.0, 0.9, 1.0, 1.0],
                "card_slot_centers": [[0.2, 0.89]],
                "card_roi_half_width": 0.09,
                "card_roi_top": 0.825,
                "card_roi_bottom": 0.955,
            }
        }

        def controlled_sleep(seconds: float) -> bool:
            clock.advance(seconds)
            return False

        engine._sleep = controlled_sleep
        timing: dict[str, float] = {}
        with (
            patch("crbot.engine.time.monotonic", side_effect=clock.monotonic),
            patch(
                "crbot.engine.estimate_elixir",
                side_effect=[(7.0, 0.9), (4.0, 0.9), (4.0, 0.9)],
            ),
        ):
            resolved, _, _, confirmation, _, _ = engine._execute_action(
                image,
                _decision(),
                {},
                timing=timing,
            )

        self.assertIsNotNone(tracker.registered_at)
        self.assertGreaterEqual(float(tracker.registered_at), 0.99)
        self.assertEqual(confirmation.status, "confirmed")
        self.assertEqual(confirmation.observed_frames, 2)
        self.assertEqual(resolved.action_status, "confirmed")
        self.assertEqual(engine.policy.resolved_status, "confirmed")
        sent_event = engine.recorder.events[-1]
        self.assertEqual(sent_event[0], "battle_action_sent")
        self.assertEqual(sent_event[1]["confirmation_status"], "confirmed")
        self.assertEqual(sent_event[1]["confirmation_frame_role"], "final_observation")
        self.assertTrue(sent_event[1]["confirmation_window_starts_after_send"])

    def test_policy_rejected_reservation_rolls_back_mutations(self) -> None:
        config = {
            "timing": {"battle_action_cooldown_s": [1.0, 1.0]},
            "vision": {"elixir_roi": [0, 0, 1, 1], "card_slot_centers": []},
            "policy": {"version": "test", "initial_elixir": 5, "seed": 1},
        }
        policy = BattlePolicy(config)
        before = policy.snapshot_state()
        policy.virtual_elixir = 1.0
        policy.action_sequence = 9
        decision = policy.prepare_action(_decision(), before)
        self.assertTrue(decision.action_id)
        self.assertEqual(policy.reserved_elixir, 3.0)
        self.assertTrue(policy.resolve_action(decision.action_id, "rejected", now=2.0))
        self.assertEqual(policy.virtual_elixir, before["virtual_elixir"])
        self.assertEqual(policy.action_sequence, before["action_sequence"])
        self.assertIsNone(policy.pending_action_id)
        self.assertEqual(policy.reserved_elixir, before["reserved_elixir"])
        self.assertFalse(policy.resolve_action(decision.action_id, "rejected", now=2.0))

    def test_replay_training_reads_only_confirmed_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "r1"
            recorder = ExperienceReplayRecorder(
                run,
                {"enabled": True, "allow_bot_training": True},
            )
            image = Image.new("RGB", (100, 100), "black")
            recorder.start_battle(1)
            payload = {
                "slot_index": 0,
                "card_id": "test_card",
                "deploy_point": [0.3, 0.7],
                "hand": ["test_card"],
                "action_id": "a",
                "action_status": "unknown",
                "elixir": 7,
            }
            recorder.record_action(1, image, payload, None)
            recorder.finish_battle(
                1,
                BattleResult("win", 1, 0, 1.0, (1, 0, 0), (0, 0, 0)),
                None,
            )
            catalog = CardCatalog(
                [
                    CardDefinition(
                        "test_card", None, "", "test", 3, "", None, "", "", (),
                        "troop", (), (), (), (),
                    )
                ]
            )
            config = {
                "training_policy_version": "unknown",
                "require_action_confirmation": True,
            }
            self.assertEqual(collect_replay_learning_actions(root, catalog, config), [])

    def test_frozen_group_split_is_deterministic_and_disjoint(self) -> None:
        groups = [f"battle-{i}" for i in range(30)]
        first = reserve_frozen_groups(
            groups, validation_fraction=0.2, freeze_fraction=0.1, seed=7
        )
        second = reserve_frozen_groups(
            list(reversed(groups)), validation_fraction=0.2, freeze_fraction=0.1, seed=7
        )
        self.assertEqual(first, second)
        self.assertTrue(first.to_dict()["groups_are_disjoint"])
        self.assertTrue(first.frozen)

    def test_content_fingerprint_detects_changed_rows_but_not_order(self) -> None:
        rows = [{"battle": "a", "reward": 1}, {"battle": "b", "reward": -1}]
        self.assertEqual(content_fingerprint(rows), content_fingerprint(reversed(rows)))
        self.assertNotEqual(
            content_fingerprint(rows),
            content_fingerprint([{**rows[0], "reward": 0}, rows[1]]),
        )

    def test_experiment_protocol_fingerprint_freezes_config_and_split(self) -> None:
        base = experiment_protocol_fingerprint(
            data_fingerprint="data",
            split={"train_groups": ["a"], "frozen_groups": ["z"]},
            filter_config={"policy": "v5"},
            model_config={"seed": 7},
        )
        changed = experiment_protocol_fingerprint(
            data_fingerprint="data",
            split={"train_groups": ["a"], "frozen_groups": ["z"]},
            filter_config={"policy": "v5"},
            model_config={"seed": 8},
        )
        self.assertNotEqual(base, changed)

    def test_champion_exposure_reports_seen_frozen_games(self) -> None:
        champion = {
            "version": "old",
            "manifest": {
                "train_battles": ["train-a", "frozen-seen"],
                "validation_battles": ["val-a"],
            },
        }
        report = champion_frozen_exposure(champion, ["frozen-new", "frozen-seen"])
        self.assertEqual(report["status"], "overlap")
        self.assertEqual(report["overlap_groups"], ["frozen-seen"])
        self.assertFalse(champion_frozen_exposure(None, ["a"])["overlap_groups"])


if __name__ == "__main__":
    unittest.main()
