from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from crbot.action_confirmation import (
    ActionConfirmationTracker,
    assess_action_evidence,
)
from crbot.battle_perception import HandCardMatch
from crbot.cards import CardCatalog, CardDefinition
from crbot.policy import BattleDecision, BattlePolicy
from crbot.replay import ExperienceReplayRecorder
from crbot.replay_evaluation import reserve_frozen_groups
from crbot.replay_learning import collect_replay_learning_actions
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
        confirmed = tracker.observe("a", evidence, now=0.7)
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


if __name__ == "__main__":
    unittest.main()
