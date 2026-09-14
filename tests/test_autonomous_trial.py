from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crbot.autonomous_trial import AutonomousTrial, DEFAULTS, judge, trial_settings
from crbot.learning_store import atomic_json, read_json
from crbot.replay_learning import (ReplayPolicyRegistry, ReplayPolicyModel,
                                    collect_replay_learning_actions, train_replay_policy)
from crbot.training_sync import SyncBusyError, exclusive
from tests import test_replay as fixtures


def row(arm, outcome="win", **changes):
    return dict(arm=arm, outcome=outcome, verified=True, actions=10, confirmed=10,
                decisions=10, fallbacks=0, learned=5 if arm == "candidate" else 0,
                max_plan_ms=100, **changes)


class TrialStatisticsTests(unittest.TestCase):
    def test_fixed_budget_and_multiple_candidate_correction(self):
        rows = [row("baseline", "loss"), row("candidate")] * 1000
        self.assertEqual(judge(rows[:100], DEFAULTS, .025)[0], "battle_trial")
        phase, report = judge(rows, DEFAULTS, .025)
        self.assertEqual(phase, "battle_pass")
        self.assertTrue(report["deployment_pending"])
        self.assertGreater(report["lower_gain"], DEFAULTS["minimum_gain"])
        self.assertLess(judge(rows, DEFAULTS, .001)[1]["lower_gain"], report["lower_gain"])

    def test_equal_or_worse_candidate_does_not_pass(self):
        rows = [row("baseline"), row("candidate")] * 1000
        self.assertEqual(judge(rows, DEFAULTS, .025)[0], "inconclusive")
        rows = [row("baseline"), row("candidate", "loss")] * 1000
        self.assertEqual(judge(rows, DEFAULTS, .025)[0], "inconclusive")

    def test_health_guardrails_do_not_wait_for_win_budget(self):
        for key, value in (("verified", False), ("confirmed", 0), ("fallbacks", 10),
                           ("max_plan_ms", 3000), ("learned", 0)):
            with self.subTest(key=key):
                bad = row("candidate")
                bad[key] = value
                self.assertEqual(judge([row("baseline"), bad] * 10, DEFAULTS, .025)[0], "rejected")

    def test_protocol_validation(self):
        for changes in ({"battles_per_arm": 13}, {"batch_size": 1.5},
                        {"family_alpha": float("nan")}, {"minimum_gain": 2}):
            with self.assertRaises(ValueError):
                trial_settings({"self_learning_trial": changes})


class TrialLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = ReplayPolicyRegistry(self.root)
        path = self.registry.root / "candidates/model.npz"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"pinned model fixture")
        self.entry = dict(version="candidate", model_path="candidates/model.npz",
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), quality_passed=True,
            manifest={"action_value": {"enabled": True}}, influence_scale=.1)
        atomic_json(self.registry.path, dict(champion=None, candidates=[self.entry]))
        self.config = dict(policy={"decision_engine": "predictive"}, replay={"allow_bot_training": True},
                           self_learning_trial=dict(enabled=True, battles_per_arm=20, batch_size=1, health_battles=2))
        self.policy = SimpleNamespace(replay_model=None)
        self.trial = self.make_trial()

    def make_trial(self):
        trial = AutonomousTrial(self.root, self.config, self.policy)
        trial._load = lambda entry: (None if entry is None else SimpleNamespace(
            champion=entry, available=True, action_value_head=SimpleNamespace(enabled=True), influence_scale=.1))
        self.addCleanup(trial.close)
        return trial

    def episode(self, metadata):
        return dict(episode_id=f"test-{metadata['sl3_index']}", timestamp_unix=time.time(),
            policy=deepcopy(metadata), reward_verified=True,
            outcome="win" if metadata["sl3_arm"] == "candidate" else "loss",
            action_count=10, confirmed_action_count=10,
            sl3_runtime=dict(decisions=10, fallbacks=0, learned=5 if metadata["sl3_arm"] == "candidate" else 0, max_plan_ms=100))

    def finish_one(self):
        self.trial.boundary()
        metadata = self.trial.start_battle()
        self.trial.observe(self.episode(metadata))
        return metadata

    def test_two_models_alternate_and_pass_does_not_write_champion(self):
        for i in range(40):
            self.assertTrue(self.trial.boundary())
            self.assertEqual(self.policy.replay_model is None, self.trial.pending["arm"] == "baseline")
            metadata = self.trial.start_battle()
            self.assertEqual(metadata["sl3_index"], i)
            self.trial.observe(self.episode(metadata))
        self.assertEqual(self.trial.ledger["active"]["status"], "battle_pass")
        self.assertIsNone(self.registry.champion())
        proposals = list((self.trial.directory / "proposals").glob("*.json"))
        self.assertEqual(len(proposals), 1)
        self.assertFalse(read_json(proposals[0])["deployed"])
        self.trial.boundary()
        self.assertIsNone(self.policy.replay_model)
        self.assertFalse(self.trial.active)

    def test_pause_keeps_assignment_and_normal_battles_out_of_trial(self):
        self.trial.boundary()
        atomic_json(self.root / "training/self_learning/control.json", {"paused": True})
        self.trial.boundary()
        self.assertFalse(self.trial.start_battle()["sl3_evaluation_only"])
        self.trial.observe({"policy": {}})
        self.assertEqual(len(self.trial.ledger["active"]["rows"]), 0)
        atomic_json(self.root / "training/self_learning/control.json", {"paused": False})
        self.trial.boundary()
        self.assertEqual(self.trial.start_battle()["sl3_index"], 0)

    def test_restart_preserves_next_arm_and_never_counts_same_result_twice(self):
        metadata = self.finish_one()
        self.trial.observe(self.episode(metadata))
        self.trial.close()
        self.trial = self.make_trial()
        self.trial.boundary()
        self.assertNotEqual(self.trial.start_battle()["sl3_arm"], metadata["sl3_arm"])
        self.assertEqual(len(self.trial.ledger["active"]["rows"]), 1)

    def test_unfinished_battle_cannot_be_silently_retried(self):
        self.trial.boundary()
        self.trial.start_battle()
        self.trial.close()
        self.trial = self.make_trial()
        self.trial.boundary()
        self.assertEqual(self.trial.ledger["history"][-1]["status"], "interrupted")
        self.assertFalse(self.trial.active)

    def test_crash_after_replay_write_recovers_exact_completed_result(self):
        self.trial.boundary()
        episode = self.episode(self.trial.start_battle())
        path = self.root / "runs/test/replay_episodes.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
        copied = self.root / "runs/copied/replay_episodes.jsonl"
        copied.parent.mkdir(parents=True)
        copied.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        self.trial.close()
        self.trial = self.make_trial()
        self.trial.boundary()
        self.assertEqual(len(self.trial.ledger["active"]["rows"]), 1)
        self.assertEqual(self.trial.start_battle()["sl3_index"], 1)

    def test_tampered_model_or_external_champion_change_aborts(self):
        self.finish_one()
        (self.registry.root / self.entry["model_path"]).write_bytes(b"changed")
        self.trial.boundary()
        self.assertEqual(self.trial.ledger["active"]["status"], "rejected")
        self.assertIsNone(self.policy.replay_model)

    def test_foreign_assignment_cannot_enter_statistics(self):
        self.trial.boundary()
        episode = self.episode(self.trial.start_battle())
        episode["policy"]["sl3_model_sha256"] = "foreign"
        self.trial.observe(episode)
        self.assertEqual(self.trial.ledger["active"]["reason"], "assignment_contamination")

    def test_training_and_second_owner_are_locked_during_trial(self):
        self.trial.boundary()
        with self.assertRaises(SyncBusyError):
            with exclusive(self.root / ".training-sync/training.lock"):
                pass
        other = self.make_trial()
        with self.assertRaises(SyncBusyError):
            other.boundary()

    def test_changed_configuration_invalidates_restored_trial(self):
        self.finish_one()
        self.trial.close()
        self.config["prediction"] = {"budget_ms": 42}
        self.trial = self.make_trial()
        self.trial.boundary()
        self.assertEqual(self.trial.ledger["active"]["status"], "rejected")

    def test_cooldown_observations_do_not_inflate_learning_evidence(self):
        self.trial.boundary()
        self.trial.start_battle()
        event = dict(world={"revision": 1}, actual_engine="predictive",
                     plan={"revision": 1, "elapsed_ms": 5, "learning": {"applied": True}})
        self.trial.decision(event)
        self.trial.decision(event)
        event["world"]["revision"] = 2
        self.trial.decision(event)
        self.assertEqual(self.trial.counters["decisions"], 1)


class TrialModelAndDataTests(unittest.TestCase):
    def test_explicit_model_load_checks_hash_and_leaves_registry_unchanged(self):
        fixture = fixtures.ReplayPolicyLearningTests()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            catalog = fixture._dataset(root)
            config = fixture._config(True)
            entry = train_replay_policy(root, catalog, config, candidate_only=True)["candidate"]
            model = ReplayPolicyModel(root, config, entry=entry)
            self.assertTrue(model.available)
            self.assertIsNone(ReplayPolicyRegistry(root).champion())
            entry["model_sha256"] = "bad"
            self.assertFalse(ReplayPolicyModel(root, config, entry=entry).available)
            entry["model_path"] = "../../outside.npz"
            self.assertFalse(ReplayPolicyModel(root, config, entry=entry).available)

    def test_trial_data_is_excluded_even_if_training_flag_was_true(self):
        fixture = fixtures.ReplayPolicyLearningTests()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            catalog = fixture._dataset(root)
            path = root / "runs/learning-run/replay_transitions.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for value in rows:
                value.setdefault("policy", {})["sl3_evaluation_only"] = True
                value["eligible_for_training"] = True
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            self.assertEqual(collect_replay_learning_actions(root, catalog, fixture._config(True)), [])


class TrialEngineTests(unittest.TestCase):
    def test_lobby_model_switch_discards_frame_before_start_click(self):
        from threading import Event
        from unittest.mock import Mock
        from PIL import Image
        from crbot.engine import BotEngine
        from tests.test_core import base_config
        with tempfile.TemporaryDirectory() as d:
            config = base_config()
            config.update(game={"package": "game"}, automation={"chest_screen_enabled": False,
                "start_battle_point": [.5, .75], "battle_ui_roi": [.1, .94, .9, .999]}, workflow=[])
            device = Mock()
            device.foreground_package.return_value = "game"
            engine = BotEngine(device, config, Path(d) / "config.json", stop_event=Event())
            engine.recognizer.match_all = Mock(return_value=[])
            engine.auto_trial = Mock(active=True, phase="battle_trial")
            engine.auto_trial.boundary.side_effect = [True, False]
            image = Image.new("RGB", (30, 30))
            def capture():
                engine.offline_verified = True
                return image
            engine._capture_frame = Mock(side_effect=capture)
            engine._update_offline_gate = Mock(return_value=SimpleNamespace(score=1.))
            def tap(*args):
                engine.request_stop()
                return [15, 20]
            engine._tap = Mock(side_effect=tap)
            with patch("crbot.engine.battle_ui_score", return_value=0), \
                 patch("crbot.engine.find_result_confirm_button", return_value=(None, 0)):
                engine._run_single_marker("game")
            self.assertEqual(engine.auto_trial.boundary.call_count, 2)
            self.assertEqual(engine._capture_frame.call_count, 2)
            engine._tap.assert_called_once()
            engine.auto_trial.start_battle.assert_not_called()

    def test_dry_run_cannot_create_live_trial(self):
        from unittest.mock import Mock
        from crbot.engine import BotEngine
        from tests.test_core import base_config
        with tempfile.TemporaryDirectory() as d:
            config = base_config()
            config["self_learning_trial"] = {"enabled": True}
            config["policy"]["decision_engine"] = "predictive"
            engine = BotEngine(Mock(), config, Path(d) / "config.json", dry_run=True)
            self.assertIsNone(engine.auto_trial)
