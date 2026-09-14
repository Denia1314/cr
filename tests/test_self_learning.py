from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch, Mock

from tests import test_replay as fixtures
from crbot.cards import CardCatalog
from crbot.learning_store import LearningStore, atomic_json, read_json
from crbot.replay_learning import ReplayPolicyRegistry, collect_replay_learning_actions, train_replay_policy
from crbot.self_learning import (SelfLearningService, content_groups, effective_learning_config,
    freeze_snapshot, learning_identity, learning_settings, request_config, run_cycle, verify_snapshot)
from crbot.training_sync import SyncBusyError


class SelfLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = fixtures.ReplayPolicyLearningTests()._dataset(self.root)
        cards = [dict(asdict(c), id=c.card_id) for c in self.catalog.cards]
        atomic_json(self.root / "data/cards.json", {"schema_version": 1, "cards": cards})
        self.config = {"policy": {"version": "v4", "decision_engine": "legacy"},
            "training": {"device": "cpu"}, "replay": fixtures.ReplayPolicyLearningTests._config(True),
            "self_learning": {"minimum_new_battles": 4, "minimum_new_actions": 8, "minimum_interval_s": 0}}
        atomic_json(self.root / "config.json", self.config)
        run = self.root / "runs/learning-run"
        rows = [json.loads(x) for x in (run / "replay_transitions.jsonl").read_text().splitlines()]
        for i, row in enumerate(rows):
            row["timestamp_unix"] = 100 + i
            row["action_status"] = "confirmed"
        (run / "replay_transitions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        episodes = [{"battle_index": i, "action_count": 2, "outcome": "win" if i <= 4 else "loss",
                     "reward_verified": True, "policy": {"rule_version": "v4"}} for i in range(1,9)]
        (run / "replay_episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in episodes))
        self.request = request_config(self.root, self.config)

    def test_candidate_only_preserves_existing_champion_and_never_syncs(self):
        registry = ReplayPolicyRegistry(self.root)
        champion = train_replay_policy(self.root, self.catalog, self.config["replay"])["candidate"]
        with patch("crbot.training_sync.prepare_training", side_effect=AssertionError("no sync")), \
             patch("crbot.training_sync.ReplaySync.sync", side_effect=AssertionError("no sync")):
            result = train_replay_policy(self.root, self.catalog, self.config["replay"], candidate_only=True)
        self.assertTrue(result["candidate"]["quality_passed"])
        self.assertFalse(result["candidate"]["promoted"])
        self.assertEqual(registry.champion(), champion)

    def test_freeze_ignores_active_partial_battle_and_detects_tampering(self):
        run = self.root / "runs/learning-run"
        with (run / "replay_transitions.jsonl").open("a") as f:
            f.write('{"battle_index":99,"policy":{"rule_version":"v4"}}\n{"partial":')
        snapshot = self.root / "snapshot"
        freeze_snapshot(self.root, snapshot, self.request)
        verify_snapshot(snapshot)
        self.assertEqual(len(collect_replay_learning_actions(snapshot, self.catalog, self.config["replay"])),16)
        (snapshot / "data/cards.json").write_text("{}")
        with self.assertRaises(ValueError): verify_snapshot(snapshot)

    def test_crash_during_freeze_is_cleaned_before_next_cycle(self):
        orphan=LearningStore(self.root).root/"snapshots"/"interrupted-freeze"
        orphan.mkdir(parents=True);(orphan/"partial.json").write_text("partial")
        self.assertEqual(run_cycle(self.root,self.request)["phase"],"candidate_ready")
        self.assertFalse(orphan.exists())

    def test_completed_count_mismatch_blocks_snapshot(self):
        path = self.root / "runs/learning-run/replay_transitions.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1])+"\n")
        with self.assertRaisesRegex(ValueError, "数量不一致"):
            freeze_snapshot(self.root, self.root/"snapshot", self.request)

    def test_mode_identity_and_shared_effective_config(self):
        atomic_json(self.root / ".royal-lab.json", {"runtime_model":"m3_c3", "decision_engine":"predictive"})
        effective = effective_learning_config(self.config, self.root/"config.json")
        self.assertEqual(effective["replay"]["training_policy_version"], "predictive_calibrated_v2")
        self.assertNotEqual(learning_identity(request_config(self.root,effective)), learning_identity(self.request))
        self.assertEqual(self.config["replay"]["training_policy_version"], "v4")

    def test_copied_content_does_not_trigger_another_job(self):
        first = run_cycle(self.root,self.request)
        self.assertEqual(first["phase"],"candidate_ready")
        self.assertFalse(ReplayPolicyRegistry(self.root).load()["candidates"][0]["promoted"])
        shutil.copytree(self.root/"runs/learning-run", self.root/"runs/copied")
        second = run_cycle(self.root,self.request)
        self.assertEqual(second["phase"], "waiting_data")
        self.assertEqual(second["new_actions"],0)
        self.assertEqual(len(ReplayPolicyRegistry(self.root).load()["candidates"]),1)

    def test_resume_after_crash_uses_frozen_inputs_not_live_data(self):
        with patch("crbot.replay_learning.train_replay_policy", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): run_cycle(self.root,self.request)
        state = LearningStore(self.root).load(learning_identity(self.request))
        self.assertEqual(state["jobs"][0]["phase"],"training")
        (self.root/"runs/learning-run/replay_transitions.jsonl").write_text("corrupt\n")
        result=run_cycle(self.root,self.request)
        self.assertEqual(result["phase"],"candidate_ready")
        self.assertEqual(LearningStore(self.root).load(learning_identity(self.request))["jobs"][0]["attempts"],2)

    def test_crash_after_candidate_commit_is_reconciled_without_retraining(self):
        with patch("crbot.self_learning._finish_job", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): run_cycle(self.root,self.request)
        with patch("crbot.replay_learning.train_replay_policy", side_effect=AssertionError("duplicate")):
            result=run_cycle(self.root,self.request)
        self.assertEqual(result["phase"],"candidate_ready")
        self.assertEqual(len(ReplayPolicyRegistry(self.root).load()["candidates"]),1)

    def test_training_lock_defers_without_spending_attempt(self):
        with patch("crbot.replay_learning.train_replay_policy", side_effect=SyncBusyError("busy")):
            self.assertEqual(run_cycle(self.root,self.request)["phase"],"deferred")
        state=LearningStore(self.root).load(learning_identity(self.request))
        self.assertEqual(state["jobs"][0]["attempts"],0)

    def test_recovery_does_not_accept_a_corrupt_registered_candidate(self):
        with patch("crbot.self_learning._finish_job",side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): run_cycle(self.root,self.request)
        registry=ReplayPolicyRegistry(self.root)
        model=registry.root/registry.load()["candidates"][0]["model_path"]
        model.write_bytes(b"corrupt")
        self.assertEqual(run_cycle(self.root,self.request)["phase"],"failed")
        self.assertIsNone(registry.champion())

    def test_collector_and_pause_do_not_train(self):
        with patch("crbot.training_sync.training_allowed",return_value=False), \
             patch("crbot.replay_learning.train_replay_policy",side_effect=AssertionError):
            self.assertEqual(run_cycle(self.root,self.request)["phase"],"collector")
        atomic_json(LearningStore(self.root).root/"control.json",{"paused":True})
        self.assertEqual(run_cycle(self.root,self.request)["phase"],"paused")

    def test_invalid_budget_and_stop_do_not_spawn(self):
        with self.assertRaises(ValueError): learning_settings({"self_learning":{"max_cycle_seconds":float('nan')}})
        service=SelfLearningService(self.root,self.config,lambda:True)
        with patch("crbot.self_learning.subprocess.Popen",side_effect=AssertionError):
            self.assertFalse(service.boundary())

    def test_failed_job_is_not_repeated_for_same_snapshot(self):
        with patch("crbot.replay_learning.train_replay_policy",side_effect=ValueError("bad candidate")):
            with self.assertRaises(ValueError): run_cycle(self.root,self.request)
        with patch("crbot.replay_learning.train_replay_policy",side_effect=AssertionError):
            self.assertEqual(run_cycle(self.root,self.request)["phase"],"waiting_data")

    def test_snapshot_capacity_and_corrupt_ledger_fail_closed(self):
        request = {**self.request,"settings":{**self.request["settings"],"max_snapshot_mb":0}}
        with self.assertRaises(ValueError): freeze_snapshot(self.root,self.root/"tiny",request)
        store=LearningStore(self.root)
        path=store.state_path(learning_identity(self.request)); path.parent.mkdir(parents=True)
        path.write_text("broken")
        with self.assertRaises(ValueError): run_cycle(self.root,self.request)

    def test_second_increment_generates_a_distinct_candidate(self):
        first=run_cycle(self.root,self.request)
        run=self.root/"runs/learning-run"
        for name in ("replay_episodes.jsonl","replay_transitions.jsonl"):
            path=run/name
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                row["battle_index"]+=8
                if "transition_id" in row:
                    row["transition_id"]="new:"+row["transition_id"]
                    row["timestamp_unix"]+=1000
            with path.open("a") as f: f.write("".join(json.dumps(r)+"\n" for r in rows))
        second=run_cycle(self.root,self.request)
        self.assertEqual(second["phase"],"candidate_ready")
        self.assertNotEqual(first["job_id"],second["job_id"])
        self.assertIsNone(ReplayPolicyRegistry(self.root).champion())

    def test_two_interruptions_exhaust_retry_budget(self):
        for _ in range(2):
            with patch("crbot.replay_learning.train_replay_policy",side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt): run_cycle(self.root,self.request)
        with patch("crbot.replay_learning.train_replay_policy",side_effect=AssertionError):
            self.assertEqual(run_cycle(self.root,self.request)["phase"],"failed")

    def test_subprocess_cycle_completes_and_registry_stays_unpromoted(self):
        with patch.dict(os.environ,{"PYTHONPATH":str(Path(__file__).resolve().parents[1])}):
            service=SelfLearningService(self.root,self.config)
            self.assertTrue(service.boundary())
        self.assertEqual(service.phase,"candidate_ready")
        self.assertIsNone(ReplayPolicyRegistry(self.root).champion())

    def test_running_child_is_terminated_on_stop(self):
        stop=Mock(side_effect=[False,True])
        service=SelfLearningService(self.root,self.config,stop)
        process=Mock(); process.poll.side_effect=[None,1];process.returncode=1
        with patch("crbot.self_learning.subprocess.Popen",return_value=process):
            self.assertTrue(service.boundary())
        process.terminate.assert_called_once()
        self.assertEqual(service.phase,"paused")

    def test_running_child_is_terminated_on_budget(self):
        service=SelfLearningService(self.root,self.config)
        process=Mock();process.poll.side_effect=[None,1];process.returncode=1
        with patch("crbot.self_learning.subprocess.Popen",return_value=process), \
             patch("crbot.self_learning.time.monotonic",side_effect=[0,8000]):
            self.assertTrue(service.boundary())
        process.terminate.assert_called_once()
        self.assertEqual(service.phase,"budget_exhausted")


if __name__ == "__main__": unittest.main()
