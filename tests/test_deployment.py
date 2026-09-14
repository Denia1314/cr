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

from crbot.autonomous_trial import AutonomousTrial, judge
from crbot.learning_store import atomic_json, digest, read_json
from crbot.replay_learning import ReplayPolicyRegistry
from crbot.training_sync import ReplaySync, SyncError, exclusive
from crbot.deployment_sync import publish_deployment, import_deployment
from tests import test_training_sync as sync_fixtures


class RealWeightDeploymentTests(unittest.TestCase):
    def test_loads_real_npz_and_predicts_with_committed_weights(self):
        import numpy as np
        from crbot.replay_learning import train_replay_policy
        from tests import test_replay
        fixture = test_replay.ReplayPolicyLearningTests()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            catalog = fixture._dataset(root)
            replay = fixture._config(True)
            entry = train_replay_policy(root, catalog, replay, candidate_only=True)["candidate"]
            path = root / "models/replay_policy" / entry["model_path"]
            # This synthetic fixture exercises loading; it is not a production admission result.
            with np.load(path) as data:
                arrays = {k: data[k].copy() for k in data.files}
            arrays["action_value_enabled"] = np.asarray([1], dtype=np.int8)
            np.savez_compressed(path, **arrays)
            entry["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            config = dict(replay=replay, policy={"decision_engine": "predictive"},
                          self_learning_deployment={"enabled": True})
            atomic_json(root / "config.json", config)
            policy = SimpleNamespace(replay_model=None)
            host = AutonomousTrial(root, config, policy)
            registry = ReplayPolicyRegistry(root).load()
            host.deployment._commit(registry, entry, source="synthetic-load-test", baseline=None,
                report={}, compatibility=entry["sync_compatibility"])
            model = policy.replay_model
            self.assertEqual(model.loaded_model_sha256, ReplayPolicyRegistry(root).champion()["model_sha256"])
            head = model.action_value_head
            prediction = head.predict(str(head.cards[0]), head.x[0], head.xy[0])
            self.assertIsNotNone(prediction)
            self.assertTrue(np.isfinite(prediction["value"]))


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(policy={"decision_engine": "predictive"}, replay={"allow_bot_training": True},
            self_learning_trial=dict(enabled=True, battles_per_arm=20, batch_size=1, health_battles=2),
            self_learning_deployment=dict(enabled=True, probation_battles=4, minimum_health_battles=2))
        atomic_json(self.root / "config.json", self.config)
        self.registry = ReplayPolicyRegistry(self.root)
        path = self.registry.root / "candidates/test.npz"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"new weights")
        self.entry = dict(version="new", model_path="candidates/test.npz", influence_scale=.1,
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), quality_passed=True,
            manifest={"action_value": {"enabled": True}}, sync_compatibility=ReplaySync(self.root).compatibility(self.config["replay"]))
        atomic_json(self.registry.path, dict(champion=None, candidates=[self.entry]))
        self.policy = SimpleNamespace(replay_model=None)
        self.host = self.make_host()
        rows = [dict(arm=arm, episode_id=f"e{i}", outcome="loss" if arm == "baseline" else "win",
            verified=True, actions=10, confirmed=10, decisions=10, fallbacks=0,
            learned=0 if arm == "baseline" else 5, max_plan_ms=100)
            for i, arm in enumerate(["baseline", "candidate"] * 20)]
        phase, report = judge(rows, self.host.settings, .025)
        self.assertEqual(phase, "battle_pass")
        self.trial = dict(id="passed", status=phase, candidate=self.entry, baseline=None,
            registry_champion=digest(None), identity=self.host._identity(), protocol=self.host.settings,
            alpha=.025, rows=rows, report=report, first_arms=["baseline"]*20)
        self.host.ledger = dict(schema=1, attempts=1, family_alpha=.05, active=None, history=[self.trial])
        self.host._save()
        self.host._proposal(self.trial)

    def make_host(self):
        host = AutonomousTrial(self.root, self.config, self.policy)
        def load(entry):
            if entry is None:
                return None
            checksum = hashlib.sha256((self.registry.root / entry["model_path"]).read_bytes()).hexdigest()
            return SimpleNamespace(champion=deepcopy(entry), available=True, loaded_model_sha256=checksum,
                                   action_value_head=SimpleNamespace(enabled=True), influence_scale=.1)
        host._load = load
        self.addCleanup(host.close)
        return host

    def episode(self, metadata, *, healthy=True):
        return dict(episode_id="deployed-" + str(metadata["sl4_index"]), timestamp_unix=time.time(),
            policy=deepcopy(metadata), outcome="win", reward_verified=True, action_count=10,
            confirmed_action_count=10 if healthy else 0,
            sl4_runtime=dict(decisions=10, fallbacks=0, learned=5, max_plan_ms=100))

    def play(self, healthy=True):
        self.host.boundary()
        metadata = self.host.start_battle()
        self.host.observe(self.episode(metadata, healthy=healthy))
        return metadata

    def test_verified_proposal_commits_and_really_loads_same_hash_once(self):
        self.assertTrue(self.host.boundary())
        registry = self.registry.load()
        self.assertEqual(registry["deployment"]["state"], "probation")
        self.assertEqual(registry["champion"]["model_sha256"], self.policy.replay_model.loaded_model_sha256)
        self.assertFalse(self.host.boundary())
        self.assertEqual(self.registry.load()["deployment"]["generation"], 1)

    def test_recovery_does_not_reject_an_already_committed_proposal(self):
        with patch.object(self.host.deployment, "_activate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.host.boundary()
        self.host.close()
        self.host = self.make_host()
        self.host.boundary()
        self.assertEqual(self.registry.load()["deployment_proposals"]["passed"], "deployed")

    def test_tampered_evidence_never_changes_champion(self):
        path = next((self.host.directory / "proposals").glob("*.json"))
        proposal = read_json(path)
        proposal["evidence_sha256"] = "wrong"
        atomic_json(path, proposal)
        self.host.boundary()
        self.assertIsNone(self.registry.champion())
        self.assertIn("rejected", self.registry.load()["deployment_proposals"]["passed"])

    def test_probation_completes_before_stable_and_continues_monitoring(self):
        for i in range(4):
            self.play()
            self.assertEqual(self.registry.load()["deployment"]["state"], "stable" if i == 3 else "probation")
        self.play(healthy=False)
        self.host.boundary()
        self.assertEqual(self.registry.load()["deployment"]["state"], "rolled_back")
        self.assertIsNone(self.registry.champion())

    def test_guardrail_does_not_switch_model_mid_battle(self):
        self.play(healthy=False)
        self.play(healthy=False)
        self.assertEqual(self.registry.load()["deployment"]["state"], "rollback_pending")
        self.assertIsNotNone(self.policy.replay_model)
        self.host.boundary()
        self.assertIsNone(self.policy.replay_model)
        self.assertEqual(self.registry.load()["deployment"]["generation"], 2)
        self.assertEqual(self.registry.load()["deployment"]["state"], "rolled_back")

    def test_crash_after_atomic_commit_recovers_without_second_promotion(self):
        with patch.object(self.host.deployment, "_activate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.host.boundary()
        self.host.close()
        self.host = self.make_host()
        self.host.boundary()
        self.assertEqual(self.policy.replay_model.loaded_model_sha256, self.entry["model_sha256"])
        self.assertEqual(self.registry.load()["deployment"]["generation"], 1)

    def test_memory_activation_failure_reverts_atomic_commit(self):
        real = self.host.deployment._activate
        calls = []
        def activate(model, dep):
            calls.append(dep["generation"])
            if len(calls) == 1:
                raise ValueError("injected load failure")
            return real(model, dep)
        with patch.object(self.host.deployment, "_activate", side_effect=activate):
            self.host.boundary()
        self.assertIsNone(self.registry.champion())
        self.assertEqual(self.registry.load()["deployment"]["state"], "rolled_back")

    def test_unfinished_probation_battle_rolls_back_after_restart(self):
        self.host.boundary()
        self.host.start_battle()
        self.host.close()
        self.host = self.make_host()
        self.host.boundary()
        self.assertEqual(self.registry.load()["deployment"]["state"], "rolled_back")

    def test_completed_replay_recovers_once_after_restart(self):
        self.host.boundary()
        episode = self.episode(self.host.start_battle())
        path = self.root / "runs/probation/replay_episodes.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
        self.host.close()
        self.host = self.make_host()
        self.host.boundary()
        self.assertEqual(self.registry.load()["deployment"]["total_battles"], 1)
        self.host.boundary()
        self.assertEqual(self.registry.load()["deployment"]["total_battles"], 1)

    def test_corrupt_deployed_file_returns_to_unweighted_anchor(self):
        self.host.boundary()
        (self.registry.root / self.entry["model_path"]).write_bytes(b"corrupt")
        self.host.boundary()
        self.assertIsNone(self.policy.replay_model)
        self.assertEqual(self.registry.load()["deployment"]["state"], "rolled_back")

    def test_pause_does_not_deploy_proposal(self):
        atomic_json(self.root / "training/self_learning/control.json", {"paused": True})
        self.host.boundary()
        self.assertIsNone(self.registry.champion())
        self.assertFalse(self.host.deployment.monitoring)

    def test_collector_activates_only_at_boundary_and_newer_rollback_preempts_probation(self):
        atomic_json(self.root / ".training-sync/device.json", dict(enabled=True, trainer=False, device_id="b"*32))
        registry = self.registry.load()
        registry["remote_deployment"] = dict(trainer_device="a"*32, generation=1, deployment_id="remote-1",
            state="stable", entry=self.entry, report=self.trial["report"],
            compatibility=self.entry["sync_compatibility"], runtime_contract=self.host.deployment.contract())
        atomic_json(self.registry.path, registry)
        self.assertIsNone(self.policy.replay_model)
        self.host.boundary()
        self.assertEqual(self.policy.replay_model.loaded_model_sha256, self.entry["model_sha256"])
        registry = self.registry.load()
        registry["remote_deployment"].update(generation=2, deployment_id="remote-rollback", state="rolled_back", entry=None)
        atomic_json(self.registry.path, registry)
        self.host.boundary()
        self.assertIsNone(self.policy.replay_model)
        self.assertEqual(self.registry.load()["remote_consumed"]["generation"], 2)
        self.host.boundary()
        self.assertEqual(self.registry.load()["deployment"]["generation"], 2)

    def test_registry_contention_defers_without_promotion_or_start(self):
        with exclusive(self.host.deployment.lock_path):
            self.assertTrue(self.host.boundary())
            self.assertIsNone(self.registry.champion())
        self.host.boundary()
        self.assertIsNotNone(self.registry.champion())


class DeploymentSyncTests(unittest.TestCase):
    setUp = sync_fixtures.ReplaySyncTests.setUp
    device = sync_fixtures.ReplaySyncTests.device
    make_model = sync_fixtures.ReplaySyncTests.make_model

    def prepare_deployment(self):
        entry = self.make_model(promoted=True)
        entry.update(model_sha256=hashlib.sha256(b"test-model-bytes").hexdigest(), deployment_generation=1,
                     deployment_id="deployment-1", influence_scale=.1)
        path = self.a.root / "models/replay_policy/registry.json"
        registry = read_json(path)
        registry.update(champion=entry, candidates=[entry], deployment=dict(
            id="deployment-1", generation=1, state="stable", entry=entry, runtime_contract="a"*64, compatibility=self.a.compatibility()))
        atomic_json(path, registry)
        return entry

    def test_notification_upload_verified_but_receiver_only_stages_it(self):
        self.prepare_deployment()
        result = self.a.sync()
        self.assertEqual(result["published_deployments"], 1)
        publication = read_json(self.a.root / "models/replay_policy/registry.json")["deployment_publication"]
        self.assertTrue(publication["uploaded"])
        config = read_json(self.b.root / "config.json")
        config["replay"]["allow_bot_training"] = True
        atomic_json(self.b.root / "config.json", config)
        self.b.sync()
        registry = read_json(self.b.root / "models/replay_policy/registry.json")
        self.assertIsNone(registry["champion"])
        self.assertEqual(registry["remote_deployment"]["generation"], 1)

    def test_old_and_equivocated_notifications_cannot_overwrite_newer(self):
        self.prepare_deployment()
        self.a.sync()
        self.b.sync()
        path = self.b.checkout / "models/deployment.json"
        pointer = read_json(path)
        pointer["generation"] = 2
        pointer["deployment_id"] = "new"
        atomic_json(path, pointer)
        self.assertEqual(import_deployment(self.b), 1)
        pointer["generation"] = 1
        atomic_json(path, pointer)
        self.assertEqual(import_deployment(self.b), 0)
        pointer["generation"] = 2
        pointer["deployment_id"] = "changed"
        atomic_json(path, pointer)
        with self.assertRaises(SyncError):
            import_deployment(self.b)

    def test_push_failure_does_not_acknowledge_and_next_sync_retries(self):
        self.prepare_deployment()
        with patch.object(self.a, "_push", side_effect=SyncError("offline")):
            with self.assertRaises(SyncError):
                self.a.sync()
        path = self.a.root / "models/replay_policy/registry.json"
        self.assertNotIn("deployment_publication", read_json(path))
        self.a.sync()
        self.assertTrue(read_json(path)["deployment_publication"]["uploaded"])

    def test_corrupt_artifact_and_wrong_authority_leave_champion_untouched(self):
        entry = self.prepare_deployment()
        self.a.sync()
        self.b._pull()
        artifact = self.b.checkout / "models" / (entry["model_sha256"] + ".npz")
        artifact.write_bytes(b"bad")
        with self.assertRaises(SyncError):
            import_deployment(self.b)
        path = self.b.checkout / "models/deployment.json"
        pointer = read_json(path)
        pointer["trainer_device"] = "foreign"
        atomic_json(path, pointer)
        with self.assertRaises(SyncError):
            import_deployment(self.b)
        self.assertFalse((self.b.root / "models/replay_policy/registry.json").exists())

    def test_rollback_notice_clears_weights_only_at_later_boundary(self):
        self.prepare_deployment()
        path = self.a.root / "models/replay_policy/registry.json"
        registry = read_json(path)
        registry["deployment"].update(generation=2, id="rollback", state="rolled_back", entry=None)
        registry["champion"] = None
        atomic_json(path, registry)
        self.a.sync()
        self.b.sync()
        remote = read_json(self.b.root / "models/replay_policy/registry.json")["remote_deployment"]
        self.assertIsNone(remote["entry"])
        self.assertEqual(remote["state"], "rolled_back")

    def test_damaged_candidate_does_not_block_rollback_publication(self):
        entry = self.prepare_deployment()
        path = self.a.root / "models/replay_policy/registry.json"
        registry = read_json(path)
        registry["deployment"].update(generation=2, id="rollback", state="rolled_back", entry=None)
        registry["champion"] = None
        atomic_json(path, registry)
        (self.a.root / "models/replay_policy" / entry["model_path"]).write_bytes(b"damaged")
        result = self.a.sync()
        self.assertIsNotNone(result["candidate_model_error"])
        self.assertTrue(read_json(path)["deployment_publication"]["uploaded"])
        self.b.sync()
        remote = read_json(self.b.root / "models/replay_policy/registry.json")["remote_deployment"]
        self.assertEqual(remote["state"], "rolled_back")
