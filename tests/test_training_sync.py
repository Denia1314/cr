from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

from crbot.training_sync import (
    ReplaySync, SyncError, atomic_write, command, encode, read_json,
    complete_rows, exclusive, prepare_training,
)
from crbot.replay import audit_replay
from crbot.cards import CardCatalog
from crbot.replay_learning import collect_replay_learning_actions, _temporal_group_split


def write_episode(root: Path, battle: int = 1, complete: bool = True) -> Path:
    run = root / "runs/20260905_120000"
    run.mkdir(parents=True, exist_ok=True)
    transition = {"transition_id": f"same-id-b{battle}", "battle_index": battle,
                  "done": True, "outcome": "win", "reward_verified": True,
                  "policy": {"rule_version": "v5"}, "state": {"hand": ["test_card", None, None, None]},
                  "action": {"card_id": "test_card", "deploy_point": [0.5, 0.7]},
                  "timestamp_unix": float(battle),
                  "frame": "frames/private.png", "next_frame": "frames/next.png"}
    episode = {"episode_id": f"same-episode-b{battle}", "battle_index": battle,
               "action_count": 1, "outcome": "win", "reward_verified": True,
               "result_frame": "frames/result.png", "policy": {"rule_version": "v5"}}
    with (run / "replay_transitions.jsonl").open("ab") as handle:
        handle.write(encode(transition) if complete else encode(transition)[:-1])
    with (run / "replay_episodes.jsonl").open("ab") as handle:
        handle.write(encode(episode))
    atomic_write(run / "replay_manifest.json", encode({"schema_version": 1}))
    return run


class ReplaySyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.remote = self.base / "remote.git"
        command(["git", "init", "--bare", "-b", "main", str(self.remote)])
        seed = self.base / "seed"
        seed.mkdir()
        command(["git", "init", "-b", "main"], seed)
        atomic_write(seed / "protocol.json", encode({"schema": 1, "trainer_device": "a" * 32}))
        command(["git", "add", "."], seed)
        command(["git", "-c", "user.name=Test", "-c", "user.email=test@localhost",
                 "-c", "commit.gpgsign=false", "commit", "-m", "Protocol"], seed)
        command(["git", "push", str(self.remote), "main"], seed)
        self.a = self.device("a", True)
        self.b = self.device("b", False)

    def device(self, name: str, trainer: bool) -> ReplaySync:
        root = self.base / name
        state = root / ".training-sync"
        atomic_write(state / "device.json", encode({"enabled": True, "trainer": trainer, "device_id": name * 32}))
        config = {"replay": {"training_policy_version": "v5", "allow_bot_training": False}}
        atomic_write(root / "config.json", encode(config))
        atomic_write(root / "data/cards.json", encode({"schema_version": 1, "cards": [{"id": "test_card", "elixir": 3}]}))
        service = ReplaySync(root)
        service.checkout.mkdir(parents=True)
        service.git("init", "-b", "main")
        service.git("remote", "add", "origin", str(self.remote))
        service.git("fetch", "origin", "main")
        service.git("checkout", "-B", "main", "FETCH_HEAD")
        return service

    def test_two_devices_same_names_merge_once_without_images(self):
        write_episode(self.a.root)
        write_episode(self.b.root)
        self.assertEqual(self.a.sync()["exported_episodes"], 1)
        self.assertEqual(self.b.sync()["imported_episodes"], 1)
        self.assertEqual(self.a.sync()["imported_episodes"], 1)
        self.assertEqual(self.a.sync()["imported_episodes"], 0)
        self.assertEqual(self.b.sync()["exported_episodes"], 0)
        for service in (self.a, self.b):
            audit = audit_replay(service.root, {})
            self.assertEqual(audit.episodes, 2)
            self.assertEqual(audit.transitions, 2)
            shared = list((service.root / "runs").glob("shared_*"))
            self.assertEqual(len(shared), 1)
            data = complete_rows(shared[0] / "replay_transitions.jsonl")[0]
            self.assertNotIn("frame", data)
            self.assertNotEqual(data["transition_id"], "same-id-b1")
        actions = []
        for service in (self.a, self.b):
            catalog = CardCatalog.load(service.root / "data/cards.json")
            actions.append(collect_replay_learning_actions(service.root, catalog, {"training_policy_version": "v5"}))
        self.assertEqual(len(actions[0]), 2)
        self.assertEqual({(a.group_id, a.transition_id) for a in actions[0]}, {(a.group_id, a.transition_id) for a in actions[1]})

    def test_temporal_validation_uses_event_time_not_device_name(self):
        write_episode(self.a.root)
        catalog = CardCatalog.load(self.a.root / "data/cards.json")
        action = collect_replay_learning_actions(self.a.root, catalog, {})[0]
        actions = [replace(action, group_id=name, timestamp_unix=stamp,
                           outcome="win" if stamp % 2 else "loss")
                   for stamp, name in enumerate(("z-old", "y-old", "b-new", "a-new"), 1)]
        train, validation = _temporal_group_split(actions, 0.5)
        self.assertEqual({a.group_id for a in validation}, {"a-new", "b-new"})
        self.assertEqual({a.group_id for a in train}, {"z-old", "y-old"})

    def test_incomplete_write_waits_until_last_newline(self):
        run = write_episode(self.a.root, complete=False)
        self.assertEqual(self.a.sync()["exported_episodes"], 0)
        with (run / "replay_transitions.jsonl").open("ab") as handle:
            handle.write(b"\n")
        self.assertEqual(self.a.sync()["exported_episodes"], 1)

    def test_failed_upload_is_retried_without_rewriting_episode(self):
        run = write_episode(self.a.root)
        with patch.object(self.a, "_push", side_effect=SyncError("offline")):
            with self.assertRaises(SyncError):
                self.a.sync()
        original = (run / "replay_transitions.jsonl").read_bytes()
        self.a.sync()
        self.assertEqual(self.b.sync()["imported_episodes"], 1)
        self.assertEqual((run / "replay_transitions.jsonl").read_bytes(), original)

    def test_simultaneous_pushes_preserve_both_devices(self):
        write_episode(self.a.root)
        write_episode(self.b.root)
        real_push = self.a._push
        def competing_push():
            self.b.sync()  # B wins the push after A already fetched.
            real_push()
        with patch.object(self.a, "_push", side_effect=competing_push):
            self.a.sync()
        self.b.sync()
        self.a.sync()
        self.assertEqual(audit_replay(self.a.root, {}).episodes, 2)
        self.assertEqual(audit_replay(self.b.root, {}).episodes, 2)

    def test_modified_published_episode_is_not_overwritten(self):
        run = write_episode(self.a.root)
        self.a.sync()
        self.b.sync()
        episode = complete_rows(run / "replay_episodes.jsonl")[0]
        episode["outcome"] = "loss"
        atomic_write(run / "replay_episodes.jsonl", encode(episode))
        with self.assertRaisesRegex(SyncError, "已发布对局被修改"):
            self.a.sync()
        self.assertEqual(audit_replay(self.b.root, {}).wins, 1)

    def test_training_role_and_cross_process_lock(self):
        with self.assertRaisesRegex(SyncError, "采集机"):
            prepare_training(self.b.root)
        with exclusive(self.a.state / "sync.lock"):
            with self.assertRaisesRegex(SyncError, "正在进行"):
                self.a.sync()

    def make_model(self, passed=True, promoted=False):
        model_root = self.a.root / "models/replay_policy"
        atomic_write(model_root / "candidates/test.npz", b"test-model-bytes")
        candidate = {"version": "one", "created_at_unix": 1, "quality_passed": passed,
                     "status": "champion" if promoted else ("shadow_pass" if passed else "rejected"),
                     "model_path": "candidates/test.npz", "promoted": promoted,
                     "sync_compatibility": self.a.compatibility()}
        atomic_write(model_root / "registry.json", encode({"candidates": [candidate], "champion": candidate if promoted else None}))
        return candidate

    def test_only_qualified_models_transfer_and_shadow_does_not_activate(self):
        self.make_model(passed=False)
        self.assertEqual(self.a.sync()["published_models"], 0)
        self.make_model()
        self.assertEqual(self.a.sync()["published_models"], 1)
        self.assertEqual(self.b.sync()["received_models"], 1)
        self.assertEqual(self.b.sync()["received_models"], 0)
        registry = read_json(self.b.root / "models/replay_policy/registry.json")
        self.assertIsNone(registry["champion"])
        self.assertEqual(len(registry["candidates"]), 1)

    def test_config_can_publish_and_receive_latest_rejected_model(self):
        for service in (self.a, self.b):
            config = read_json(service.root / "config.json")
            config["training_sync"] = {"publish_unvalidated_models": True}
            atomic_write(service.root / "config.json", encode(config))
        self.make_model(passed=False)
        self.assertEqual(self.a.sync()["published_models"], 1)
        self.assertEqual(self.b.sync()["received_models"], 1)
        registry = read_json(self.b.root / "models/replay_policy/registry.json")
        self.assertEqual(registry["candidates"][0]["status"], "rejected")
        self.assertIsNone(registry["champion"])

        status = self.b.status()
        self.assertEqual(status["models"]["remote_latest_candidate"]["status"], "rejected")
        self.assertEqual(status["models"]["downloaded_latest_candidate"]["status"], "rejected")
        self.assertIsNone(status["models"]["champion"])
        self.assertFalse(status["runtime"]["verified"])

    def test_status_distinguishes_remote_downloaded_champion_and_runtime(self):
        candidate = self.make_model(promoted=True)
        self.a.sync()
        self.b._pull()
        before = self.b.status()
        self.assertEqual(before["models"]["remote_latest_candidate"]["version"], "one")
        self.assertIsNone(before["models"]["downloaded_latest_candidate"])
        self.assertIsNone(before["models"]["champion"])

        self.b.sync()
        downloaded = self.b.status()
        self.assertEqual(downloaded["models"]["downloaded_latest_candidate"]["version"], "one")
        self.assertIsNone(downloaded["models"]["champion"])

        config = read_json(self.b.root / "config.json")
        config["replay"]["allow_bot_training"] = True
        atomic_write(self.b.root / "config.json", encode(config))
        self.b.sync()
        champion = self.b.status()
        self.assertEqual(champion["models"]["champion"]["version"], candidate["version"])
        self.assertTrue(champion["models"]["champion_file_present"])

        self.b.write_runtime_status({
            "rule_version": "v5",
            "replay_model": {"version": "one", "loaded": False, "load_error": "bad model"},
        })
        runtime = self.b.status()["runtime"]
        self.assertTrue(runtime["verified"])
        self.assertFalse(runtime["replay_model"]["loaded"])
        self.assertEqual(runtime["replay_model"]["load_error"], "bad model")

    def test_status_marks_incompatible_remote_candidate_without_importing(self):
        self.make_model()
        self.a.sync()
        config = read_json(self.b.root / "config.json")
        config["replay"]["training_policy_version"] = "other"
        atomic_write(self.b.root / "config.json", encode(config))

        status = self.b.status()

        self.assertFalse(status["models"]["remote_compatibility_matches"])
        self.assertIsNone(status["models"]["downloaded_latest_candidate"])

    def test_model_compatibility_and_checksum_failure_preserve_existing(self):
        atomic_write(self.a.root / "crbot/replay.py", b"first\nsecond\n")
        atomic_write(self.b.root / "crbot/replay.py", b"first\r\nsecond\r\n")
        self.assertEqual(self.a.compatibility(), self.b.compatibility())
        self.make_model()
        self.a.sync()
        config = read_json(self.b.root / "config.json")
        config["replay"]["training_policy_version"] = "different"
        atomic_write(self.b.root / "config.json", encode(config))
        self.assertEqual(self.b.sync()["received_models"], 0)
        config["replay"]["training_policy_version"] = "v5"
        atomic_write(self.b.root / "config.json", encode(config))
        pointer = read_json(self.b.checkout / "models/replay_policy.json")
        atomic_write(self.b.checkout / "models" / (pointer["sha256"] + ".npz"), b"corrupt")
        with self.assertRaisesRegex(SyncError, "校验失败"):
            self.b.import_model()
        self.assertFalse((self.b.root / "models/replay_policy/registry.json").exists())

    def test_promoted_model_requires_local_opt_in(self):
        self.make_model(promoted=True)
        self.a.sync()
        self.b.sync()
        self.assertIsNone(read_json(self.b.root / "models/replay_policy/registry.json")["champion"])
        config = read_json(self.b.root / "config.json")
        config["replay"]["allow_bot_training"] = True
        atomic_write(self.b.root / "config.json", encode(config))
        self.assertEqual(self.b.sync()["received_models"], 1)
        self.assertIsNotNone(read_json(self.b.root / "models/replay_policy/registry.json")["champion"])

    def test_bad_device_path_is_rejected(self):
        write_episode(self.a.root)
        self.a.sync()
        self.b._pull()
        path = next((self.b.checkout / "records").glob("*/*/*.json"))
        data = read_json(path)
        data["device_id"] = "../../escape"
        atomic_write(path, encode(data))
        with self.assertRaisesRegex(SyncError, "编号无效"):
            self.b.import_records()


class RuntimeOptInTests(unittest.TestCase):
    def test_policy_does_not_load_replay_model_when_disabled(self):
        from contextlib import ExitStack
        from crbot.policy import BattlePolicy
        config = read_json(Path(__file__).resolve().parents[1] / "config.json")
        config["replay"]["allow_bot_training"] = False
        with ExitStack() as stack:
            for name in ("CardCatalog.load", "UniversalHandRecognizer", "LearnedBattlefieldDetector", "ImitationPolicyModel"):
                stack.enter_context(patch("crbot.policy." + name))
            model = stack.enter_context(patch("crbot.policy.ReplayPolicyModel"))
            policy = BattlePolicy(config)
            model.assert_not_called()
            self.assertIsNone(policy.replay_model)


if __name__ == "__main__":
    unittest.main()
