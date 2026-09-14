"""SL0/SL1: bounded, resumable, candidate-only learning at battle boundaries."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .learning_store import LearningStore, atomic_json, digest, read_json

DEFAULTS = {
    "enabled": True, "audit_every_battles": 20, "minimum_new_battles": 100,
    "minimum_new_actions": 500, "minimum_interval_s": 7200,
    "max_cycle_seconds": 7200, "max_attempts": 2,
    "max_snapshot_mb": 512, "retained_snapshots": 2,
}


def learning_settings(config: dict) -> dict:
    value = {**DEFAULTS, **config.get("self_learning", {})}
    for key in DEFAULTS.keys() - {"enabled"}:
        number = float(value[key])
        if not math.isfinite(number) or number < (0 if key == "minimum_interval_s" else 1):
            raise ValueError(f"self_learning.{key} 无效")
        value[key] = number if key.endswith("_s") else int(number)
    if not isinstance(value["enabled"], bool):
        raise ValueError("self_learning.enabled 必须为布尔值")
    return value


def effective_learning_config(config: dict, config_path: Path) -> dict:
    """Shared GUI/CLI overlay. Never write profile keys or base configuration."""
    from .runtime_model import (apply_decision_engine, apply_runtime_model,
                                load_decision_engine, load_runtime_model)
    effective = apply_runtime_model(config, load_runtime_model(config_path, config))
    return apply_decision_engine(effective, load_decision_engine(config_path, effective))


def code_identity(root: Path) -> str:
    return digest({p.name: hashlib.sha256(p.read_text(encoding="utf-8-sig").encode()).hexdigest()
                   for p in sorted((root / "crbot").glob("*.py"))})


def request_config(root: Path, config: dict) -> dict:
    # Persist only learning-related configuration, never ADB/machine/credentials.
    path = Path(config.get("dataset", {}).get("card_catalog", "data/cards.json"))
    catalog = (root / path).resolve()
    return {"schema": 1, "replay": config.get("replay", {}),
            "policy": config.get("policy", {}), "prediction": config.get("prediction", {}),
            "training": {"device": config.get("training", {}).get("device", "auto")},
            "settings": learning_settings(config), "catalog_path": str(catalog),
            "catalog_hash": hashlib.sha256(catalog.read_bytes()).hexdigest(),
            "code_hash": code_identity(root)}


def learning_identity(request: dict) -> str:
    return digest({k: request[k] for k in
                   ("schema", "replay", "policy", "prediction", "catalog_hash", "code_hash")})


def _rows(path: Path):
    if not path.exists():
        return []
    # Only newline-terminated records are committed. Do not swallow corrupt rows.
    data = path.read_bytes()
    lines = data[:data.rfind(b"\n") + 1].splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def freeze_snapshot(root: Path, destination: Path, request: dict) -> None:
    """Copy completed replay records, not images, models, or active partial battles."""
    from .replay_learning import _policy_version
    versions = {request["replay"].get("training_policy_version", "")}
    versions.update(request["replay"].get("transfer_policy_versions", []))
    total = 0
    files = {}

    def write(relative: Path, data: bytes):
        nonlocal total
        total += len(data)
        if total > request["settings"]["max_snapshot_mb"] * 1024 * 1024:
            raise ValueError("冻结快照超过容量预算")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[relative.as_posix()] = hashlib.sha256(data).hexdigest()

    for run in sorted((root / "runs").glob("*")):
        if not run.is_dir():
            continue
        episodes = [r for r in _rows(run / "replay_episodes.jsonl") if _policy_version(r) in versions
                    and not r.get("policy", {}).get("sl3_evaluation_only", False)]
        if not episodes:
            continue
        indices = {r.get("battle_index") for r in episodes}
        transitions = [r for r in _rows(run / "replay_transitions.jsonl")
                       if r.get("battle_index") in indices and _policy_version(r) in versions]
        for episode in episodes:
            observed = [r for r in transitions if r.get("battle_index") == episode.get("battle_index")]
            if len(observed) != int(episode.get("action_count", -1)):
                raise ValueError("已结算对局与动作数量不一致，暂缓自主训练")
            if any(r.get("outcome") != episode.get("outcome") or
                   bool(r.get("reward_verified")) != bool(episode.get("reward_verified")) for r in observed):
                raise ValueError("已结算对局与动作结果不一致，暂缓自主训练")
        for name, rows in (("replay_episodes.jsonl", episodes), ("replay_transitions.jsonl", transitions)):
            data = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode()
            write(Path("runs") / run.name / name, data)
        for name in (".sync-origin.json", ".shared-replay.json"):
            if (run / name).is_file():
                write(Path("runs") / run.name / name, (run / name).read_bytes())
    data = Path(request["catalog_path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != request["catalog_hash"]:
        raise ValueError("卡库在审计期间发生变化，请重新运行")
    write(Path("data/cards.json"), data)
    atomic_json(destination / "snapshot.json", {"schema": 1, "files": files, "bytes": total})


def verify_snapshot(snapshot: Path) -> None:
    manifest = read_json(snapshot / "snapshot.json")
    if not manifest.get("files"):
        raise ValueError("冻结快照不完整")
    for name, expected in manifest["files"].items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("冻结数据校验失败")


def content_groups(actions) -> dict[str, int]:
    grouped = defaultdict(list)
    for action in actions:
        grouped[action.group_id].append(action)
    result = {}
    for group, rows in grouped.items():
        content = []
        for action in rows:
            value = asdict(action)
            value.pop("group_id")
            value.pop("transition_id")
            content.append(value)
        # Match the production dedup boundary: do not merge unidentifiable short/untimed games.
        marker = None if len(rows) >= 2 and all(a.timestamp_unix > 0 for a in rows) else group
        result[digest({"rows": sorted(content, key=digest), "unidentified_group": marker})] = len(rows)
    return result


def _remove_snapshot(path: Path, store: LearningStore) -> None:
    resolved = path.resolve()
    if resolved.is_relative_to((store.root / "snapshots").resolve()) and resolved != (store.root / "snapshots").resolve():
        shutil.rmtree(resolved, ignore_errors=True)


def cleanup_uncommitted_snapshots(store: LearningStore) -> None:
    # Called only while owning cycle.lock: a crash during freezing may leave a
    # directory which has never become a queued job. Preserve every recorded job.
    referenced = {job["snapshot"] for path in (store.root / "identities").glob("*.json")
                  for job in read_json(path).get("jobs", [])}
    for path in (store.root / "snapshots").glob("*"):
        if path.is_dir() and path.name not in referenced:
            _remove_snapshot(path, store)


def run_cycle(root: Path, request: dict) -> dict:
    """One trainer owns a cycle. Live inputs are never used after snapshot freeze."""
    from .cards import CardCatalog
    from .replay_learning import (ReplayPolicyRegistry, audit_replay_learning,
                                  collect_replay_learning_actions, train_replay_policy)
    from .training_sync import exclusive, training_allowed, SyncBusyError
    root = root.resolve()
    store = LearningStore(root)
    active_trial = read_json(store.root / "trials/ledger.json").get("active")
    if active_trial and active_trial.get("status") == "battle_trial":
        return store.status("waiting_trial", "固定对照实测进行中，暂缓产生下一候选")
    if not training_allowed(root):
        return store.status("collector", "本机仅采集；候选由指定训练机生成")
    if read_json(store.root / "control.json").get("paused"):
        return store.status("paused", "自主学习已暂停")
    if request["code_hash"] != code_identity(root):
        raise ValueError("代码已改变，请使用新版本重新审计")
    settings = request["settings"]
    os.environ["CRBOT_COMPUTE_DEVICE"] = request["training"].get("device", "auto")
    identity = learning_identity(request)
    with exclusive(store.root / "cycle.lock"):
        cleanup_uncommitted_snapshots(store)
        state = store.load(identity)
        store.status("auditing", "核对当前引擎与冻结数据", identity=identity,
                     target=request["replay"].get("training_policy_version"))
        pending = next((j for j in state["jobs"] if j["phase"] in {"queued", "training"}), None)
        registry = ReplayPolicyRegistry(root)
        if pending:
            # Crash after registry commit but before ledger commit: reuse the existing candidate.
            existing = next((c for c in registry.load().get("candidates", [])
                             if c.get("manifest", {}).get("self_learning", {}).get("job_id") == pending["id"]), None)
            if existing:
                model = (registry.root / existing.get("model_path", "")).resolve()
                if (not model.is_relative_to(registry.root.resolve()) or not model.is_file() or
                        hashlib.sha256(model.read_bytes()).hexdigest() != existing.get("model_sha256")):
                    pending.update(phase="failed", error="中断恢复时候选文件校验失败")
                    store.save(state)
                    return store.status("failed", pending["error"], identity=identity)
                return _finish_job(store, state, pending, existing)
        if not pending:
            snapshot = store.root / "snapshots" / uuid.uuid4().hex
            try:
                freeze_snapshot(root, snapshot, request)
                catalog = CardCatalog.load(snapshot / "data/cards.json")
                audit = audit_replay_learning(snapshot, catalog, request["replay"])
                groups = content_groups(collect_replay_learning_actions(snapshot, catalog, request["replay"]))
                fresh = set(groups) - set(state["consumed"])
                details = {"identity": identity, "target": request["replay"].get("training_policy_version"),
                           "new_battles": len(fresh), "new_actions": sum(groups[k] for k in fresh),
                           "required_new_battles": settings["minimum_new_battles"],
                           "required_new_actions": settings["minimum_new_actions"],
                           "audit": audit.to_dict()}
                if not audit.ready or len(fresh) < settings["minimum_new_battles"] or details["new_actions"] < settings["minimum_new_actions"]:
                    return store.status("waiting_data", "等待同版本新增有效对局/动作或最低覆盖", **details)
                if time.time() - state["last_training_at"] < settings["minimum_interval_s"]:
                    return store.status("cooldown", "尚未达到训练间隔", **details)
                parent = registry.champion() or {}
                job_id = digest({"identity": identity, "groups": sorted(groups), "parent": parent.get("model_sha256", parent.get("version")),
                                 "recipe": "existing_replay_v1", "seed": request["replay"].get("training_seed")})
                if any(j["id"] == job_id for j in state["jobs"]):
                    return store.status("waiting_data", "该快照已尝试，等待新数据", **details)
                pending = {"id": job_id, "phase": "queued", "snapshot": snapshot.name,
                           "groups": sorted(groups), "attempts": 0, "request": request,
                           "parent_version": parent.get("version"), "created_at": time.time()}
                state["jobs"].append(pending)
                store.save(state)
            finally:
                if pending is None or pending.get("snapshot") != snapshot.name:
                    _remove_snapshot(snapshot, store)
        snapshot = store.root / "snapshots" / pending["snapshot"]
        if not snapshot.resolve().is_relative_to((store.root / "snapshots").resolve()):
            raise ValueError("无效快照路径")
        if pending["attempts"] >= settings["max_attempts"]:
            pending["phase"] = "failed"
            store.save(state)
            return store.status("failed", "该快照已达到重试预算；等待新数据", identity=identity)
        try:
            verify_snapshot(snapshot)
        except (OSError, ValueError) as exc:
            pending.update(phase="failed", error=str(exc))
            store.save(state)
            raise
        pending["attempts"] += 1
        pending["phase"] = "training"
        state["last_training_at"] = time.time()
        store.save(state)
        store.status("training", "局间生成候选；现役模型保持不变", identity=identity, job_id=pending["id"])
        try:
            result = train_replay_policy(root, CardCatalog.load(snapshot / "data/cards.json"),
                pending["request"]["replay"], candidate_only=True, snapshot_root=snapshot,
                learning_manifest={"job_id": pending["id"], "identity": identity,
                    "parent_version": pending["parent_version"], "snapshot_hash": digest(read_json(snapshot / "snapshot.json")),
                    "code_hash": request["code_hash"], "effective_config": {k: request[k] for k in ("policy", "prediction", "replay")},
                    "label_source": "verified_outcomes_and_confirmed_actions; local_feedback_is_proxy"})
        except SyncBusyError:
            pending["attempts"] -= 1
            pending["phase"] = "queued"
            store.save(state)
            return store.status("deferred", "训练锁正在使用，下次局间重试", identity=identity)
        except Exception as exc:
            pending["phase"] = "failed"
            pending["error"] = str(exc)
            store.save(state)
            raise
        return _finish_job(store, state, pending, result["candidate"])


def _finish_job(store, state, job, candidate):
    job.update(phase="completed", candidate_version=candidate["version"],
               quality_passed=bool(candidate.get("quality_passed")), completed_at=time.time())
    state["consumed"] = sorted(set(state["consumed"]) | set(job["groups"]))
    store.save(state)
    retained = job["request"]["settings"]["retained_snapshots"]
    for old in [j for j in state["jobs"] if j["phase"] in {"completed", "failed"}][:-retained]:
        _remove_snapshot(store.root / "snapshots" / old["snapshot"], store)
    return store.status("candidate_ready", "候选已保存，未实战验收、未切换冠军", identity=state["identity"],
                        candidate_version=candidate["version"], quality_passed=bool(candidate.get("quality_passed")),
                        rejection_reasons=candidate.get("rejection_reasons", []), job_id=job["id"])


class SelfLearningService:
    """Parent supervisor: only called at verified offline lobby boundaries."""
    def __init__(self, root: Path, config: dict, stop_requested=lambda: False):
        self.root, self.config = root.resolve(), config
        self.settings = learning_settings(config)
        self.stop_requested = stop_requested
        self.store = LearningStore(self.root)
        self.completed = 0
        self.due = True  # First lobby after restart audits/reconciles interrupted jobs.
        self.phase = "waiting_boundary"

    def on_battle_completed(self, episode: dict) -> None:
        if episode.get("reward_verified"):
            self.completed += 1
            self.due |= self.completed % self.settings["audit_every_battles"] == 0

    def boundary(self) -> bool:
        """Return True if the caller must discard its stale frame and reobserve."""
        if not self.settings["enabled"] or not self.due or self.stop_requested():
            if not self.settings["enabled"]:
                self.phase = "disabled"
                self.store.status(self.phase, "配置未启用自主学习")
            return False
        self.due = False
        if read_json(self.store.root / "control.json").get("paused"):
            self.phase = "paused"
            return False
        from .training_sync import training_allowed
        if not training_allowed(self.root):
            self.phase = "collector"
            self.store.status(self.phase, "本机仅采集；候选由指定训练机生成")
            return False
        request = request_config(self.root, self.config)
        request_path = self.store.root / "requests" / (uuid.uuid4().hex + ".json")
        atomic_json(request_path, request)
        log_path = request_path.with_suffix(".log")
        start = time.monotonic()
        process = None
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "crbot.self_learning", "--root", str(self.root),
                    "--request", str(request_path)], cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                last = None
                while process.poll() is None:
                    paused = read_json(self.store.root / "control.json").get("paused")
                    timed_out = time.monotonic() - start >= self.settings["max_cycle_seconds"]
                    if self.stop_requested() or paused or timed_out:
                        process.terminate()
                        process.wait(timeout=10)
                        self.phase = "paused" if not timed_out else "budget_exhausted"
                        self.store.status(self.phase, "任务已中断；下次启动按快照恢复，冠军未切换")
                        break
                    status = self.store.current_status()
                    self.phase = status.get("phase", "auditing")
                    if status.get("updated_at") != last:
                        print(f"[自主学习] {self.phase}：{status.get('reason', '')}", flush=True)
                        last = status.get("updated_at")
                    time.sleep(0.25)
                if process.returncode == 0:
                    status = self.store.current_status()
                    self.phase = status.get("phase", "idle")
                    print(f"[自主学习] {self.phase}：{status.get('reason', '')}", flush=True)
                elif self.phase not in {"paused", "budget_exhausted"}:
                    self.phase = "failed"
                    prior = self.store.current_status()
                    reason = prior.get("reason") if prior.get("phase") == "failed" else "子进程异常结束，保留现役"
                    self.store.status("failed", reason, diagnostic_log=str(log_path))
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            request_path.unlink(missing_ok=True)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_cycle(args.root, read_json(args.request))
    except Exception as exc:
        from .training_sync import SyncBusyError
        if isinstance(exc, SyncBusyError):
            LearningStore(args.root).status("deferred", "已有自主任务运行，下次局间重试")
            return
        LearningStore(args.root).status("failed", str(exc))
        raise


if __name__ == "__main__":
    main()
