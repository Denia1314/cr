"""SL4: battle-boundary deployment with one atomic registry commit and a rollback anchor."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import time
import uuid

from .learning_store import atomic_json, digest, read_json
from .replay_learning import ReplayPolicyRegistry
from .training_sync import ReplaySync, SyncBusyError, complete_rows, exclusive, training_allowed, MAX_MODEL_BYTES

DEFAULTS = dict(enabled=True, probation_battles=20, minimum_health_battles=10,
                maximum_unknown_rate=.10, maximum_unconfirmed_rate=.15,
                maximum_confirmation_drop=.05, maximum_fallback_rate=.10,
                maximum_plan_ms=2500., minimum_learning_rate=.01, maximum_utility_drop=.20)


def model_id(entry):
    return None if entry is None else (entry.get("model_sha256"), entry.get("version"), entry.get("influence_scale"))


class DeploymentService:
    def __init__(self, trial):
        self.host, self.root, self.policy = trial, trial.root, trial.policy
        self.settings = {**DEFAULTS, **trial.config.get("self_learning_deployment", {})}
        self.enabled = bool(trial.config.get("self_learning_deployment", {}).get("enabled", False))
        for key, value in self.settings.items():
            if key == "enabled":
                if not isinstance(value, bool):
                    raise ValueError("部署开关必须为布尔值")
            elif not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"无效部署参数：{key}")
        for key in ("probation_battles", "minimum_health_battles"):
            if int(self.settings[key]) != self.settings[key]:
                raise ValueError("部署观察局数必须为整数")
            self.settings[key] = int(self.settings[key])
        if self.settings["minimum_health_battles"] > self.settings["probation_battles"]:
            raise ValueError("部署观察预算小于健康检查门槛")
        for key in ("maximum_unknown_rate", "maximum_unconfirmed_rate", "maximum_confirmation_drop",
                    "maximum_fallback_rate", "minimum_learning_rate", "maximum_utility_drop"):
            if self.settings[key] >= 1:
                raise ValueError("部署比例参数必须小于 1")
        self.registry = ReplayPolicyRegistry(self.root)
        self.lock_path = self.root / ".training-sync/model-registry.lock"
        self.loaded_id = None
        self.state = {}
        self.last_revision = None
        self.counters = {}
        self.publication = {}
        self.remote_error = None

    def contract(self):
        from .self_learning import code_identity
        config = self.host.config
        replay = {k: v for k, v in config.get("replay", {}).items() if k != "allow_bot_training"}
        components = {}
        for name in ("imitation_model", "learned_detector"):
            model = getattr(self.policy, name, None)
            champion = getattr(model, "champion", None)
            components[name] = (champion or {}).get("model_sha256") or champion
        return digest(dict(code=code_identity(self.root), policy=config.get("policy", {}),
            prediction=config.get("prediction", {}), replay=replay, components=components,
            catalog=read_json(self.root / config.get("dataset", {}).get("card_catalog", "data/cards.json")),
            knowledge=read_json(self.root / config.get("prediction", {}).get("knowledge_path", "data/battle_knowledge.json"))))

    @property
    def monitoring(self):
        return self.blocks_trial or (self.state.get("state") == "stable" and self.state.get("entry") is not None)

    @property
    def blocks_trial(self):
        return self.state.get("state") in {"probation", "rollback_pending"}

    def _write(self, registry):
        atomic_json(self.registry.path, registry)
        self.state = deepcopy(registry.get("deployment") or {})

    def _recover_commit(self, registry, source):
        canonical = self.registry.load()
        if (canonical.get("deployment") or {}).get("source") != source:
            return False
        registry.clear()
        registry.update(canonical)
        self.loaded_id = None
        return True

    def _load(self, entry):
        if entry is None:
            return None
        path = (self.registry.root / entry["model_path"]).resolve()
        if not path.is_relative_to(self.registry.root) or path.stat().st_size > MAX_MODEL_BYTES:
            raise ValueError("部署模型路径或容量无效")
        model = self.host._load(entry)
        if model.loaded_model_sha256 != entry["model_sha256"]:
            raise ValueError("实际加载哈希不一致")
        return model

    def _activate(self, model, deployment):
        if model is not None and model.loaded_model_sha256 != deployment["entry"]["model_sha256"]:
            raise ValueError("实际加载内容与部署记录不一致")
        self.policy.replay_model = self.host.original_model = model
        self.loaded_id = deployment["id"]

    def _proof(self, proposal, registry):
        from .autonomous_trial import judge
        ledger = self.host.ledger or {}
        trial = next((t for t in [ledger.get("active"), *ledger.get("history", [])]
                      if t and t.get("id") == proposal.get("trial_id")), None)
        if trial is None or trial.get("status") != "battle_pass":
            raise ValueError("找不到实战通过的原始账本")
        rows = trial["rows"]
        if len(rows) != 2 * trial["protocol"]["battles_per_arm"] or len({r["episode_id"] for r in rows}) != len(rows):
            raise ValueError("实战证据数量或唯一性无效")
        for i, row in enumerate(rows):
            batch = i // trial["protocol"]["batch_size"]
            first = trial["first_arms"][batch // 2]
            expected = first if batch % 2 == 0 else ("candidate" if first == "baseline" else "baseline")
            if row["arm"] != expected:
                raise ValueError("实战分组与预定顺序不符")
        phase, report = judge(rows, trial["protocol"], trial["alpha"])
        if (phase != "battle_pass" or digest(rows) != proposal.get("evidence_sha256")
                or digest(report) != digest(proposal.get("report"))
                or digest(trial["candidate"]) != digest(proposal.get("candidate"))
                or proposal.get("identity") != trial["identity"] or trial["identity"] != self.host._identity()
                or trial["registry_champion"] != digest(registry.get("champion"))
                or model_id(trial["baseline"]) != model_id(getattr(self.host.original_model, "champion", None)
                    if getattr(self.host.original_model, "available", False) else None)):
            raise ValueError("部署证据、运行身份或现役锚点已改变")
        return trial

    def _commit(self, registry, entry, *, source, baseline, report, compatibility, remote=None):
        if compatibility != ReplaySync(self.root).compatibility(self.host.config.get("replay", {})):
            raise ValueError("部署通知与当前模式不兼容")
        if remote and remote.get("runtime_contract") != self.contract():
            raise ValueError("远端部署的策略、知识或辅助权重与本机不一致")
        model = self._load(entry)
        if entry is not None and not (remote and remote.get("state") == "rolled_back") and (not model.action_value_head or not model.action_value_head.enabled
                                  or not math.isfinite(model.influence_scale) or model.influence_scale <= 0):
            raise ValueError("部署候选的学习评分未启用")
        old = deepcopy(registry.get("deployment"))
        generation = int((old or {}).get("generation", 0)) + 1
        identifier = uuid.uuid4().hex
        promoted = None if entry is None else {**entry, "status": "champion", "promoted": True,
                    "deployment_generation": generation, "deployment_id": identifier}
        state = "rolled_back" if remote and remote.get("state") == "rolled_back" else "probation" if entry else "stable"
        deployment = dict(id=identifier, generation=generation, state=state,
            source=source, entry=promoted, fallback=deepcopy(baseline), created_at=time.time(),
            previous=old, report=report, compatibility=compatibility, protocol=deepcopy(self.settings),
            rows=[], total_battles=0, identity=self.host._identity(), runtime_contract=self.contract(), remote=remote, loaded_sha256=None)
        registry["champion"] = promoted
        registry["deployment"] = deployment
        if promoted is not None:
            registry.setdefault("candidates", []).append(deepcopy(promoted))
            model.champion = deepcopy(promoted)
        if remote:
            registry["remote_consumed"] = dict(trainer=remote["trainer_device"], generation=remote["generation"])
            registry.pop("remote_deployment_error", None)
            self.remote_error = None
        self._write(registry)  # Crash after this point is recovered by loading the committed entry.
        try:
            self._activate(model, deployment)
        except Exception:
            deployment.update(state="rollback_pending", reason="memory_activation_failed")
            self._write(registry)
            self._rollback(registry)
            raise
        deployment["loaded_sha256"] = getattr(model, "loaded_model_sha256", None)
        self._write(registry)

    def _rollback(self, registry):
        failed = registry["deployment"]
        fallback = failed.get("fallback")
        try:
            model = self._load(fallback)
        except (OSError, ValueError, KeyError):
            # A damaged anchor cannot be called restored. Use the unweighted P1 fallback.
            model, fallback = None, None
        generation = int(failed["generation"]) + 1
        identifier = uuid.uuid4().hex
        entry = None if fallback is None else {**fallback, "status": "champion", "promoted": True,
                    "deployment_generation": generation, "deployment_id": identifier}
        rollback = dict(id=identifier, generation=generation, state="rolled_back", entry=entry,
            fallback=None, source=failed.get("source"), previous=failed, rows=[], protocol=failed["protocol"],
            reason=failed.get("reason", "deployment_guardrail"), compatibility=failed["compatibility"],
            created_at=time.time(), identity=self.host._identity(), runtime_contract=self.contract(),
            loaded_sha256=getattr(model, "loaded_model_sha256", None))
        registry.update(champion=entry, deployment=rollback)
        for candidate in registry.get("candidates", []):
            if candidate.get("deployment_id") == failed["id"]:
                candidate.update(status="rejected", promoted=False, deployment_health_passed=False,
                                 rollback_reason=rollback["reason"])
        if entry:
            registry.setdefault("candidates", []).append(entry)
            model.champion = deepcopy(entry)
        self._write(registry)
        self._activate(model, rollback)

    def _recover_pending(self, registry):
        dep = registry["deployment"]
        pending = dep.get("pending")
        if not pending:
            return
        matches = [r for path in (self.root / "runs").glob("*/replay_episodes.jsonl") for r in complete_rows(path)
                   if r.get("policy", {}).get("sl4_deployment_id") == dep["id"]
                   and r.get("policy", {}).get("sl4_index") == pending]
        matches = list({digest(r): r for r in matches}.values())
        if len(matches) != 1:
            dep.update(state="rollback_pending", reason="interrupted_probation_battle")
        else:
            self._observe(registry, matches[0])

    def boundary(self, *, allow_new=True):
        try:
            return self._boundary(allow_new=allow_new)
        except SyncBusyError:
            # No start click may race a registry writer; retry on a fresh lobby frame.
            return True

    def _boundary(self, *, allow_new=True):
        if not self.enabled:
            return False
        with exclusive(self.root / ".training-sync/training.lock"), exclusive(self.lock_path):
            registry = self.registry.load()
            self.publication = registry.get("deployment_publication", {})
            self.remote_error = registry.get("remote_deployment_error")
            dep = registry.get("deployment")
            remote = registry.get("remote_deployment")
            consumed = registry.get("remote_consumed", {})
            new_remote = (not training_allowed(self.root) and remote and
                          (consumed.get("trainer") != remote["trainer_device"] or remote["generation"] > consumed.get("generation", 0)))
            if dep:
                self.state = deepcopy(dep)
                if not dep.get("source", "").startswith("remote:"):
                    handled = registry.setdefault("deployment_proposals", {})
                    outcome = "rolled_back" if dep["state"] == "rolled_back" else "deployed"
                    if handled.get(dep["source"]) != outcome:
                        handled[dep["source"]] = outcome
                        self._write(registry)
                if model_id(registry.get("champion")) != model_id(dep.get("entry")):
                    raise ValueError("冠军被外部流程改变，停止自动部署")
                if dep.get("pending"):
                    self._recover_pending(registry)
                    self._write(registry)
                if self.loaded_id != dep["id"]:
                    try:
                        if dep.get("identity") != self.host._identity():
                            raise ValueError("部署后运行身份改变")
                        model = self._load(dep.get("entry"))
                        self._activate(model, dep)
                    except (OSError, ValueError, KeyError):
                        dep.update(state="rollback_pending", reason="deployment_load_or_identity_failed")
                    if dep["state"] == "rollback_pending":
                        self._rollback(registry)
                    else:
                        dep["loaded_sha256"] = getattr(self.policy.replay_model, "loaded_model_sha256", None)
                        self._write(registry)
                    return True
                if dep["state"] == "rollback_pending":
                    self._rollback(registry)
                    return True
                try:
                    if dep.get("identity") != self.host._identity():
                        raise ValueError("部署运行身份改变")
                    if dep.get("entry"):
                        path = (self.registry.root / dep["entry"]["model_path"]).resolve()
                        if not path.is_relative_to(self.registry.root) or hashlib.sha256(path.read_bytes()).hexdigest() != dep["entry"]["model_sha256"]:
                            raise ValueError("部署文件内容改变")
                except (OSError, ValueError, KeyError):
                    dep.update(state="rollback_pending", reason="deployed_file_corrupt")
                    self._rollback(registry)
                    return True
                # Check the live arrays' recorded identity, not just the registry label.
                if (model_id(getattr(self.policy.replay_model, "champion", None)) != model_id(dep.get("entry"))
                        or getattr(self.policy.replay_model, "loaded_model_sha256", None) != dep.get("loaded_sha256")):
                    dep.update(state="rollback_pending", reason="runtime_loaded_identity_changed")
                    self._rollback(registry)
                    return True
                if dep["state"] == "probation" and not (new_remote and allow_new):
                    self.state = deepcopy(dep)
                    return False
            if not allow_new:
                return False
            # Collector notifications are staged by sync, never activated in its background thread.
            if new_remote:
                source = "remote:" + remote["deployment_id"]
                try:
                    self._commit(registry, remote.get("entry"), source=source,
                        baseline=dep.get("fallback") if dep and dep["state"] == "probation" else
                            getattr(self.host.original_model, "champion", None) if getattr(self.host.original_model, "available", False) else None,
                        report=remote.get("report", {}), compatibility=remote["compatibility"], remote=remote)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    committed = self._recover_commit(registry, source)
                    registry["remote_deployment_error"] = {"deployment_id": remote["deployment_id"], "reason": str(exc)}
                    self.remote_error = registry["remote_deployment_error"]
                    self._write(registry)
                    return committed
                return True
            if training_allowed(self.root):
                handled = registry.setdefault("deployment_proposals", {})
                for path in sorted((self.host.directory / "proposals").glob("*.json")):
                    proposal = read_json(path)
                    identifier = proposal.get("trial_id") or path.name
                    if identifier in handled:
                        continue
                    try:
                        trial = self._proof(proposal, registry)
                        baseline_rows = [r for r in trial["rows"] if r["arm"] == "baseline"]
                        report = {**trial["report"], "baseline_utility": sum(
                            {"win": 1., "loss": 0., "draw": .5}.get(r["outcome"], 1.) for r in baseline_rows) / len(baseline_rows)}
                        self._commit(registry, trial["candidate"], source=identifier, baseline=trial["baseline"],
                                     report=report, compatibility=trial["candidate"]["sync_compatibility"])
                        handled[identifier] = "deployed"
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        committed = self._recover_commit(registry, identifier)
                        handled = registry.setdefault("deployment_proposals", {})
                        handled[identifier] = "committed_recovery_pending" if committed else "rejected: " + str(exc)
                    self._write(registry)
                    return True
            self.state = deepcopy(registry.get("deployment") or {})
            return False

    def start_battle(self):
        model = self.policy.replay_model
        entry = getattr(model, "champion", None) or {}
        metadata = dict(replay_policy_version=entry.get("version"), replay_policy_status=entry.get("status"), runtime_replay_version=entry.get("version"),
            runtime_replay_loaded=bool(model and model.available), sl4_deployment_id=None,
            runtime_replay_influence_scale=float(getattr(model, "influence_scale", 0.)))
        if not self.enabled or not self.monitoring:
            return metadata
        with exclusive(self.lock_path):
            registry = self.registry.load()
            dep = registry["deployment"]
            if dep["id"] != self.loaded_id or dep["state"] not in {"probation", "stable"} or dep.get("pending"):
                raise ValueError("部署观察状态尚未恢复，禁止开始新局")
            index = dep.get("total_battles", 0) + 1
            dep["pending"] = index
            self._write(registry)
        self.counters = self.host.counters = dict(decisions=0, fallbacks=0, learned=0, max_plan_ms=0.)
        self.last_revision = None
        return {**metadata, "sl4_deployment_id": dep["id"], "sl4_index": index,
                "sl4_model_sha256": entry.get("model_sha256")}

    def decision(self, prediction):
        if not self.monitoring or not self.state.get("pending"):
            return
        revision = prediction.get("world", {}).get("revision")
        if revision == self.last_revision:
            return
        self.last_revision = revision
        plan = prediction.get("plan") or {}
        if plan.get("revision") != revision and prediction.get("actual_engine") != "legacy_fallback":
            return
        self.counters["decisions"] += 1
        self.counters["fallbacks"] += prediction.get("actual_engine") != "predictive"
        self.counters["learned"] += bool(plan.get("learning", {}).get("applied")) and prediction.get("actual_engine") == "predictive"
        self.counters["max_plan_ms"] = max(self.counters["max_plan_ms"], float(plan.get("elapsed_ms", 0)))

    def _observe(self, registry, episode):
        dep = registry["deployment"]
        policy, stats = episode.get("policy", {}), episode.get("sl4_runtime", {})
        if (policy.get("sl4_deployment_id") != dep["id"] or policy.get("sl4_index") != dep.get("pending")
                or policy.get("sl4_model_sha256") != (dep.get("entry") or {}).get("model_sha256")
                or not all(k in stats and math.isfinite(float(stats[k])) and stats[k] >= 0
                           for k in ("decisions", "fallbacks", "learned", "max_plan_ms"))
                or not 0 <= int(episode.get("confirmed_action_count", 0)) <= int(episode.get("action_count", 0))
                or stats.get("fallbacks", 0) > stats.get("decisions", 0) or stats.get("learned", 0) > stats.get("decisions", 0)
                or episode.get("timestamp_unix", 0) < dep["created_at"]
                or any(r["episode_id"] == episode.get("episode_id") for r in dep["rows"])):
            dep.update(state="rollback_pending", reason="probation_evidence_missing_or_mismatched")
            return
        dep["total_battles"] = dep.pop("pending")
        verified = bool(episode.get("reward_verified")) and episode.get("outcome") in {"win", "loss", "draw"}
        dep["rows"].append(dict(episode_id=episode["episode_id"], verified=verified,
            utility={"win": 1., "loss": 0., "draw": .5}.get(episode.get("outcome"), 0.) if verified else 0.,
            actions=int(episode.get("action_count", 0)), confirmed=int(episode.get("confirmed_action_count", 0)), **stats))
        cfg = dep["protocol"]
        dep["rows"] = dep["rows"][-cfg["probation_battles"]:]
        rows = dep["rows"]
        if len(rows) < cfg["minimum_health_battles"]:
            return
        actions = sum(r["actions"] for r in rows)
        decisions = sum(r["decisions"] for r in rows)
        confirmation = sum(r["confirmed"] for r in rows) / max(1, actions)
        baseline = dep.get("report", {}).get("arms", {}).get("baseline", {})
        bad = (not actions or not decisions or sum(not r["verified"] for r in rows) / len(rows) > cfg["maximum_unknown_rate"]
               or 1-confirmation > cfg["maximum_unconfirmed_rate"]
               or baseline.get("confirmation_rate", confirmation)-confirmation > cfg["maximum_confirmation_drop"]
               or sum(r["fallbacks"] for r in rows) / max(1, decisions) > cfg["maximum_fallback_rate"]
               or sum(r["learned"] for r in rows) / max(1, decisions) < cfg["minimum_learning_rate"]
               or max(r["max_plan_ms"] for r in rows) > cfg["maximum_plan_ms"]
               or sum(r["utility"] for r in rows) / len(rows) < dep.get("report", {}).get("baseline_utility", 0.) - cfg["maximum_utility_drop"])
        if bad:
            dep.update(state="rollback_pending", reason="probation_health_guardrail")
        elif dep["state"] == "probation" and len(rows) >= cfg["probation_battles"]:
            dep.update(state="stable", stable_at=time.time())

    def observe(self, episode):
        if not self.enabled or not self.monitoring or not self.state.get("pending"):
            return
        try:
            with exclusive(self.lock_path):
                registry = self.registry.load()
                self._observe(registry, episode)
                self._write(registry)
        except SyncBusyError:
            # The recorder has already durably written this result. Recover it at the lobby.
            return

    def status(self):
        return dict(enabled=self.enabled, state=self.state.get("state", "waiting_proposal"),
                    reason=self.state.get("reason"), remote_error=self.remote_error,
                    generation=self.state.get("generation"), model=model_id(self.state.get("entry")),
                    loaded_sha256=self.state.get("loaded_sha256"), completed=self.state.get("total_battles", 0),
                    publication="uploaded" if self.publication.get("generation") == self.state.get("generation")
                        and self.publication.get("uploaded") else "waiting_stable" if self.blocks_trial else "pending")
