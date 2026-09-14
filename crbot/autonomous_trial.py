"""SL3: pinned future-battle comparison. Passing creates a deployment proposal only."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time
import uuid

from .learning_store import atomic_json, digest, read_json
from .replay_learning import ReplayPolicyModel, ReplayPolicyRegistry
from .self_learning import code_identity
from .training_sync import ReplaySync, complete_rows, exclusive, training_allowed

PROTOCOL = "paired_blocks_fixed_budget_hoeffding_v1"
DEFAULTS = dict(enabled=True, battles_per_arm=1000, batch_size=5,
                minimum_gain=.03, family_alpha=.05, health_battles=10,
                maximum_unknown_rate=.10, maximum_unconfirmed_rate=.15,
                maximum_confirmation_drop=.05, maximum_plan_ms=2500,
                maximum_fallback_rate=.10, minimum_learning_rate=.01)


def trial_settings(config):
    settings = {**DEFAULTS, **config.get("self_learning_trial", {})}
    if not isinstance(settings["enabled"], bool):
        raise ValueError("self_learning_trial.enabled 必须为布尔值")
    for key, value in settings.items():
        if key == "enabled":
            continue
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"self_learning_trial.{key} 无效")
    for key in ("battles_per_arm", "batch_size", "health_battles"):
        if int(settings[key]) != settings[key]:
            raise ValueError(f"self_learning_trial.{key} 必须为整数")
        settings[key] = int(settings[key])
    if settings["battles_per_arm"] % settings["batch_size"]:
        raise ValueError("实测局数必须是批次大小的倍数")
    if settings["battles_per_arm"] < settings["health_battles"]:
        raise ValueError("实测预算小于健康观察预算")
    if any(settings[k] >= 1 for k in ("minimum_gain", "family_alpha", "maximum_unknown_rate",
            "maximum_unconfirmed_rate", "maximum_confirmation_drop", "maximum_fallback_rate", "minimum_learning_rate")):
        raise ValueError("实测比例必须小于 1")
    return settings


def judge(rows, protocol, alpha):
    """One efficacy look only. Unknown outcomes cannot improve the challenger."""
    arms = {a: [r for r in rows if r["arm"] == a] for a in ("baseline", "candidate")}
    metrics = {}
    for arm, group in arms.items():
        n = len(group)
        actions = sum(r["actions"] for r in group)
        confirmed = sum(r["confirmed"] for r in group)
        decisions = sum(r["decisions"] for r in group)
        metrics[arm] = dict(battles=n, unknown_rate=sum(not r["verified"] for r in group) / max(1, n),
            confirmation_rate=confirmed / max(1, actions), actions=actions,
            fallback_rate=sum(r["fallbacks"] for r in group) / max(1, decisions),
            max_plan_ms=max((r["max_plan_ms"] for r in group), default=0),
            learning_rate=sum(r["learned"] for r in group) / max(1, decisions))
        if n >= protocol["health_battles"]:
            m = metrics[arm]
            if (m["unknown_rate"] > protocol["maximum_unknown_rate"] or not actions or not decisions
                    or 1 - m["confirmation_rate"] > protocol["maximum_unconfirmed_rate"]
                    or m["fallback_rate"] > protocol["maximum_fallback_rate"]
                    or m["max_plan_ms"] > protocol["maximum_plan_ms"]):
                return "rejected", dict(reason=arm + "_health_guardrail", arms=metrics)
            if arm == "candidate" and m["learning_rate"] < protocol["minimum_learning_rate"]:
                return "rejected", dict(reason="candidate_not_participating", arms=metrics)
    b, c = metrics["baseline"], metrics["candidate"]
    if min(b["battles"], c["battles"]) >= protocol["health_battles"]:
        if b["confirmation_rate"] - c["confirmation_rate"] > protocol["maximum_confirmation_drop"]:
            return "rejected", dict(reason="confirmation_regression", arms=metrics)
    if min(b["battles"], c["battles"]) < protocol["battles_per_arm"]:
        return "battle_trial", dict(reason="fixed_budget_pending", arms=metrics)
    size = protocol["batch_size"]
    def utility(row, arm):
        if not row["verified"]:
            return 1. if arm == "baseline" else 0.
        return {"win": 1., "draw": .5, "loss": 0.}[row["outcome"]]
    differences = []
    for start in range(0, protocol["battles_per_arm"], size):
        differences.append(sum(utility(r, "candidate") for r in arms["candidate"][start:start+size]) / size
                           - sum(utility(r, "baseline") for r in arms["baseline"][start:start+size]) / size)
    gain = sum(differences) / len(differences)
    radius = math.sqrt(2 * math.log(1 / alpha) / len(differences))
    passed = gain - radius > protocol["minimum_gain"]
    return ("battle_pass" if passed else "inconclusive"), dict(
        reason="fixed_budget_complete", arms=metrics, gain=gain, lower_gain=gain-radius,
        alpha=alpha, independent_block_pairs=len(differences), deployment_pending=passed,
        independence_assumption="paired time blocks are independent; within-block correlation allowed")


class AutonomousTrial:
    def __init__(self, root, config, policy):
        self.root, self.config, self.policy = Path(root), config, policy
        self.settings = trial_settings(config)
        self.enabled = bool(config.get("self_learning_trial", {}).get("enabled", False))
        self.trial_enabled = self.enabled
        self.enabled |= bool(config.get("self_learning_deployment", {}).get("enabled", False))
        self.directory = self.root / "training/self_learning/trials"
        self.path = self.directory / "ledger.json"
        self.lock = None
        self.training_lock = None
        self.ledger = None
        self.models = {}
        self.original_model = policy.replay_model
        self.phase = "idle"
        self.pending = None
        self.counters = {}
        from .deployment import DeploymentService
        self.deployment = DeploymentService(self)

    @property
    def active(self):
        return bool(self.ledger and self.ledger.get("active") and
                    self.ledger["active"]["status"] == "battle_trial")

    def _save(self):
        atomic_json(self.path, self.ledger)
        trial = self.ledger.get("active")
        if trial and trial["status"] == "battle_pass":
            self._proposal(trial)

    def _proposal(self, trial):
        atomic_json(self.directory / "proposals" / (digest(trial["id"]) + ".json"), dict(
            trial_id=trial["id"], state="battle_pass", deployed=False,
            candidate=trial["candidate"], baseline=trial["baseline"], identity=trial["identity"],
            protocol=trial["protocol"], report=trial["report"], evidence_sha256=digest(trial["rows"])))

    def _identity(self):
        paths = [self.root / self.config.get("prediction", {}).get("knowledge_path", "data/battle_knowledge.json"),
                 self.root / self.config.get("dataset", {}).get("card_catalog", "data/cards.json")]
        # Other loaded learning components must stay fixed across restarts too.
        for name in ("imitation_model", "learned_detector"):
            model = getattr(self.policy, name, None)
            registry = getattr(model, "registry", None)
            if registry is not None:
                for method in ("champion_path", "champion_model_path"):
                    if hasattr(registry, method):
                        path = getattr(registry, method)()
                        if path is not None:
                            paths.append(Path(path))
        return digest(dict(code=code_identity(self.root), config=self.config,
                           loaded_components={name: getattr(getattr(self.policy, name, None), "champion", None)
                                              for name in ("imitation_model", "learned_detector")},
                           resources=[hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None for p in paths]))

    def _load(self, entry):
        if entry is None:
            return None
        if entry.get("sync_compatibility") != ReplaySync(self.root).compatibility(self.config.get("replay", {})):
            raise ValueError("trial_model_incompatible")
        model = ReplayPolicyModel(self.root, self.config.get("replay", {}), entry=entry)
        if not model.available:
            raise ValueError(model.load_error or "trial model unavailable")
        return model

    def _finish(self, status, reason):
        trial = self.ledger["active"]
        trial.update(status=status, reason=reason, finished_at=time.time())
        trial.pop("pending", None)
        self.pending = None
        self._save()
        self.phase = status

    def _restore(self):
        self.ledger = read_json(self.path, dict(schema=1, attempts=0, history=[], active=None,
                                               family_alpha=self.settings["family_alpha"]))
        trial = self.ledger.get("active")
        if not trial or trial["status"] != "battle_trial":
            return
        pending = trial.get("pending")
        if pending and pending.get("started"):
            matches = [r for path in (self.root / "runs").glob("*/replay_episodes.jsonl") for r in complete_rows(path)
                       if r.get("policy", {}).get("sl3_trial_id") == trial["id"]
                       and r.get("policy", {}).get("sl3_index") == pending["index"]]
            matches = list({digest(r): r for r in matches}.values())
            if len(matches) != 1:
                self._finish("interrupted", "unfinished_or_ambiguous_battle_after_restart")
            else:
                self.pending = pending
                self.observe(matches[0])
        elif pending:
            trial.pop("pending")
            self._save()

    def boundary(self):
        """Only call at a verified offline lobby. True requires frame recapture."""
        if not self.enabled:
            return False
        if self.lock is None:
            self.lock = exclusive(self.directory / "owner.lock")
            try:
                self.lock.__enter__()
            except Exception:
                self.lock = None
                raise
            self._restore()
        paused = read_json(self.root / "training/self_learning/control.json").get("paused", False)
        allowed = (not paused
                   and self.config.get("policy", {}).get("decision_engine") == "predictive"
                   and self.config.get("prediction", {}).get("learned_action_value", True)
                   and self.config.get("replay", {}).get("allow_bot_training", False))
        if not allowed:
            if not self.active and self.deployment.boundary(allow_new=False):
                return True
            self.phase = "paused_or_not_eligible"
            changed = self.policy.replay_model is not self.original_model
            self.policy.replay_model = self.original_model
            return changed
        if self.active and not training_allowed(self.root):
            self._finish("interrupted", "trainer_role_changed")
        registry = ReplayPolicyRegistry(self.root).load()
        trial = self.ledger.get("active")
        if trial and trial["status"] != "battle_trial":
            if trial["status"] == "battle_pass":
                self._proposal(trial)  # Repair a crash between verdict and proposal writes.
            if self.training_lock is not None:
                self.training_lock.__exit__(None, None, None)
                self.training_lock = None
            self.ledger["history"].append(trial)
            self.ledger["active"] = None
            self._save()
            changed = self.policy.replay_model is not self.original_model
            self.policy.replay_model = self.original_model
            if changed:
                return True
            trial = None
        if trial is None:
            if self.deployment.boundary():
                self.phase = self.deployment.state.get("state", "deployment")
                return True
            if self.deployment.blocks_trial or not self.trial_enabled or not training_allowed(self.root):
                self.phase = "deployment_monitoring" if self.deployment.blocks_trial else "collector"
                return False
            registry = ReplayPolicyRegistry(self.root).load()
        if trial is None:
            attempted = {t["candidate"]["model_sha256"] for t in self.ledger["history"]}
            candidates = [c for c in registry.get("candidates", []) if c.get("quality_passed") is True
                          and c.get("manifest", {}).get("action_value", {}).get("enabled") is True
                          and c.get("model_sha256") not in attempted and c != registry.get("champion")]
            if not candidates:
                self.phase = "waiting_for_qualified_candidate"
                return False
            if self.training_lock is None:
                self.training_lock = exclusive(self.root / ".training-sync/training.lock")
                try:
                    self.training_lock.__enter__()
                except Exception:
                    self.training_lock = None
                    raise
            candidate = max(candidates, key=lambda c: c.get("created_at_unix", 0))
            baseline = deepcopy(getattr(self.original_model, "champion", None)) if getattr(self.original_model, "available", False) else None
            number = self.ledger["attempts"] + 1
            trial = dict(id=uuid.uuid4().hex, status="battle_trial", candidate=deepcopy(candidate), baseline=baseline,
                         registry_champion=digest(registry.get("champion")), identity=self._identity(),
                         protocol={**self.settings, "version": PROTOCOL},
                         alpha=min(self.ledger["family_alpha"], self.settings["family_alpha"]) / (number * (number + 1)),
                         started_at=time.time(), rows=[])
            trial["first_arms"] = ["baseline" if int(digest([trial["id"], i]), 16) % 2 else "candidate"
                                   for i in range(self.settings["battles_per_arm"] // self.settings["batch_size"])]
            self.ledger.update(active=trial, attempts=number)
            self._save()
        if self.training_lock is None:
            self.training_lock = exclusive(self.root / ".training-sync/training.lock")
            try:
                self.training_lock.__enter__()
            except Exception:
                self.training_lock = None
                raise
        try:
            if trial["identity"] != self._identity() or trial["registry_champion"] != digest(registry.get("champion")):
                raise ValueError("runtime_or_champion_changed")
            for arm in ("baseline", "candidate"):
                entry = trial[arm]
                if entry is not None:
                    path = (ReplayPolicyRegistry(self.root).root / entry["model_path"]).resolve()
                    if not path.is_relative_to(ReplayPolicyRegistry(self.root).root) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["model_sha256"]:
                        raise ValueError("pinned_model_changed")
                key = (trial["id"], arm)
                if key not in self.models:
                    self.models[key] = self._load(entry)
            candidate_model = self.models[(trial["id"], "candidate")]
            if not candidate_model.action_value_head or not candidate_model.action_value_head.enabled or candidate_model.influence_scale <= 0:
                raise ValueError("candidate_head_not_active")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._finish("rejected", str(exc))
            self.policy.replay_model = self.original_model
            return True
        if trial.get("pending"):
            self.phase = "battle_trial"
            self.pending = trial["pending"]
            # Resume after a pause using exactly the previously reserved arm.
            model = self.models[(trial["id"], self.pending["arm"])]
            changed = self.policy.replay_model is not model
            self.policy.replay_model = model
            return changed
        index = len(trial["rows"])
        batch = index // trial["protocol"]["batch_size"]
        first = trial["first_arms"][batch // 2]
        arm = first if batch % 2 == 0 else ("candidate" if first == "baseline" else "baseline")
        self.pending = dict(index=index, arm=arm, started=False)
        trial["pending"] = self.pending
        self._save()  # Persist assignment before exposing its model to execution.
        self.policy.replay_model = self.models[(trial["id"], arm)]
        self.phase = "battle_trial"
        return True

    def start_battle(self):
        if not self.active or self.pending is None or self.phase == "paused_or_not_eligible":
            return {"sl3_trial_id": None, "sl3_evaluation_only": False, **self.deployment.start_battle()}
        if self.policy.replay_model is not self.models.get((self.ledger["active"]["id"], self.pending["arm"])):
            self._finish("rejected", "runtime_model_changed")
            return {"sl3_trial_id": None, "sl3_evaluation_only": False}
        self.pending["started"] = True
        self._save()
        self.counters = dict(decisions=0, fallbacks=0, learned=0, max_plan_ms=0.)
        self.last_decision = None
        trial, arm = self.ledger["active"], self.pending["arm"]
        entry = trial[arm] or {}
        return dict(sl3_trial_id=trial["id"], sl3_index=self.pending["index"], sl3_arm=arm,
                    sl4_deployment_id=None,
                    sl3_model_sha256=entry.get("model_sha256"), sl3_model_version=entry.get("version"),
                    replay_policy_version=entry.get("version"), replay_policy_status=entry.get("status"),
                    sl3_protocol=PROTOCOL, sl3_evaluation_only=True,
                    runtime_replay_version=entry.get("version"), runtime_replay_loaded=bool(entry))

    def decision(self, prediction):
        if not self.active or not self.pending or not self.pending.get("started"):
            self.deployment.decision(prediction)
            return
        plan = prediction.get("plan") or {}
        revision = prediction.get("world", {}).get("revision")
        if revision == getattr(self, "last_decision", None):
            return
        self.last_decision = revision
        if plan.get("revision") != revision:
            # Cooldown observations reuse an earlier plan; they are not decisions.
            if prediction.get("actual_engine") == "legacy_fallback":
                self.counters["decisions"] += 1
                self.counters["fallbacks"] += 1
            return
        self.counters["decisions"] += 1
        self.counters["fallbacks"] += prediction.get("actual_engine") != "predictive"
        self.counters["learned"] += bool(plan.get("learning", {}).get("applied")) and prediction.get("actual_engine") == "predictive"
        self.counters["max_plan_ms"] = max(self.counters["max_plan_ms"], float(plan.get("elapsed_ms", 0)))

    def observe(self, episode):
        if episode.get("policy", {}).get("sl4_deployment_id"):
            self.deployment.observe(episode)
            return
        if not self.active or not self.pending:
            return
        trial, pending = self.ledger["active"], self.pending
        if not pending.get("started"):
            return
        policy = episode.get("policy", {})
        expected = trial[pending["arm"]] or {}
        if (policy.get("sl3_trial_id") != trial["id"] or policy.get("sl3_index") != pending["index"]
                or policy.get("sl3_arm") != pending["arm"] or policy.get("sl3_model_sha256") != expected.get("model_sha256")
                or policy.get("sl3_model_version") != expected.get("version") or policy.get("sl3_protocol") != PROTOCOL
                or not policy.get("sl3_evaluation_only") or episode.get("timestamp_unix", 0) < trial["started_at"]):
            self._finish("rejected", "assignment_contamination")
            return
        stats = episode.get("sl3_runtime", {})
        required = ("decisions", "fallbacks", "learned", "max_plan_ms")
        if not all(k in stats and math.isfinite(float(stats[k])) and stats[k] >= 0 for k in required):
            self._finish("rejected", "missing_runtime_evidence")
            return
        if (not 0 <= int(episode.get("confirmed_action_count", 0)) <= int(episode.get("action_count", 0))
                or stats["fallbacks"] > stats["decisions"] or stats["learned"] > stats["decisions"]
                or any(r["episode_id"] == episode.get("episode_id") for r in trial["rows"])):
            self._finish("rejected", "invalid_or_duplicate_evidence")
            return
        trial["rows"].append(dict(arm=pending["arm"], episode_id=episode["episode_id"],
            outcome=episode.get("outcome", "unknown"), verified=bool(episode.get("reward_verified")) and episode.get("outcome") in {"win", "loss", "draw"},
            actions=int(episode.get("action_count", 0)), confirmed=int(episode.get("confirmed_action_count", 0)), **stats))
        trial.pop("pending", None)
        self.pending = None
        status, report = judge(trial["rows"], trial["protocol"], trial["alpha"])
        trial.update(status=status, report=report)
        self.phase = status
        self._save()

    def status(self):
        ledger = self.ledger or {}
        trial = ledger.get("active") or (ledger.get("history") or [{}])[-1]
        return dict(phase=self.phase, trial_id=trial.get("id"), completed=len(trial.get("rows", [])),
                    trial_status=trial.get("status"), report=trial.get("report"), deployment=self.deployment.status())

    def close(self):
        self.policy.replay_model = self.original_model
        if self.training_lock is not None:
            self.training_lock.__exit__(None, None, None)
            self.training_lock = None
        if self.lock is not None:
            self.lock.__exit__(None, None, None)
            self.lock = None
