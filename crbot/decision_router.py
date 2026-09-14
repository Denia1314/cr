"""Opt-in planner integration retaining the original policy and confirmation protocol."""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from .battle_world import BattleWorld
from .knowledge import KnowledgeBase
from .policy import BattleDecision, BattlePolicy
from .predictive_planner import PredictivePlanner


class PredictiveBattlePolicy(BattlePolicy):
    def __init__(self, config, config_path=None):
        super().__init__(config, config_path)
        self.decision_engine = config.get("policy", {}).get("decision_engine", "legacy")
        self.planning_config = dict(config.get("prediction", {}))
        self.world = BattleWorld()
        self.planner = None
        self.knowledge_error = ""
        self.last_plan = None
        self.last_execution_engine = "none"
        self.fallback_count = 0
        self._proposed_prediction = None
        root = config_path.parent if config_path else Path.cwd()
        path = root / self.planning_config.get("knowledge_path", "data/battle_knowledge.json")
        try:
            kb = KnowledgeBase.load(path)
            self.planner = PredictivePlanner(kb, {**self.policy, **self.planning_config})
            self.knowledge_audit = kb.audit(int(self.planning_config.get("assumed_level", 11)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.knowledge_error = str(exc)
            self.knowledge_audit = {}

    def reset_battle(self, now=None):
        super().reset_battle(now)
        self.world.reset()
        self.last_plan = None
        self.last_execution_engine = "none"
        self._proposed_prediction = None

    def prediction_status(self):
        return {"selected_engine": self.decision_engine, "actual_engine": self.last_execution_engine,
                "planner_available": self.planner is not None, "load_error": self.knowledge_error,
                "fallback_count": self.fallback_count, "knowledge": self.knowledge_audit,
                "world": self.world.status(), "plan": self.last_plan,
                "battle_acceptance": False}

    def _fallback(self, current, previous, now, reason):
        self.last_execution_engine = "legacy_fallback"
        self.fallback_count += 1
        self.last_plan = {**(self.last_plan or {}), "fallback_reason": reason,
                          "actual_engine": "legacy_fallback"}
        decision = super().decide(current, previous, now=now)
        return replace(decision, decision_engine="legacy_fallback") if decision is not None else None

    def decide(self, current, previous, *, now=None):
        now = time.monotonic() if now is None else now
        if self.pending_action_id is not None or now < self.next_action_at:
            return None
        if self.planner is None or not self.reactive_ready:
            return self._fallback(current, previous, now, self.knowledge_error or "perception_unavailable")
        # Capture observations before either strategy mutates action state.
        already_observed = (self._observed_image is current and self._observed_at == now
                            and self.last_hand_image is current and self._last_hand_observed_at == now)
        if not already_observed:
            self.observe_replay_state(current, previous, now=now)
        threats = self._perceive_threats(current, previous, now)
        detector = self.learned_detector
        detections = []
        for side, attr in ((-1, "observed_enemies"), (1, "observed_allies")):
            for d in getattr(detector, attr, []):
                detections.append({**d, "side": side})
        uncertain = ["enemy_deployment_events_unconfirmed", "cycle_unknown"]
        for lane, threat in threats.items():
            known = [d for d in detections if d["side"] == -1 and ("left" if d["x"] < .5 else "right") == lane]
            if threat.score < float(self.policy.get("enemy_pressure_threshold", .17)) or known:
                continue
            air = "air" in threat.unit_layers
            if air:
                hypotheses = ("minions", "baby_dragon", "balloon")
            elif threat.threat == "swarm":
                hypotheses = ("skeletons", "goblins", "barbarians")
            elif threat.threat == "heavy":
                hypotheses = ("giant", "valkyrie", "pekka")
            else:
                hypotheses = ("knight", "musketeer", "mini_pekka")
            centers = threat.centers or ((.28 if lane == "left" else .72, max(.25, threat.proximity)),)
            for x, y in centers[:8]:
                detections.append({"card_id": f"unknown:{lane}:{threat.threat}", "side": -1,
                                   "x": x, "y": y, "confidence": .5, "hypotheses": hypotheses})
            uncertain.append("enemy_identity_unknown:" + lane)
        matches = (self.hand_history.matches_for_decision(self.last_hand_matches, now)
                   if self.temporal_perception_enabled else self.last_hand_matches)
        hand = [(m.slot_index, m.card_id) for m in matches
                if m.card_id and m.confidence >= float(self.policy.get("hand_min_confidence", .4))]
        costs = {cid: c["elixir"] for cid, c in self.planner.kb.cards.items() if c.get("elixir") is not None}
        elixir = float(self.last_elixir_estimate_value or 0)
        snapshot = self.world.update(detections, now=now, elapsed=now-self.battle_started_at,
                                     elixir=elixir, hand=hand, costs=costs,
                                     seconds_per_elixir=float(self.policy.get("seconds_per_elixir", 2.8)) / self._elixir_rate_multiplier,
                                     uncertain=uncertain)
        if not hand or self.last_hand_image is not current:
            return self._fallback(current, previous, now, "hand_unavailable")
        result = self.planner.plan(snapshot)
        self.last_plan = result.to_dict()
        self.last_plan["perception_mode"] = "exact_cards" if getattr(detector, "last_detection_succeeded", False) and not any(x.startswith("enemy_identity_unknown") for x in uncertain) else "lane_hypotheses"
        if self.decision_engine == "shadow":
            self.last_execution_engine = "legacy_shadow"
            decision = super().decide(current, previous, now=now)
            return replace(decision, decision_engine="legacy_shadow") if decision is not None else None
        if result.status not in {"ready", "wait"}:
            return self._fallback(current, previous, now, result.reason)
        if result.elapsed_ms / 1000 > result.valid_until - result.observed_at:
            return self._fallback(current, previous, now, "plan_expired")
        self.last_execution_engine = "predictive"
        if result.status == "wait":
            # WAIT never consumes hand/elixir or changes action-confirmation state.
            self.next_action_at = now + min(.4, max(.1, result.action.wait_s))
            return None
        action = result.action
        if (action.slot, action.card_id) not in snapshot.hand:
            return self._fallback(current, previous, now, "hand_changed")
        card = self.catalog.get(action.card_id)
        if card is None or card.elixir is None or card.elixir > elixir:
            return self._fallback(current, previous, now, "cost_or_card_invalid")
        deploy = [action.x, action.y]
        lane = "left" if action.x < .5 else "right"
        strongest = threats[lane]
        self.virtual_elixir = max(0, elixir - card.elixir)
        self.last_deploy_point, self.last_deploy_at = tuple(deploy), now
        self.action_sequence += 1
        self.next_action_at = now + float(self.planning_config.get("action_cooldown_s", .65))
        self._remember_formation(lane, card, deploy, now, "predictive")
        self._proposed_prediction = (card.card_id, action.x, action.y)
        return BattleDecision(action.slot, list(self.vision["card_slot_centers"][action.slot]), deploy,
                              lane, "推演：" + result.reason, elixir, self.last_elixir_estimate_source, 0, 0,
                              card_id=card.card_id, card_name=card.name_zh or card.name_en, card_cost=card.elixir,
                              hand=tuple(m.card_id for m in matches),
                              enemy_cards=strongest.enemy_cards, threat_type=strongest.threat,
                              threat_proximity=strongest.proximity, left_threat=threats["left"].score,
                              right_threat=threats["right"].score, battle_elapsed_s=now-self.battle_started_at,
                              decision_engine="predictive", plan_revision=snapshot.revision,
                              plan_valid_until=result.valid_until, knowledge_version=self.planner.kb.version,
                              hand_confidence=self.hand_history.metadata(now).get(str(action.slot), {}).get("confidence", .5),
                              elixir_confidence=self.last_elixir_confidence)

    def resolve_action(self, action_id, status, *, now=None):
        changed = super().resolve_action(action_id, status, now=now)
        if changed and self._proposed_prediction is not None:
            if status == "confirmed":
                cid, x, y = self._proposed_prediction
                self.world.confirm(action_id, cid, x, y, time.monotonic() if now is None else now)
            self._proposed_prediction = None
        return changed


def create_policy(config, config_path=None):
    mode = config.get("policy", {}).get("decision_engine", "legacy")
    if mode not in {"legacy", "shadow", "predictive"}:
        raise ValueError(f"未知决策引擎：{mode}")
    return BattlePolicy(config, config_path) if mode == "legacy" else PredictiveBattlePolicy(config, config_path)
