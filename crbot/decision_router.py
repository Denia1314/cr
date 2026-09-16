"""Opt-in planner integration retaining the original policy and confirmation protocol."""
from __future__ import annotations

import time
import math
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
        from .tower_observation import TowerHealthTracker
        self.tower_observer = TowerHealthTracker()
        self.planner = None
        self.knowledge_error = ""
        self.last_plan = None
        self.last_execution_engine = "none"
        self.fallback_count = 0
        self._proposed_prediction = None
        from .grid_world import GridWorldAudit
        self.grid_audit = GridWorldAudit()
        self.grid_frame = None
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
        self.tower_observer.reset()
        self.last_plan = None
        self.last_execution_engine = "none"
        self._proposed_prediction = None
        from .grid_world import GridWorldAudit
        self.grid_audit = GridWorldAudit()
        self.grid_frame = None

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
        decision_started = time.perf_counter()
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
            if threat.score < float(self.policy.get("enemy_pressure_threshold", .17)):
                continue
            layers = set(threat.unit_layers) if threat.layer_confidence >= float(self.policy.get("threat_layer_min_confidence", .70)) else set()
            if layers == {"air"}:
                hypotheses = ("minions", "baby_dragon", "balloon")
            elif threat.threat == "swarm":
                hypotheses = ("skeletons", "goblins", "barbarians")
            elif threat.threat == "heavy":
                hypotheses = ("giant", "valkyrie", "pekka")
            else:
                hypotheses = ("knight", "musketeer", "mini_pekka")
            if layers != {"ground"} and layers != {"air"}:
                # Missing layer evidence is not evidence of ground troops.
                hypotheses = (hypotheses[0], "minions", "balloon")
            centers = threat.centers or ((.28 if lane == "left" else .72, max(.25, threat.proximity)),)
            unmatched = list(centers)
            for detection in known:
                if unmatched:
                    index = min(range(len(unmatched)), key=lambda i: math.hypot(unmatched[i][0]-detection["x"], unmatched[i][1]-detection["y"]))
                    if math.hypot(unmatched[index][0]-detection["x"], unmatched[index][1]-detection["y"]) <= .09:
                        unmatched.pop(index)
            for x, y in unmatched[:8]:
                detections.append({"card_id": f"unknown:{lane}:{threat.threat}", "side": -1,
                                   "x": x, "y": y, "confidence": .5, "hypotheses": hypotheses})
            if unmatched:
                uncertain.append("enemy_identity_unknown:" + lane)
                if not layers or layers == {"ground", "air"}:
                    uncertain.append("enemy_layer_unknown:" + lane)
        bars = []
        if self.planning_config.get('unit_health_observation', True):
            from .unit_health import detect_unit_health_bars
            from .unit_badges import find_level_badges
            enemy_badges=[b for b in find_level_badges(current) if b['side']==-1]
            for detection in detections:
                if detection['side']!=-1 or not detection['card_id'].startswith('unknown:'):continue
                nearby=[b for b in enemy_badges if math.hypot(b['x']-detection['x'],b['y']-detection['y'])<.025]
                if len(nearby)==1:
                    detection.update(level=nearby[0]['level'],side_evidence='level_badge')
            bars = detect_unit_health_bars(current, self.planner.sim.geometry.tower_points)
            claimed = set()
            for bar in bars:
                candidates = [(math.hypot(d['x']-bar['x'],d['y']-bar['y']),i)
                              for i,d in enumerate(detections) if i not in claimed and d['side']==bar['side']]
                candidates.sort()
                if candidates and candidates[0][0] <= .065 and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>.015):
                    index=candidates[0][1];claimed.add(index)
                    if detections[index].get('hp_fraction') is None:
                        detections[index].update({k:v for k,v in bar.items() if k.startswith('hp_')})
                elif not candidates or candidates[0][0] > .065:
                    if bar['side']==-1 and not any(abs(b['x']-bar['x'])<.055 and abs(b['y']-bar['y'])<.065 for b in enemy_badges):
                        continue
                    lane='left' if bar['x']<.5 else 'right'
                    hypotheses=('knight','minions','balloon') if bar['side']==-1 else ()
                    detections.append(dict(bar,card_id=f"unknown:{lane}:health",confidence=.8,hypotheses=hypotheses))
                    claimed.add(len(detections)-1)
                    uncertain.append(('enemy' if bar['side']==-1 else 'ally')+'_identity_unknown:'+lane)
        matches = (self.hand_history.matches_for_decision(self.last_hand_matches, now)
                   if self.temporal_perception_enabled else self.last_hand_matches)
        hand = [(m.slot_index, m.card_id) for m in matches
                if m.card_id and m.confidence >= float(self.policy.get("hand_min_confidence", .4))]
        costs = {cid: c["elixir"] for cid, c in self.planner.kb.cards.items() if c.get("elixir") is not None}
        elixir = float(self.last_elixir_estimate_value or 0)
        tower_health=self.tower_observer.observe(current,self.planning_config.get("tower_bar_rois",[]),
            self.planner.sim.geometry.tower_points,now=now)
        snapshot = self.world.update(detections, now=now, elapsed=now-self.battle_started_at,
                                     elixir=elixir, hand=hand, costs=costs,
                                     seconds_per_elixir=float(self.policy.get("seconds_per_elixir", 2.8)) / self._elixir_rate_multiplier,
                                     uncertain=uncertain,tower_health=tower_health)
        if self.pending_action_id is not None or now < self.next_action_at:
            return None
        if not hand or self.last_hand_image is not current:
            return self._fallback(current, previous, now, "hand_unavailable")
        if self.planning_config.get("fast_defense", False) and not self.planning_config.get("unified_tactics",False) and not any(
            t.side == -1 and t.hp_fraction != 0 and now - t.last_seen <= 1.5 for t in snapshot.tracks
        ):
            return self._fallback(current, previous, now, "no_active_enemy_use_normal_play")
        frame_age = max(0, float(getattr(self, "prediction_frame_age_s", 0)))
        snapshot=replace(snapshot,observation_delay_s=frame_age+time.perf_counter()-decision_started)
        scorer, learning = None, {"available": False, "reason": "disabled"}
        if self.planning_config.get("learned_action_value", True):
            learning["reason"] = "model_unavailable"
            if self.replay_model is not None:
                scorer, learning = self.replay_model.action_scorer(
                    self.catalog, elixir, threats, current, now - self.battle_started_at)
        result = self.planner.plan(snapshot, action_scorer=scorer, learning_status=learning)
        result.valid_until -= frame_age
        self.last_plan = result.to_dict()
        self.last_plan['visible_health_bars'] = bars
        if self.planning_config.get('grid_world', True):
            try:
                state=self.planner.initial(snapshot,snapshot.enemy_elixir_estimate,1.)
                packet=self.grid_audit.update(snapshot,state,self.planner.sim,result)
                packet['tower_bar_rois']=self.planning_config.get('tower_bar_rois',[])
                packet['battle_started_at']=self.battle_started_at
                self.last_plan['grid_world']=packet
                self.grid_frame=(current,packet)
            except (ValueError,KeyError,TypeError) as exc:
                self.last_plan['grid_world_error']=str(exc)
        self.last_plan['unit_health'] = [dict(track_id=t.track_id,card_id=t.card_id,side=t.side,
            fraction=t.hp_fraction,observed_at=t.hp_observed_at,confidence=t.hp_confidence,
            identity_known=not t.card_id.startswith('unknown:')) for t in snapshot.tracks]
        self.last_plan["frame_age_before_perception_s"] = round(frame_age, 4)
        self.last_plan["perception_mode"] = "exact_cards" if getattr(detector, "last_detection_succeeded", False) and not any(x.startswith("enemy_identity_unknown") for x in uncertain) else "lane_hypotheses"
        if self.decision_engine == "shadow":
            self.last_execution_engine = "legacy_shadow"
            decision = super().decide(current, previous, now=now)
            return replace(decision, decision_engine="legacy_shadow") if decision is not None else None
        if max(result.elapsed_ms / 1000, time.perf_counter() - decision_started) > result.valid_until - result.observed_at:
            self.last_plan["fallback_reason"] = "plan_expired_recapture"
            self.last_execution_engine = "none"
            return None
        if result.status == "timeout":
            self.last_execution_engine = "none"
            return None
        if result.status not in {"ready", "wait"}:
            return self._fallback(current, previous, now, result.reason)
        self.last_execution_engine = "predictive"
        if result.status == "wait":
            # WAIT never consumes hand/elixir or changes action-confirmation state.
            self.next_action_at = now + min(float(self.planning_config.get("wait_recheck_s",.1)), max(.05, result.action.wait_s))
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
        self.last_plan["learning"]["selected_for_execution"] = bool(result.learning.get("applied"))
        return BattleDecision(action.slot, list(self.vision["card_slot_centers"][action.slot]), deploy,
                              lane, "推演：" + result.reason, elixir, self.last_elixir_estimate_source, 0, 0,
                              card_id=card.card_id, card_name=card.name_zh or card.name_en, card_cost=card.elixir,
                              hand=tuple(m.card_id for m in matches),
                              enemy_cards=strongest.enemy_cards, threat_type=strongest.threat,
                              threat_proximity=strongest.proximity, left_threat=threats["left"].score,
                              right_threat=threats["right"].score, battle_elapsed_s=now-self.battle_started_at,
                              left_threat_type=threats["left"].threat, right_threat_type=threats["right"].threat,
                              left_threat_proximity=threats["left"].proximity, right_threat_proximity=threats["right"].proximity,
                              left_threat_approach_rate=threats["left"].approach_rate, right_threat_approach_rate=threats["right"].approach_rate,
                              left_unit_count=threats["left"].unit_count, right_unit_count=threats["right"].unit_count,
                              replay_learning_used=bool(result.learning.get("applied")),
                              learned_action_value=dict(self.last_plan["learning"]),
                              action_intent=('preparation' if result.compute.get('elixir_overflow',{}).get('preparation') else 'overflow_development')
                                  if result.compute.get('elixir_overflow',{}).get('changed') else 'reactive',
                              decision_engine="predictive", plan_revision=snapshot.revision,
                              plan_valid_until=result.valid_until, knowledge_version=self.planner.kb.version,
                              hand_confidence=self.hand_history.metadata(now).get(str(action.slot), {}).get("confidence", .5),
                              elixir_confidence=self.last_elixir_confidence)

    def resolve_action(self, action_id, status, *, now=None):
        changed = super().resolve_action(action_id, status, now=now)
        if changed and self._proposed_prediction is not None:
            if self.last_plan is not None:
                self.last_plan.setdefault("learning", {})["action_confirmation"] = status
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
