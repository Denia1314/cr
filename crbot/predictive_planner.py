"""Receding-horizon search over friendly actions and uncertain enemy responses."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .battle_simulation import SimAction, Simulator, SimState
from .battle_world import WorldSnapshot
from .knowledge import KnowledgeBase
from .placement_search import placement_points


@dataclass
class PlanResult:
    revision: int
    observed_at: float
    valid_until: float
    status: str
    action: SimAction = field(default_factory=SimAction)
    candidates: list[dict] = field(default_factory=list)
    reason: str = ""
    elapsed_ms: float = 0
    knowledge_version: str = ""
    simulation_version: str = "spatial_v2_placement"
    nodes: int = 0
    completed_depth: int = 0
    budget_exhausted: bool = False
    placement_mode: str = "scene_grid_and_intercepts"

    def to_dict(self):
        return asdict(self)


class PredictivePlanner:
    def __init__(self, kb: KnowledgeBase, config: dict):
        self.kb, self.config = kb, config
        self.sim = Simulator(kb, int(config.get("assumed_level", 11)))
        self.clock = time.perf_counter

    def candidates(self, s: SimState, side: int, *, limit=12) -> list[SimAction]:
        result = [SimAction(wait_s=.4)]
        for slot, cid in s.hands[side]:
            card = self.kb.cards.get(cid, {})
            cost = card.get("elixir")
            if cost is None or cost > s.elixir[side]:
                continue
            if not self.kb.roster(cid, self.sim.level) and not self.kb.spell(cid, self.sim.level):
                continue
            points = placement_points(self.sim, s, cid, side)
            for x, y in points:
                result.append(SimAction(cid, slot, x, y))
        # Round-robin by card avoids exhausting the budget on the first hand slot.
        ordered = [result[0]]
        groups = {}
        for a in result[1:]:
            groups.setdefault(a.card_id, []).append(a)
        while groups and len(ordered) < limit:
            for cid in list(groups):
                ordered.append(groups[cid].pop(0))
                if not groups[cid]:
                    del groups[cid]
                if len(ordered) >= limit:
                    break
        return ordered

    def initial(self, world: WorldSnapshot, enemy_elixir: float, health: float) -> SimState:
        s = SimState(elixir={1: world.elixir, -1: enemy_elixir}, hands={1: list(world.hand), -1: []})
        s.seconds_per_elixir = float(self.config.get("seconds_per_elixir", 2.8))
        if world.elapsed >= float(self.config.get("double_elixir_after_s", 120)):
            s.seconds_per_elixir /= float(self.config.get("double_elixir_multiplier", 2))
        s.uncertainties.update(world.uncertainty)
        s.uncertainties.add("historical_balance_and_spatial_approximation")
        tower = self.kb.unit("PrincessTower", self.sim.level)
        if tower is None:
            raise ValueError("缺少塔参数")
        i = 0
        for side, y in ((1, .77), (-1, .23)):
            for x in (.28, .72):
                px, py = self.sim.xy(x, y)
                fraction = world.tower_health[i]
                if fraction is None or fraction > 0:
                    self.sim.add(s, tower, side, px, py, tower=True,
                                 hp_fraction=1 if fraction is None else fraction)
                i += 1
        for t in world.tracks:
            if world.at - t.last_seen > 1.5:
                s.uncertainties.add("occluded_entity")
                continue
            cid = t.card_id
            if t.hypotheses:
                index = 0 if health < .75 else (1 if health < .95 else 2)
                cid = t.hypotheses[min(index, len(t.hypotheses)-1)]
                s.uncertainties.add("enemy_identity_hypothesis:" + cid)
            roster = self.kb.roster(cid, self.sim.level)
            if not roster:
                s.uncertainties.add("unknown_unit:" + t.card_id)
                continue
            spec = roster[0][0]  # one detected box represents one entity, not another full deployment
            if sum(n for _, n in roster) != 1:
                s.uncertainties.add("group_identity:" + t.card_id)
            x, y = self.sim.xy(t.x, t.y)
            fraction = t.hp_fraction if t.hp_fraction is not None else (health if t.side == -1 else .7)
            card = self.kb.cards[cid]
            self.sim.add(s, spec, t.side, x, y, hp_fraction=fraction,
                         value=float(card.get("elixir") or 0) / max(1, sum(n for _, n in roster)))
            if t.variant != "base":
                s.uncertainties.add("variant_unknown")
        for event in world.confirmed_placements:
            # No duplicate units when a detector has supplied allies of this card.
            if any(t.side == 1 and t.card_id == event["card_id"] for t in world.tracks):
                continue
            age = max(0, world.at - event["at"])
            roster = self.kb.roster(event["card_id"], self.sim.level)
            total = sum(n for _, n in roster)
            for spec, count in roster:
                x, y = self.sim.xy(event["x"], event["y"])
                y = max(16., y - spec.speed * max(0, age - spec.deploy))
                # This is a declining survival hypothesis, never a claim of detection.
                fraction = max(.15, .8 - age * .08)
                for i in range(min(20, count)):
                    self.sim.add(s, spec, 1, x + (i % 3) * .3, y, hp_fraction=fraction,
                                 value=float(self.kb.cards[event["card_id"]].get("elixir") or 0) / max(1, total))
            if roster:
                s.uncertainties.add("unobserved_ally_survival_hypothesis")
        seen = list(world.enemy_seen)
        # Unseen responses are explicit hypotheses, never claims about the enemy deck.
        prior = list(self.config.get("unknown_response_cards", ["knight", "musketeer", "fireball"]))
        pool = list(dict.fromkeys(seen + prior))[:12]
        s.hands[-1] = [(i, cid) for i, cid in enumerate(pool) if cid in self.kb.cards]
        s.uncertainties.add("enemy_hand_hypotheses_not_observed")
        return s

    def plan(self, world: WorldSnapshot) -> PlanResult:
        started = self.clock()
        deadline = started + max(.01, min(2., float(self.config.get("budget_ms", 180)) / 1000))
        result = PlanResult(world.revision, world.at, world.at + float(self.config.get("max_plan_age_s", 1.0)),
                            "unavailable", knowledge_version=self.kb.version)
        try:
            initial = self.initial(world, world.enemy_elixir[1], 1.)
            roots = self.candidates(initial, 1, limit=int(self.config.get("candidate_limit", 9)))
            horizon = max(4, min(15, float(self.config.get("horizon_s", 8))))
            # Same scenarios and same horizon for every root, with conservative lower-tail weighting.
            scenarios = [(world.enemy_elixir[0], .65, False),
                         (max(world.enemy_elixir[0], min(world.enemy_elixir[1], world.enemy_elixir_estimate)), .85, True),
                         (world.enemy_elixir[1], 1., True)]
            coarse_rows = []
            for root in roots:
                outcomes, branches = [], []
                for enemy_cost, hp, responds in scenarios:
                    s = self.initial(world, enemy_cost, hp)
                    if not self.sim.apply(s, root, 1):
                        continue
                    self.sim.advance(s, 1.5, deadline=deadline, clock=self.clock)
                    response = self.prior_response(s) if responds else SimAction()
                    self.sim.apply(s, response, -1)
                    self.sim.advance(s, horizon - 1.5, deadline=deadline, clock=self.clock)
                    score, parts = self.sim.evaluate(s)
                    result.nodes += 1
                    outcomes.append(score)
                    branches.append({"enemy_elixir_assumption": enemy_cost, "enemy_response": response.label,
                                     "own_followup": "WAIT", "score": round(score, 4), **parts})
                if len(outcomes) == len(scenarios):
                    risk = float(self.config.get("downside_weight", .65))
                    score = (1-risk) * sum(outcomes) / len(outcomes) + risk * min(outcomes)
                    coarse_rows.append({"action": asdict(root), "label": root.label,
                                        "score": round(score, 4), "branches": branches})
            # Publish only a complete, equally evaluated layer. Refinements remain private until complete.
            result.candidates = coarse_rows
            result.completed_depth = 1
            refined_rows = []
            for root in roots:
                outcomes, branches = [], []
                for enemy_cost, hp, responds in scenarios:
                    s = self.initial(world, enemy_cost, hp)
                    if not self.sim.apply(s, root, 1):
                        continue
                    self.sim.advance(s, 1.5, deadline=deadline, clock=self.clock)
                    response = SimAction()
                    if responds:
                        # Enemy chooses its best short-horizon response, including holding resources.
                        best_enemy = float("inf")
                        for enemy in self.candidates(s, -1, limit=5):
                            trial = s.clone()
                            if not self.sim.apply(trial, enemy, -1):
                                continue
                            self.sim.advance(trial, 2, deadline=deadline, clock=self.clock)
                            value, _ = self.sim.evaluate(trial)
                            result.nodes += 1
                            if value < best_enemy:
                                best_enemy, response = value, enemy
                    self.sim.apply(s, response, -1)
                    self.sim.advance(s, 2, deadline=deadline, clock=self.clock)
                    # Own second move is conditional on the response, not an unconditional macro.
                    best, best_parts, follow = float("-inf"), {}, SimAction()
                    for action in self.candidates(s, 1, limit=5):
                        trial = s.clone()
                        if not self.sim.apply(trial, action, 1):
                            continue
                        self.sim.advance(trial, horizon - 3.5, deadline=deadline, clock=self.clock)
                        score, parts = self.sim.evaluate(trial)
                        result.nodes += 1
                        if score > best:
                            best, best_parts, follow = score, parts, action
                    outcomes.append(best)
                    branches.append({"enemy_elixir_assumption": enemy_cost, "enemy_response": response.label,
                                     "own_followup": follow.label, "score": round(best, 4), **best_parts})
                if len(outcomes) != len(scenarios):
                    continue
                risk = float(self.config.get("downside_weight", .65))
                score = (1-risk) * sum(outcomes) / len(outcomes) + risk * min(outcomes)
                refined_rows.append({"action": asdict(root), "label": root.label,
                                     "score": round(score, 4), "branches": branches})
            result.candidates = refined_rows
            result.completed_depth = 2
        except TimeoutError:
            result.budget_exhausted = True
            if not result.candidates:
                result.status, result.reason = "timeout", "incomplete_search_fallback"
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            result.status, result.reason = "unavailable", f"world_model_error: {exc}"
            result.candidates = []
        if result.candidates:
            result.candidates.sort(key=lambda c: c["score"], reverse=True)
            best = result.candidates[0]
            result.action = SimAction(**best["action"])
            result.status = "wait" if result.action.card_id is None else "ready"
            responses = "; ".join(f"敌方 {b['enemy_response']} → 我方 {b['own_followup']}" for b in best["branches"])
            result.reason = f"完整比较 {len(result.candidates)} 个方案 / 深度 {result.completed_depth} / {horizon:g} 秒；{responses}"
        result.elapsed_ms = round((self.clock() - started) * 1000, 2)
        return result

    def prior_response(self, s):
        """Cheap, explicit opponent-action prior for the first complete search layer."""
        targets = [e for e in s.entities if e.side == 1 and not e.tower]
        if not targets:
            return SimAction()
        def score(a):
            if a.card_id is None:
                return 0
            x, y = self.sim.xy(a.x, a.y)
            roster = self.kb.roster(a.card_id, self.sim.level)
            dps = sum(unit.damage / unit.period * count for unit, count in roster)
            distance = min(((x-e.x)**2 + (y-e.y)**2)**.5 for e in targets)
            return dps / (1 + distance) / (1 + self.kb.cards[a.card_id]["elixir"])
        return max(self.candidates(s, -1, limit=5), key=score)
