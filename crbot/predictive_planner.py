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
    simulation_version: str = "spatial_v3_card_effects"
    nodes: int = 0
    completed_depth: int = 0
    budget_exhausted: bool = False
    placement_mode: str = "scene_grid_and_intercepts"
    hand_evaluations: list[dict] = field(default_factory=list)
    enemy_forecast: list[dict] = field(default_factory=list)
    compute: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class PredictivePlanner:
    def __init__(self, kb: KnowledgeBase, config: dict):
        self.kb, self.config = kb, config
        self.sim = Simulator(kb, int(config.get("assumed_level", 11)))
        if config.get("gpu_placement", False):
            from .gpu_placement import PlacementBatch
            self.sim.placement_batch = PlacementBatch(str(config.get("compute_device", "auto")),trajectory=bool(config.get("gpu_trajectory_screening",False)))
        self.sim.placement_grid_step = float(config.get("placement_grid_step", 1.))
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
            positions = max(2,min(24,int(self.config.get('positions_per_card',12))))
            points = placement_points(self.sim, s, cid, side,
                                      limit=None if self.config.get("all_placement_points",False) else positions)
            for x, y in points:
                result.append(SimAction(cid, slot, x, y))
        # Round-robin by card avoids exhausting the budget on the first hand slot.
        if limit is None:
            limit = len(result)
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
            if t.hp_fraction == 0:
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
            entity = self.sim.add(s, spec, t.side, x, y, hp_fraction=fraction,
                                 value=float(card.get("elixir") or 0) / max(1, sum(n for _, n in roster)))
            entity.observed_vx, entity.observed_vy = t.vx*20, t.vy*50
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
        fast = bool(self.config.get("fast_defense", False))
        started = self.clock()
        deadline = started + max(.01, min(2., float(self.config.get("budget_ms", 180)) / 1000))
        result = PlanResult(world.revision, world.at, world.at + float(self.config.get("max_plan_age_s", 1.0)),
                            "unavailable", knowledge_version=self.kb.version)
        try:
            initial = self.initial(world, world.enemy_elixir[1], 1.)
            for enemy in (e for e in initial.entities if e.side == -1 and not e.tower):
                tower = min((t for t in initial.entities if t.side == 1 and t.tower),
                            key=lambda t: self.sim.distance(enemy,t), default=None)
                if tower is not None:
                    eta=max(0,self.sim.distance(enemy,tower)-enemy.spec.reach)/max(.1,enemy.spec.speed)
                    result.enemy_forecast.append(dict(unit=enemy.spec.name, x=enemy.x, y=enemy.y,
                        target_x=tower.x,target_y=tower.y,unopposed_tower_eta_s=round(eta,2),
                        observed_velocity=[enemy.observed_vx,enemy.observed_vy], status='hypothesis'))
            # Every legal hand card receives its full spatial shortlist before refinement.
            positions = max(2,min(24,int(self.config.get('positions_per_card',12))))
            roots = self.candidates(initial, 1, limit=None if self.config.get("all_placement_points",False) else 1 + positions * len(initial.hands[1]))
            for slot, cid in initial.hands[1]:
                card = self.kb.cards.get(cid, {})
                positions = sum(a.card_id == cid for a in roots)
                reason = ('dynamic_cost_unknown' if card.get('elixir') is None else
                          'insufficient_elixir' if card['elixir'] > initial.elixir[1] else
                          'mechanism_or_target_unavailable' if not positions else 'pending')
                result.hand_evaluations.append(dict(slot=slot, card_id=cid, positions=positions,
                                                    evaluated=0, status=reason, spatial_scored=positions,
                                                    full_domain=bool(self.config.get("all_placement_points",False))))
            horizon = max(4, min(15, float(self.config.get("horizon_s", 8))))
            if result.enemy_forecast:
                horizon=max(horizon,min(24,min(f['unopposed_tower_eta_s'] for f in result.enemy_forecast)+3))
            # Same scenarios and same horizon for every root, with conservative lower-tail weighting.
            scenarios = [(world.enemy_elixir[0], .65, False),
                         (max(world.enemy_elixir[0], min(world.enemy_elixir[1], world.enemy_elixir_estimate)), .85, True),
                         (world.enemy_elixir[1], 1., True)]
            if fast:
                # Current pressure first; new enemy deployments trigger a fresh plan.
                scenarios = [(world.enemy_elixir[1], 1., False)]
                if any(t.hypotheses for t in world.tracks):
                    scenarios.insert(0, (world.enemy_elixir[0], .65, False))
                    scenarios.insert(1, (world.enemy_elixir_estimate, .85, False))
            coarse_rows = []
            published_round = 0
            initial_scenarios = [self.initial(world, cost, hp) for cost, hp, _ in scenarios]
            for root in roots:
                outcomes, branches = [], []
                for index, (enemy_cost, hp, responds) in enumerate(scenarios):
                    s = initial_scenarios[index].clone()
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
                    for entry in result.hand_evaluations:
                        if entry['card_id'] == root.card_id:
                            entry['evaluated'] += 1
                            entry['status'] = 'complete' if entry['evaluated'] == entry['positions'] else 'partial'
                    active = [e for e in result.hand_evaluations if e['positions']]
                    completed_round = min((e['evaluated'] for e in active if e['evaluated'] < e['positions']), default=positions)
                    if active and completed_round > published_round:
                        # Commit equal spatial rounds across all usable cards. A timeout cannot
                        # prefer a card simply because its next location happened to finish first.
                        counts = {}
                        balanced = []
                        for row in coarse_rows:
                            cid = row['action']['card_id']
                            if cid is None or counts.get(cid, 0) < completed_round:
                                balanced.append(row)
                                counts[cid] = counts.get(cid, 0) + 1
                        result.candidates = balanced
                        result.completed_depth = 1
                        published_round = completed_round
            # Publish only a complete, equally evaluated layer. Refinements remain private until complete.
            result.candidates = coarse_rows
            result.completed_depth = 1
            refined_rows = []
            for root in ([] if fast else roots):
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
            if not fast:
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
            if fast and any(e.side == -1 and not e.tower for e in initial.entities):
                def loss(row):
                    return max(100000 * (sum(e.tower and e.side == 1 for e in initial.entities) - b.get("own_towers_remaining", 2))
                               + b["own_tower_damage"] + b.get("imminent_tower_exposure", 0)
                               for b in row["branches"])
                waiting = next((c for c in result.candidates if c["action"]["card_id"] is None), None)
                if waiting is not None:
                    # Charge for real mitigation, not the residual value of an unnecessary troop.
                    best_loss = min(map(loss, result.candidates))
                    tolerance = float(self.config.get("defense_damage_tolerance", 30))
                    sufficient = [c for c in result.candidates if loss(c) <= best_loss + tolerance]
                    def expense(row):
                        cid = row["action"]["card_id"]
                        return self.kb.cards[cid]["elixir"] if cid else 0
                    best = min(sufficient, key=lambda c: (expense(c), loss(c), -c["score"]))
                    result.placement_mode = "fast_defense_cost_and_tower_loss"

            for entry in result.hand_evaluations:
                option=next((c for c in result.candidates if c['action']['card_id']==entry['card_id']),None)
                if option is not None:
                    entry['best_position']=[option['action']['x'],option['action']['y']]
                    entry['best_score']=option['score']
                    entry['worst_tower_damage']=max(b['own_tower_damage'] for b in option['branches'])
            result.action = SimAction(**best["action"])
            result.status = "wait" if result.action.card_id is None else "ready"
            responses = "; ".join(f"敌方 {b['enemy_response']} → 我方 {b['own_followup']}" for b in best["branches"])
            result.reason = f"完整比较 {len(result.candidates)} 个方案 / 深度 {result.completed_depth} / {horizon:g} 秒；{responses}"
        if getattr(self.sim, "placement_batch", None):
            result.compute = self.sim.placement_batch.status()
        result.compute["grid_step_tiles"] = self.sim.placement_grid_step
        result.compute["all_placement_points"] = bool(self.config.get("all_placement_points",False))
        result.compute["spatial_scored"] = sum(e["spatial_scored"] for e in result.hand_evaluations)
        result.compute["combat_evaluated"] = sum(e["evaluated"] for e in result.hand_evaluations)
        result.compute["combat_complete"] = bool(result.hand_evaluations) and all(e["evaluated"] == e["positions"] for e in result.hand_evaluations)
        if self.config.get("all_placement_points",False):
            result.reason = f"格点评分 {result.compute['spatial_scored']}；战斗精算 {result.compute['combat_evaluated']}/{result.compute['spatial_scored']}；" + result.reason
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
