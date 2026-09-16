"""Receding-horizon search over friendly actions and uncertain enemy responses."""
from __future__ import annotations

import time
import math
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
    simulation_version: str = "spatial_v4_calibrated_windup"
    nodes: int = 0
    completed_depth: int = 0
    budget_exhausted: bool = False
    placement_mode: str = "scene_grid_and_intercepts"
    hand_evaluations: list[dict] = field(default_factory=list)
    enemy_forecast: list[dict] = field(default_factory=list)
    compute: dict = field(default_factory=dict)
    tactical_phase: str = "legacy"
    combo_candidates: list[dict] = field(default_factory=list)
    learning: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class PredictivePlanner:
    def __init__(self, kb: KnowledgeBase, config: dict):
        self.kb, self.config = kb, config
        self.sim = Simulator(kb, int(config.get("assumed_level", 11)))
        from .arena_geometry import ArenaGeometry
        self.sim.geometry=ArenaGeometry.from_config(config.get("arena_geometry"))
        self.sim.grid_navigation = bool(config.get('grid_world', True))
        if config.get("gpu_placement", False):
            from .gpu_placement import PlacementBatch
            self.sim.placement_batch = PlacementBatch(str(config.get("compute_device", "auto")),trajectory=bool(config.get("gpu_trajectory_screening",False)),trajectory_step=float(config.get("gpu_trajectory_step_s",.25)))
        self.sim.defense_pipeline_s=float(config.get("defense_pipeline_s",.35))
        self.sim.placement_grid_step = float(config.get("placement_grid_step", 1.))
        self.clock = time.perf_counter
        self.combat_pool = None

    def warm_pool(self):
        if self.config.get('parallel_combat', False) and self.combat_pool is None:
            from .parallel_combat import CombatPool
            self.combat_pool = CombatPool(self.kb.payload, self.config)

    def candidates(self, s: SimState, side: int, *, limit=12) -> list[SimAction]:
        result = [SimAction(wait_s=.4)]
        for slot, cid in s.hands[side]:
            card = self.kb.cards.get(cid, {})
            cost = card.get("elixir")
            if cost is None or cost > s.elixir[side]:
                continue
            if not self.kb.roster(cid, self.sim.level) and not self.kb.spell(cid, self.sim.level):
                continue
            positions = max(2,min(24,int(self.config.get('positions_per_card',16))))
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
        for i,(x,y) in enumerate(self.sim.geometry.tower_points):
            side=1 if i<2 else -1
            px,py=self.sim.xy(x,y)
            fraction=world.tower_health[i]
            if fraction is None or fraction>0:
                self.sim.add(s,tower,side,px,py,tower=True,hp_fraction=1 if fraction is None else fraction)
        if self.config.get('king_towers', False):
            king = self.kb.unit('KingTower', self.sim.level)
            if king is not None:
                for i,side in enumerate((1,-1)):
                    points = self.config.get('king_tower_points')
                    px,py = self.sim.xy(*points[i]) if points else (9.,30.5 if side==1 else 1.5)
                    fraction = world.tower_health[4+i] if len(world.tower_health)>4+i else None
                    if fraction is None or fraction > 0:
                        entity=self.sim.add(s,king,side,px,py,tower=True,hp_fraction=fraction if fraction is not None else 1)
                        entity.tower_kind='king';entity.active=fraction is not None and fraction < 1
                s.uncertainties.add('king_activation_and_unobserved_health_estimated')
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
            delay=min(.75,max(0,world.observation_delay_s)+max(0,world.at-t.last_seen))
            x, y = self.sim.xy(t.x+t.vx*delay,t.y+t.vy*delay)
            x,y=max(0,min(18,x)),max(0,min(32,y))
            observed_hp = t.hp_fraction if t.hp_observed_at is None or world.at-t.hp_observed_at <= .6 else None
            fraction = observed_hp if observed_hp is not None else (health if t.side == -1 else max(.25,1.35-health))
            card = self.kb.cards[cid]
            entity = self.sim.add(s, spec, t.side, x, y, hp_fraction=fraction,
                                 value=float(card.get("elixir") or 0) / max(1, sum(n for _, n in roster)))
            entity.track_id=t.track_id
            entity.observed_vx, entity.observed_vy = t.vx*18/(self.sim.geometry.bounds[2]-self.sim.geometry.bounds[0]), t.vy*32/(self.sim.geometry.bounds[3]-self.sim.geometry.bounds[1])
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

    def plan(self, world: WorldSnapshot, *, action_scorer=None, learning_status=None) -> PlanResult:
        self.warm_pool()
        combat_mode = 'serial'
        fast = bool(self.config.get("fast_defense", False))
        started = self.clock()
        deadline = started + max(.01, min(2., float(self.config.get("budget_ms", 180)) / 1000))
        total_deadline = deadline
        if action_scorer is not None:
            reserve = max(0., min(.02, float(self.config.get("learning_budget_ms", 8)) / 1000))
            deadline -= min(reserve, (deadline - started) / 4)
        published_round = 0
        coarse_rows = []
        coarse_deadline=deadline
        refinement_start=started+(deadline-started)*.5
        result = PlanResult(world.revision, world.at, world.at + float(self.config.get("max_plan_age_s", 1.0)),
                            "unavailable", knowledge_version=self.kb.version)
        if self.sim.grid_navigation:
            result.simulation_version = 'spatial_v5_grid_navigation'
        result.learning = {**(learning_status or {}), "applied": False,
                           "selected_for_execution": False, "changed_selection": False}
        stage_started = time.perf_counter()
        result.compute['stage_ms'] = {}
        try:
            initial = self.initial(world, world.enemy_elixir[1], 1.)
            from .tactical_objective import phase_for,score_action
            result.tactical_phase=phase_for(initial) if self.config.get("unified_tactics",False) else "legacy"
            self.sim.tactical_phase=result.tactical_phase
            from .defense_timing import forecast
            result.enemy_forecast=forecast(self.sim,initial,world.hand,float(self.config.get('defense_pipeline_s',.35)))
            urgent=any(f['intervention_slack_s']<=3 for f in result.enemy_forecast)
            if urgent and result.tactical_phase=='prepare':
                result.tactical_phase='defend'
                self.sim.tactical_phase='defend'
            dense=sum(not e.tower for e in initial.entities)>16
            self.sim.placement_grid_step=float(self.config.get('placement_grid_step',1.))
            if dense and self.config.get('adaptive_defense_budget',False):
                self.sim.placement_grid_step=max(1.,self.sim.placement_grid_step)
                deadline=started+max(.01,min(.24,float(self.config.get('dense_defense_budget_ms',240))/1000))
                total_deadline=deadline
                coarse_deadline=deadline
                refinement_start=started+(deadline-started)*.5
            result.compute['defense_timing']=dict(urgent=urgent,dense=dense,budget_ms=round((deadline-started)*1000),
                enemy_elixir_interval=list(world.enemy_elixir),enemy_elixir_estimate=world.enemy_elixir_estimate,
                enemy_hand_observed=False)
            result.compute['stage_ms']['world_and_forecast'] = round((time.perf_counter()-stage_started)*1000, 2)
            stage_started = time.perf_counter()
            # Every legal hand card receives its full spatial shortlist before refinement.
            positions = max(2,min(24,int(self.config.get('positions_per_card',16))))
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
            result.compute['stage_ms']['spatial_ranking'] = round((time.perf_counter()-stage_started)*1000, 2)
            stage_started = time.perf_counter()
            horizon = max(4, min(15, float(self.config.get("horizon_s", 8))))
            if result.enemy_forecast:
                horizon=max(horizon,min(24,min(f['unopposed_tower_eta_s'] for f in result.enemy_forecast)+3))
            requested_horizon = horizon
            if fast and self.config.get('decision_horizon_cap_s') is not None:
                horizon = min(horizon, max(4., float(self.config['decision_horizon_cap_s'])))
            result.compute['decision_horizon'] = dict(requested_s=requested_horizon, simulated_s=horizon,
                                                      bounded=horizon < requested_horizon)
            urgent = any(f['intervention_slack_s'] <= 3 for f in result.enemy_forecast)
            if urgent:
                # Do not reserve half the budget for optional combinations while
                # the first defensive comparison is still unfinished.
                coarse_deadline = deadline
            result.compute['urgent_tower_defense'] = urgent
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
            initial_scenarios = [self.initial(world, cost, hp) for cost, hp, _ in scenarios]
            if (self.combat_pool is not None and self.combat_pool.available
                    and all(future.done() for future in self.combat_pool.pending)):
                combat_mode = 'parallel'
                rows = self.combat_pool.rows(world, roots, scenarios, horizon, result.tactical_phase, coarse_deadline, completion_order=True)
            else:
                rows = (self.evaluate_root(world, root, scenarios, horizon, result.tactical_phase,
                                           coarse_deadline, initial_scenarios) for root in roots)
            for row in rows:
                if row is not None:
                    root = SimAction(**row['action'])
                    result.nodes += len(row['branches'])
                    coarse_rows.append(row)
                    for entry in result.hand_evaluations:
                        if entry['card_id'] == root.card_id:
                            entry['evaluated'] += 1
                            entry['status'] = 'complete' if entry['evaluated'] == entry['positions'] else 'partial'
                    active = [e for e in result.hand_evaluations if e['positions']]
                    completed_round = min((e['evaluated'] for e in active), default=0)
                    required_rounds = 1  # Commit a fair first decision before optional spatial refinement.
                    if (active and completed_round >= required_rounds and completed_round > published_round
                            and any(r['action']['card_id'] is None for r in coarse_rows)):
                        # Commit equal spatial rounds across all usable cards. A timeout cannot
                        # prefer a card simply because its next location happened to finish first.
                        order = {(a.slot,a.card_id,a.x,a.y): i for i,a in enumerate(roots)}
                        coarse_rows.sort(key=lambda r: order[(r['action']['slot'],r['action']['card_id'],r['action']['x'],r['action']['y'])])
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
                        if fast and self.config.get('unified_tactics',False) and self.clock()>=refinement_start:
                            break
            if hasattr(rows,'close'):
                rows.close()
            # Publish only a complete, equally evaluated layer. Refinements remain private until complete.
            if not result.candidates or not fast:
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
                        if result.tactical_phase != 'legacy':
                            score = score_action(score, parts, world, self.kb, root, result.tactical_phase,
                                                 float(self.config.get('attack_reserve',3)), self.sim.geometry)
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
            if not result.candidates and self.config.get('allow_partial_comparison', False):
                waiting = next((r for r in coarse_rows if r['action']['card_id'] is None), None)
                # Only complete same-frame, same-scenario rows are eligible.
                # The WAIT baseline is mandatory for tower and reserve checks.
                if waiting is not None:
                    order = {(a.slot,a.card_id,a.x,a.y): i for i,a in enumerate(roots)}
                    complete = sorted((r for r in coarse_rows if r['action']['card_id']),
                        key=lambda r: order[(r['action']['slot'],r['action']['card_id'],r['action']['x'],r['action']['y'])])
                    compared = set()
                    subset = [waiting]
                    for row in complete:
                        if row['action']['card_id'] not in compared:
                            subset.append(row)
                            compared.add(row['action']['card_id'])
                    if compared:
                        result.candidates = subset
                        result.completed_depth = 1
                        result.compute['partial_hand_comparison'] = dict(
                            compared=sorted(compared), missing=[e['card_id'] for e in result.hand_evaluations
                                                               if e['positions'] and e['card_id'] not in compared],
                            global_best_claimed=False)
            if not result.candidates:
                result.status, result.reason = "timeout", "incomplete_search_fallback"
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            result.status, result.reason = "unavailable", f"world_model_error: {exc}"
            result.candidates = []
        result.compute['stage_ms']['first_comparison'] = round((time.perf_counter()-stage_started)*1000, 2)
        if result.candidates and fast and self.config.get("unified_tactics",False):
            try:
                result.combo_candidates=self.refine_combinations(world,result,roots,scenarios,horizon,deadline)
                if result.combo_candidates:result.completed_depth=2
            except TimeoutError:
                result.budget_exhausted=True
        if result.candidates:
            # An urgent WAIT must stand on the immediate no-play simulation;
            # a conditional future card is not an executed defensive action.
            selectable = result.candidates + [r for r in result.combo_candidates
                if not (urgent and r['action']['card_id'] is None)]
            if self.config.get('development_reserve_gate', False):
                from .tactical_objective import guard_development_reserve
                selectable, result.compute['development_reserve'] = guard_development_reserve(
                    selectable, world, self.kb, result.tactical_phase,
                    float(self.config.get('attack_reserve', 3)), urgent=urgent, level=self.sim.level, immediate_candidates=result.candidates)
            prefer_play = bool(self.config.get('prefer_immediate_play', False))
            if prefer_play:
                playable = [r for r in selectable
                            if (r['action']['slot'], r['action']['card_id']) in world.hand
                            and self.kb.cards[r['action']['card_id']].get('elixir') is not None
                            and self.kb.cards[r['action']['card_id']]['elixir'] <= world.elixir
                            and self.sim.legal_placement(initial, r['action']['card_id'], 1,
                                                        r['action']['x'], r['action']['y'])]
                result.compute['execution_preference'] = {
                    'enabled': True, 'completed_play_options': len(playable),
                    'reason': 'choose_immediate_play' if playable else 'no_completed_legal_play',
                }
                if playable:
                    selectable = playable
            baseline_selection = sorted(selectable,
                                        key=lambda c: c["score"], reverse=True)
            self.apply_learning(result, action_scorer, total_deadline)
            result.candidates.sort(key=lambda c: c["score"], reverse=True)
            selection=sorted(selectable,key=lambda c:c["score"],reverse=True)
            best = selection[0]
            baseline = baseline_selection[0]
            if fast and result.tactical_phase != 'prepare' and any(e.side == -1 and not e.tower for e in initial.entities):
                def loss(row):
                    return max(100000 * (sum(e.tower and e.side == 1 for e in initial.entities) - b.get("own_towers_remaining", 2))
                               + b["own_tower_damage"] + b.get("imminent_tower_exposure", 0)
                               for b in row["branches"])
                waiting = next((c for c in selection if c["action"]["card_id"] is None), None)
                if waiting is not None or prefer_play:
                    # Charge for real mitigation, not the residual value of an unnecessary troop.
                    best_loss = min(map(loss, selection))
                    tolerance = float(self.config.get("defense_damage_tolerance", 30))
                    sufficient = [c for c in selection if loss(c) <= best_loss + tolerance]
                    def expense(row):
                        cid = row["action"]["card_id"]
                        return row.get("planned_cost",self.kb.cards[cid]["elixir"] if cid else 0)
                    best = min(sufficient, key=lambda c: (expense(c), loss(c), -c["score"]))
                    baseline = min((c for c in baseline_selection if loss(c) <= best_loss + tolerance),
                                   key=lambda c: (expense(c), loss(c), -c.get("simulation_score", c["score"])))
                    result.placement_mode = "fast_defense_cost_and_tower_loss"

            from .tactical_objective import avoid_overflow
            preparation = result.tactical_phase == 'prepare' and not any(
                e.side == 1 and not e.tower and e.hp > 0 for e in initial.entities)
            eligible_ids = {id(row) for row in selectable}
            overflow_candidates = [row for row in result.candidates if id(row) in eligible_ids]
            best, overflow = avoid_overflow(best, overflow_candidates, world, self.kb,
                                           float(self.config.get('attack_reserve', 3)), 'score', preparation)
            baseline, _ = avoid_overflow(baseline, overflow_candidates, world, self.kb,
                                        float(self.config.get('attack_reserve', 3)), 'simulation_score', preparation)
            result.compute['elixir_overflow'] = overflow
            # Apply in preparation as well as contact defense. A deferred combo
            # cannot justify WAIT when a completed immediate play protects towers.
            from .tactical_objective import protect_towers_before_wait
            best, tower_wait = protect_towers_before_wait(
                best, overflow_candidates, world, self.kb,
                self.config.get('defense_damage_tolerance', 30),
                self.config.get('tower_wait_minimum_mitigation', 30))
            baseline, _ = protect_towers_before_wait(
                baseline, overflow_candidates, world, self.kb,
                self.config.get('defense_damage_tolerance', 30),
                self.config.get('tower_wait_minimum_mitigation', 30))
            result.compute['tower_wait_guard'] = tower_wait
            from .tactical_objective import emergency_tower_defense
            best, emergency = emergency_tower_defense(best, result.candidates, world, self.sim, initial,
                float(self.config.get('tower_emergency_damage', 600)))
            baseline, _ = emergency_tower_defense(baseline, result.candidates, world, self.sim, initial,
                float(self.config.get('tower_emergency_damage', 600)))
            result.compute['tower_emergency_defense'] = emergency
            result.learning.update(changed_selection=best["action"] != baseline["action"],
                                   baseline_action=baseline["action"],
                                   selected_action_supported="learned_value" in best)

            for entry in result.hand_evaluations:
                option=next((c for c in result.candidates if c['action']['card_id']==entry['card_id']),None)
                if option is not None:
                    entry['best_position']=[option['action']['x'],option['action']['y']]
                    entry['best_score']=option['score']
                    entry['worst_tower_damage']=max(b['own_tower_damage'] for b in option['branches'])
            result.action = SimAction(**best["action"])
            result.status = "wait" if result.action.card_id is None else "ready"
            responses = "; ".join(f"敌方 {b['enemy_response']} → 我方 {b['own_followup']}" for b in best["branches"])
            result.reason = f"首步比较 {len(result.candidates)} 个方案 / 深度 {result.completed_depth} / {horizon:g} 秒；{responses}"
            if overflow['changed']:
                result.reason += '；敌方沉底，提前安全展开' if preparation else '；接近满费，执行安全低费发展'
            if tower_wait['changed']:
                result.reason += f"；等待将增加皇家塔受伤风险，提前防守（预计减少伤害及暴露 {tower_wait['prevented_damage_and_exposure']:g}）"
            if emergency['changed']:
                result.reason += f"；皇家塔高危 {emergency['wait_damage_and_exposure']:g}，使用预留费用执行应急防守（不保证最坏场景减伤）"
            elif emergency['active']:
                result.reason += '；已解除高危留费限制，但完整方案中无可支付、合法且能交战的防守'
            elif best['action']['card_id'] is None and 'wait_damage_and_exposure' in tower_wait:
                result.reason += f"；立即等待预计塔伤及暴露 {tower_wait['wait_damage_and_exposure']:g}"
                if tower_wait['reason'] == 'no_completed_effective_defense':
                    result.reason += '，已完成方案中暂无可支付且有效减伤的立即防守'
            if result.combo_candidates:
                result.reason += f"；条件式后续完整比较 {len(result.combo_candidates)} 个根动作（每牌最佳首步及等待）"
        if getattr(self.sim, "placement_batch", None):
            result.compute.update(self.sim.placement_batch.status())
        if self.combat_pool is not None:
            result.compute["parallel_combat"] = self.combat_pool.status()
            result.compute["parallel_combat"]['last_mode'] = combat_mode
        result.compute["arena_geometry"]=asdict(self.sim.geometry)
        result.compute["grid_step_tiles"] = self.sim.placement_grid_step
        result.compute["all_placement_points"] = bool(self.config.get("all_placement_points",False))
        result.compute["spatial_scored"] = sum(e["spatial_scored"] for e in result.hand_evaluations)
        result.compute["combat_evaluated"] = sum(e["evaluated"] for e in result.hand_evaluations)
        result.compute["combat_complete"] = bool(result.hand_evaluations) and all(e["evaluated"] == e["positions"] for e in result.hand_evaluations)
        if self.config.get("all_placement_points",False):
            result.reason = f"格点评分 {result.compute['spatial_scored']}；战斗精算 {result.compute['combat_evaluated']}/{result.compute['spatial_scored']}；" + result.reason
        result.compute['usable_comparison'] = bool(result.candidates)
        result.compute['completed_fair_rounds'] = published_round
        if result.status in {'ready', 'wait'}:
            result.reason = f"选择 {result.action.label}；" + result.reason
            if result.compute.get('partial_hand_comparison'):
                missing = ','.join(result.compute['partial_hand_comparison']['missing'])
                result.reason += f"；使用完整子集比较，未完成手牌：{missing}"
            reserve_audit = result.compute.get('development_reserve', {})
            if result.status == 'wait' and reserve_audit.get('blocked') and not result.compute.get('tower_emergency_defense',{}).get('active'):
                requirements = [o['required'] for o in reserve_audit.get('options', []) if not o['accepted']]
                needed = min(requirements, default=float(self.config.get('attack_reserve',3)))
                result.reason += f"；留费防下一波：出牌后需保留至少 {needed:g} 费"
        result.elapsed_ms = round((self.clock() - started) * 1000, 2)
        return result

    def evaluate_root(self, world, root, scenarios, horizon, phase, deadline, initial_scenarios=None):
        """Identical detailed simulation for serial and process workers."""
        from .tactical_objective import score_action
        self.sim.tactical_phase = phase
        initial_scenarios = initial_scenarios or [self.initial(world, cost, hp) for cost, hp, _ in scenarios]
        outcomes, branches = [], []
        fast = bool(self.config.get('fast_defense', False))
        for index, (enemy_cost, hp, responds) in enumerate(scenarios):
            if self.clock() >= deadline:
                raise TimeoutError()
            state = initial_scenarios[index].clone()
            pipeline=max(0.,min(1.,float(self.config.get('defense_pipeline_s',0))))
            if pipeline:self.sim.advance(state,pipeline,deadline=deadline,clock=self.clock)
            if not self.sim.apply(state, root, 1):
                return None
            self.sim.advance(state, 1.5, deadline=deadline, clock=self.clock)
            response = (self.attack_response(state) if fast and phase in {'develop','counterpush','prepare'} and root.card_id
                        else self.prior_response(state) if responds else SimAction())
            self.sim.apply(state, response, -1)
            self.sim.advance(state, horizon-1.5-pipeline, deadline=deadline, clock=self.clock)
            score, parts = self.sim.evaluate(state)
            if phase != 'legacy':
                score = score_action(score, parts, world, self.kb, root, phase,
                                     float(self.config.get('attack_reserve',3)), self.sim.geometry)
            outcomes.append(score)
            branches.append(dict(enemy_elixir_assumption=enemy_cost, enemy_response=response.label,
                                 own_followup='WAIT', pipeline_delay_s=pipeline, score=round(score,4), **parts))
        risk = float(self.config.get('downside_weight',.65))
        score = (1-risk)*sum(outcomes)/len(outcomes)+risk*min(outcomes)
        return dict(action=asdict(root), label=root.label, score=round(score,4), branches=branches)

    def close(self):
        if self.combat_pool is not None:
            self.combat_pool.close()

    def apply_learning(self, result, scorer, deadline):
        """Atomically adjust completed rows; preserve tower and cost selection guards."""
        if scorer is None:
            result.learning.setdefault("reason", "no_action_scorer")
            return
        rows = result.candidates + result.combo_candidates
        updates = []
        try:
            for row in rows:
                if self.clock() >= deadline:
                    raise TimeoutError()
                estimate = scorer(row["action"]) if row["action"].get("card_id") else None
                if estimate is not None:
                    delta = float(estimate["delta"])
                    if not math.isfinite(delta):
                        raise ValueError("nonfinite learned score")
                    updates.append((row, estimate, max(-1., min(1., delta))))
            if self.clock() >= deadline:
                raise TimeoutError()
        except (TimeoutError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
            result.learning.update(reason="learning_budget_exhausted" if isinstance(exc, TimeoutError)
                                   else "learning_error", error=str(exc), applied=False)
            return
        for row, estimate, delta in updates:
            row["simulation_score"] = row["score"]
            row["learned_value"] = estimate
            row["score"] += delta
        result.learning.update(applied=bool(updates), supported_rows=len(updates),
                               compared_rows=len(rows), reason="applied" if updates else "insufficient_support")

    def refine_combinations(self,world,result,roots,scenarios,horizon,deadline):
        counts={};shortlist=[]
        per_card=max(1,min(3,int(self.config.get('combo_root_positions',1))))
        def priority(row):
            if result.tactical_phase == 'defend':
                return (max(-100000*b['own_towers_remaining']+b['own_tower_damage']
                            +b.get('imminent_tower_exposure',0) for b in row['branches']), -row['score'])
            return (0, -row['score'])
        for row in sorted(result.candidates,key=priority):
            cid=row['action']['card_id']
            if counts.get(cid,0)<(per_card if cid else 1):
                shortlist.append(SimAction(**row['action']));counts[cid]=counts.get(cid,0)+1
        result.compute['combination_search']=dict(root_positions=len(shortlist),complete=False,
            followup_positions_per_card=max(1,min(4,int(self.config.get('combo_positions_per_card',2)))))
        if self.combat_pool is not None and self.combat_pool.available:
            from concurrent.futures import wait
            wait(self.combat_pool.pending,timeout=min(.01,max(0,deadline-self.clock())))
            if all(f.done() for f in self.combat_pool.pending):
                result.compute['combination_search']['mode']='parallel'
                source=self.combat_pool.rows(world,shortlist,scenarios,horizon,result.tactical_phase,deadline,kind='combo')
            else:
                raise TimeoutError()
        else:
            result.compute['combination_search']['mode']='serial'
            source=(self.evaluate_combination(world,root,scenarios,horizon,result.tactical_phase,deadline) for root in shortlist)
        rows=[]
        try:
            for row in source:
                if row is not None:rows.append(row)
        except TimeoutError:
            result.budget_exhausted=True
        finally:
            if hasattr(source,'close'):source.close()
        result.nodes += sum(row.pop('simulation_nodes',0) for row in rows)
        result.compute['combination_search']['completed_roots']=len(rows)
        result.compute['combination_search']['complete']=len(rows)==len(shortlist)
        return rows

    def evaluate_combination(self,world,root,scenarios,horizon,phase,deadline):
        from .tactical_objective import score_action
        self.sim.tactical_phase=phase
        outcomes=[];branches=[];costs=[];nodes=0
        positions=max(1,min(4,int(self.config.get('combo_positions_per_card',2))))
        for enemy_cost,hp,_ in scenarios:
            if self.clock()>=deadline:raise TimeoutError()
            state=self.initial(world,enemy_cost,hp)
            pipeline=max(0.,min(1.,float(self.config.get('defense_pipeline_s',0))))
            if pipeline:self.sim.advance(state,pipeline,deadline=deadline,clock=self.clock)
            if not self.sim.apply(state,root,1):raise ValueError('invalid tactical root')
            self.sim.advance(state,1.5,deadline=deadline,clock=self.clock)
            response=self.attack_response(state) if phase in {'develop','counterpush','prepare'} and root.card_id else SimAction()
            self.sim.apply(state,response,-1)
            # Recompute locations from the actual simulated survivors, health and elixir.
            followups=[SimAction()]
            for slot,cid in state.hands[1]:
                cost=self.kb.cards[cid].get('elixir')
                if cost is None or cost>state.elixir[1]:continue
                for x,y in placement_points(self.sim,state,cid,1,limit=positions):
                    followups.append(SimAction(cid,slot,x,y))
            best=None; alternatives=[]
            for follow in followups:
                if self.clock()>=deadline:raise TimeoutError()
                trial=state.clone()
                if not self.sim.apply(trial,follow,1):continue
                second_response=SimAction()
                if follow.card_id and phase in {'develop','counterpush','prepare'}:
                    second_response=self.attack_response(trial);self.sim.apply(trial,second_response,-1)
                self.sim.advance(trial,max(0,horizon-1.5),deadline=deadline,clock=self.clock)
                score,parts=self.sim.evaluate(trial)
                score=score_action(score,parts,world,self.kb,root,phase,float(self.config.get('attack_reserve',3)),self.sim.geometry)
                spent=sum(float(self.kb.cards[c]['elixir']) for c in (root.card_id,follow.card_id) if c)
                if phase!='defend':
                    reserve=float(self.config.get('attack_reserve',3))
                    root_cost=float(self.kb.cards[root.card_id]['elixir']) if root.card_id else 0
                    score-=(max(0,reserve-world.elixir+spent)-max(0,reserve-world.elixir+root_cost))*1.5
                if best is None or score>best[0]:best=(score,parts,follow,spent,second_response)
                alternatives.append((score,parts,follow,spent,second_response))
                nodes+=1
            if phase=='defend':
                towers=sum(e.side==1 and e.tower for e in state.entities)
                def damage(option):
                    parts=option[1]
                    return 100000*(towers-parts['own_towers_remaining'])+parts['own_tower_damage']+parts.get('imminent_tower_exposure',0)
                least=min(map(damage,alternatives))
                tolerance=float(self.config.get('defense_damage_tolerance',30))
                best=min((option for option in alternatives if damage(option)<=least+tolerance),key=lambda option:(option[3],damage(option),-option[0]))
            score,parts,follow,spent,second_response=best
            outcomes.append(score);costs.append(spent)
            branches.append(dict(enemy_elixir_assumption=enemy_cost,enemy_response=response.label,
                own_followup=follow.label,followup_action=asdict(follow),followup_after_s=1.5+pipeline,pipeline_delay_s=pipeline,
                followup_candidates=len(followups),enemy_followup_response=second_response.label,score=round(score,4),**parts))
        risk=float(self.config.get('downside_weight',.65))
        return dict(action=asdict(root),label=root.label,score=round((1-risk)*sum(outcomes)/len(outcomes)+risk*min(outcomes),4),
                    branches=branches,planned_cost=max(costs),simulation_nodes=nodes,scope='conditional_two_card_dynamic_positions')

    def attack_response(self,s):
        targets=[e for e in s.entities if e.side==1 and not e.tower]
        towers=[e for e in s.entities if e.side==-1 and e.tower]
        if not targets or not towers:return SimAction()
        target=min(targets,key=lambda e:min(self.sim.distance(e,t) for t in towers))
        tower=min(towers,key=lambda t:self.sim.distance(target,t))
        choices=[]
        for slot,cid in s.hands[-1]:
            cost=self.kb.cards[cid].get('elixir')
            if cost is None or cost>s.elixir[-1]:continue
            roster=self.kb.roster(cid,self.sim.level)
            dps=sum(u.damage/max(.1,u.period)*n for u,n in roster if ('air' if target.spec.air else 'ground') in u.targets)
            x,y=self.sim.screen(tower.x,min(15.,tower.y+2.))
            if dps and self.sim.legal_placement(s,cid,-1,x,y):choices.append((dps/(cost+1),SimAction(cid,slot,x,y)))
        return max(choices,key=lambda v:v[0])[1] if choices else SimAction()

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
