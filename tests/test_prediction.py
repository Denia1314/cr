from __future__ import annotations

import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import MagicMock

from PIL import Image

from crbot.battle_world import BattleWorld, Track, WorldSnapshot
from crbot.battle_simulation import Simulator, SimState, SimAction
from crbot.knowledge import KnowledgeBase
from crbot.predictive_planner import PredictivePlanner, PlanResult
from crbot.decision_router import create_policy, PredictiveBattlePolicy
from crbot.policy import BattlePolicy
from crbot.cards import CardCatalog
from crbot.battle_perception import HandCardMatch, LaneThreat
from crbot.runtime_model import (apply_decision_engine, load_decision_engine, save_decision_engine,
                                 load_runtime_model, save_runtime_model, DECISION_ENGINE_LABELS)
from crbot.prediction_audit import audit_predictions
from tests.test_core import base_config

ROOT = Path(__file__).resolve().parents[1]


def knowledge():
    return KnowledgeBase.load(ROOT / "data/battle_knowledge.json")


def world(**kwargs):
    defaults = dict(revision=1, at=100., elapsed=20., elixir=7., enemy_elixir=(0., 7.),
                    hand=((0, "knight"), (1, "musketeer"), (2, "cannon"), (3, "fireball")),
                    tracks=(Track(1, "giant", -1, .28, .5, 99, 100, .9, 3),),
                    enemy_seen=("giant",), enemy_recent=(), uncertainty=())
    defaults.update(kwargs)
    return WorldSnapshot(**defaults)


class KnowledgeTests(unittest.TestCase):
    def test_every_legacy_card_retained_and_provenance_visible(self):
        kb = knowledge()
        ids = set(CardCatalog.load(ROOT / "data/cards.json").ids())
        self.assertEqual(ids, set(kb.cards))
        self.assertEqual(len(kb.source["revision"]), 40)
        self.assertFalse(kb.audit()["validated_current_balance"])

    def test_level_scaling_uses_rarity_offset_and_projectile_damage(self):
        kb = knowledge()
        raw = kb.units["Musketeer"]
        self.assertEqual(kb.unit("Musketeer", 11).hp, raw["hitpoints_per_level"][8])
        self.assertGreater(kb.unit("Musketeer").damage, 0)
        self.assertGreater(kb.unit("Knight", 12).hp, kb.unit("Knight", 11).hp)

    def test_missing_not_zero_and_invalid_level_rejected(self):
        kb = knowledge()
        self.assertIsNone(kb.unit("not_a_unit"))
        self.assertEqual(kb.roster("not_a_card"), [])
        with self.assertRaises(ValueError):
            kb.unit("Knight", 100)

    def test_card_and_spawn_entity_are_distinct(self):
        kb = knowledge()
        self.assertEqual(sum(n for _, n in kb.roster("skeleton_army")), 15)
        self.assertEqual(kb.roster("golem")[0][0].death_spawn, "Golemite")

    def test_source_is_not_mutated_by_queries(self):
        kb = knowledge()
        before = json.dumps(kb.payload, sort_keys=True)
        kb.audit()
        self.assertEqual(before, json.dumps(kb.payload, sort_keys=True))


class WorldTests(unittest.TestCase):
    def setUp(self):
        self.w = BattleWorld()

    def update(self, detections=(), now=100):
        return self.w.update(list(detections), now=now, elapsed=20, elixir=5, hand=[(0, "knight")],
                             costs={"giant": 5}, seconds_per_elixir=2.8)

    def enemy(self, **kwargs):
        return dict(card_id="giant", side=-1, x=.28, y=.35, confidence=.9, **kwargs)

    def test_repeated_unit_not_repeated_play(self):
        for n in range(6):
            self.update([self.enemy()], 100+n*.2)
        self.assertEqual(len(self.w.tracks), 1)
        self.assertEqual(len(self.w.events), 1)
        self.assertEqual(self.w.events[0]["kind"], "sighting")
        self.assertEqual(self.w.enemy_elixir, (0, 10))

    def test_group_members_do_not_multiply_cost(self):
        a, b = self.enemy(), self.enemy()
        b["x"] += .02
        self.update([a, b])
        self.assertEqual(len(self.w.tracks), 2)
        self.assertEqual(len(self.w.events), 1)

    def test_confirmed_deployment_is_deduplicated(self):
        d = self.enemy(deploy_event_id="d1", deployment_confirmed=True)
        self.update([d])
        self.assertEqual(self.w.enemy_elixir, (0, 5))
        self.update([d], now=100)
        self.assertEqual(self.w.enemy_elixir, (0, 5))

    def test_snapshot_does_not_share_mutable_tracks(self):
        before = self.update([self.enemy()])
        self.update([dict(self.enemy(), x=.3)], 100.2)
        self.assertEqual(before.tracks[0].x, .28)

    def test_occlusion_is_not_death_and_old_tracks_expire(self):
        self.update([self.enemy()])
        self.assertEqual(len(self.update(now=101).tracks), 1)
        self.assertEqual(len(self.update(now=105).tracks), 0)

    def test_time_reversal_rejected(self):
        self.update()
        with self.assertRaises(ValueError):
            self.update(now=99)

    def test_confirmed_ally_event_not_invented_entity(self):
        self.w.confirm("a1", "knight", .3, .6, 100)
        self.w.confirm("a1", "knight", .3, .6, 100)
        self.assertEqual(len(self.w.events), 1)
        self.assertFalse(self.w.tracks)

    def test_reset_clears_previous_opponent(self):
        self.update([self.enemy()])
        self.w.reset()
        self.assertFalse(self.w.enemy_seen)
        self.assertFalse(self.w.events)


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.kb = knowledge()
        self.sim = Simulator(self.kb)

    def test_illegal_elixir_hand_and_enemy_placement(self):
        s = SimState(elixir={1: 2, -1: 5}, hands={1: [(0, "knight")], -1: []})
        self.assertFalse(self.sim.apply(s, SimAction("knight", 0), 1))
        s.elixir[1] = 5
        self.assertFalse(self.sim.apply(s, SimAction("knight", 0, .3, .3), 1))
        self.assertFalse(self.sim.apply(s, SimAction("giant", 0), 1))
        self.assertEqual(s.elixir[1], 5)

    def test_apply_consumes_once_and_wait_is_valid(self):
        s = SimState(hands={1: [(0, "knight")], -1: []})
        self.assertTrue(self.sim.apply(s, SimAction("knight", 0), 1))
        self.assertFalse(self.sim.apply(s, SimAction("knight", 0), 1))
        self.assertEqual(s.elixir[1], 2)
        self.assertTrue(self.sim.apply(s, SimAction(), 1))

    def test_ground_only_cannot_damage_air(self):
        s = SimState()
        self.sim.add(s, self.kb.unit("Knight"), 1, 4, 16)
        air = self.sim.add(s, replace(self.kb.unit("Minion"), speed=0, damage=0), -1, 4, 16)
        hp = air.hp
        self.sim.advance(s, 3)
        self.assertEqual(air.hp, hp)

    def test_building_target_ignores_nearby_troop(self):
        s = SimState()
        giant = self.sim.add(s, self.kb.unit("Giant"), -1, 4, 18)
        knight = self.sim.add(s, self.kb.unit("Knight"), 1, 4, 18)
        tower = self.sim.add(s, self.kb.unit("PrincessTower"), 1, 4, 20, tower=True)
        self.sim.advance(s, .25)
        self.assertEqual(giant.target, tower.uid)
        self.assertNotEqual(giant.target, knight.uid)

    def test_shield_absorbs_hit_without_hp_overflow(self):
        s = SimState()
        e = self.sim.add(s, self.kb.unit("DarkPrince"), 1, 4, 18)
        hp = e.hp
        self.sim.hit(s, e, 10000)
        self.assertEqual(e.hp, hp)
        self.assertEqual(e.shield, 0)

    def test_death_spawns_children(self):
        s = SimState()
        golem = self.sim.add(s, self.kb.unit("Golem"), -1, 4, 18)
        golem.hp = 0
        self.sim.advance(s, .25)
        self.assertEqual(sum(e.spec.name == "Golemite" for e in s.entities), 2)

    def test_clone_keeps_branch_state_independent(self):
        s = SimState(hands={1: [(0, "knight")], -1: []})
        other = s.clone()
        self.sim.apply(other, SimAction("knight", 0), 1)
        self.assertFalse(s.entities)
        self.assertEqual(s.elixir[1], 5)
        self.assertEqual(len(s.hands[1]), 1)

    def test_deadline_stops_simulation(self):
        with self.assertRaises(TimeoutError):
            self.sim.advance(SimState(), 8, deadline=1, clock=lambda: 2)

    def test_elixir_caps_at_ten(self):
        s = SimState()
        self.sim.advance(s, 100)
        self.assertEqual(s.elixir[1], 10)

    def test_freeze_damage_is_not_repeated_for_each_second(self):
        s = SimState(hands={1: [(0, "freeze")], -1: []})
        self.assertTrue(self.sim.apply(s, SimAction("freeze", 0, .3, .5), 1))
        self.assertEqual(len(s.impacts), 1)

    def test_charge_and_ramp_increase_damage(self):
        for name in ("Prince", "InfernoDragon"):
            with self.subTest(name=name):
                s = SimState()
                attacker = self.sim.add(s, self.kb.unit(name), 1, 4, 18)
                target = self.sim.add(s, replace(self.kb.unit("Giant"), hp=10000, damage=0, speed=0), -1, 4, 18)
                attacker.target, attacker.walked, attacker.locked_at = target.uid, 10, -10
                self.sim.advance(s, .5)
                self.assertGreater(10000 - target.hp, attacker.spec.damage)


class PlannerTests(unittest.TestCase):
    def test_search_compares_response_and_followup(self):
        planner = PredictivePlanner(knowledge(), {"budget_ms": 2000})
        result = planner.plan(world())
        self.assertEqual(result.status, "ready")
        self.assertGreaterEqual(len(result.candidates), 5)
        self.assertTrue(any(c["label"] == "WAIT" for c in result.candidates))
        self.assertTrue(all(len(c["branches"]) == 3 for c in result.candidates))
        self.assertIn("own_followup", result.candidates[0]["branches"][0])
        self.assertGreater(result.nodes, 0)

    def test_low_elixir_waits_without_overspending(self):
        result = PredictivePlanner(knowledge(), {"budget_ms": 2000}).plan(world(elixir=0))
        self.assertEqual(result.status, "wait")

    def test_explicit_timeout_is_not_partial_best_action(self):
        planner = PredictivePlanner(knowledge(), {"budget_ms": 1})
        counter = iter(range(100000))
        planner.clock = lambda: next(counter) * .01
        result = planner.plan(world())
        self.assertEqual(result.status, "timeout")
        self.assertIsNone(result.action.card_id)

    def test_changes_in_enemy_lane_change_placement(self):
        planner = PredictivePlanner(knowledge(), {"budget_ms": 2000})
        left = planner.plan(world())
        right = planner.plan(world(tracks=(Track(1, "giant", -1, .72, .5, 99, 100, .9, 3),)))
        self.assertLess(left.action.x, .5)
        self.assertGreaterEqual(right.action.x, .5)  # central ranged support can cover the right lane
        self.assertGreater(right.action.x,left.action.x)

    def test_no_full_swarm_is_spawned_per_detected_box(self):
        planner = PredictivePlanner(knowledge(), {})
        s = planner.initial(world(tracks=(Track(1, "skeleton_army", -1, .3, .5, 99, 100, .9),)), 5, 1)
        self.assertEqual(len([e for e in s.entities if not e.tower]), 1)

    def test_timeout_after_complete_layer_keeps_fair_comparison(self):
        planner = PredictivePlanner(knowledge(), {"budget_ms": 100})
        timer, calls = [0.], [0]
        planner.clock = lambda: timer[0]
        original = planner.sim.evaluate
        root_count=len(planner.candidates(planner.initial(world(),7,1),1,limit=49))
        def evaluate(s):
            calls[0] += 1
            if calls[0] == root_count*3:
                timer[0] = 1.
            return original(s)
        with patch.object(planner.sim, "evaluate", side_effect=evaluate):
            result = planner.plan(world())
        self.assertEqual(result.completed_depth, 1)
        self.assertTrue(result.budget_exhausted)
        self.assertIn(result.status, {"ready", "wait"})
        self.assertEqual(len(result.candidates), root_count)
        self.assertTrue(all(e['status']=='complete' for e in result.hand_evaluations))


class SelectionTests(unittest.TestCase):
    def test_selections_preserve_each_other_and_unknown_settings(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            (p.parent / ".royal-lab.json").write_text('{"other": 42}', encoding="utf-8")
            save_decision_engine(p, "predictive")
            save_runtime_model(p, "m2")
            self.assertEqual(load_decision_engine(p), "predictive")
            self.assertEqual(load_runtime_model(p), "m2")
            self.assertEqual(json.loads((p.parent / ".royal-lab.json").read_text())["other"], 42)

    def test_legacy_default_and_model_identity_isolation(self):
        config = base_config()
        self.assertIs(type(create_policy(config)), BattlePolicy)
        predictive = apply_decision_engine(config, "predictive")
        self.assertEqual(predictive["replay"]["training_policy_version"], "predictive_spatial_v1")
        self.assertEqual(predictive["replay"]["transfer_policy_versions"], [])
        self.assertNotIn("decision_engine", config["policy"])

    def test_corrupt_settings_are_not_destroyed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            target = p.parent / ".royal-lab.json"
            target.write_text('{bad', encoding="utf-8")
            with self.assertRaises(ValueError):
                save_decision_engine(p, "shadow")
            self.assertEqual(target.read_text(), '{bad')

    def test_stage_is_in_actual_predictor_option_not_legacy_option(self):
        from crbot import current_stage_label
        from crbot.runtime_model import RUNTIME_MODEL_LABELS
        self.assertEqual(DECISION_ENGINE_LABELS["predictive"], current_stage_label())
        self.assertIn("M3-E", RUNTIME_MODEL_LABELS["m3_c3"])


class RouterTests(unittest.TestCase):
    def setUp(self):
        config = base_config()
        config["policy"]["decision_engine"] = "predictive"
        config["prediction"] = {"knowledge_path": str(ROOT / "data/battle_knowledge.json")}
        self.p = PredictiveBattlePolicy(config)
        self.p.catalog = CardCatalog.load(ROOT / "data/cards.json")
        self.p.hand_recognizer = SimpleNamespace(available=True, templates={"x": 1})
        self.p.learned_detector = SimpleNamespace(available=True, observed_enemies=[
            {"card_id": "giant", "x": .28, "y": .5, "confidence": .9}], observed_allies=[])
        self.p._last_hand_matches = [HandCardMatch(0, "knight", .9, 50, 1, 80, False)]
        self.p._last_elixir_estimate_value = 7
        self.p.next_action_at = 0
        self.p.battle_started_at = 80
        self.image = Image.new("RGB", (50, 50))
        self.p._last_hand_image = self.image
        self.p.hand_history.update(self.p._last_hand_matches, 100)
        threats = {lane: LaneThreat(lane, .4 if lane == "left" else 0, 1, .5, "heavy", ((.28, .5),), ("giant",))
                   for lane in ("left", "right")}
        self.observe = patch.object(self.p, "observe_replay_state", return_value={}).start()
        self.threats = patch.object(self.p, "_perceive_threats", return_value=threats).start()
        self.addCleanup(patch.stopall)

    def result(self, status="ready"):
        return PlanResult(1, 100, 101, status, SimAction("knight", 0, .28, .65), reason="test")

    def test_shadow_does_not_apply_planned_action(self):
        self.p.decision_engine = "shadow"
        before = self.p.snapshot_state()
        with patch.object(self.p.planner, "plan", return_value=self.result()), patch.object(BattlePolicy, "decide", return_value=None) as old:
            self.p.decide(self.image, None, now=100)
        old.assert_called_once()
        self.assertEqual(self.p.virtual_elixir, before["virtual_elixir"])
        self.assertEqual(self.p.action_sequence, 0)

    def test_timeout_waits_for_fresh_frame_without_legacy_click(self):
        with patch.object(self.p.planner, "plan", return_value=self.result("timeout")), patch.object(BattlePolicy, "decide", return_value=None) as old:
            self.p.decide(self.image, None, now=100)
        old.assert_not_called()
        self.assertEqual(self.p.last_execution_engine, "none")

    def test_unknown_enemy_is_hypothesis_not_a_recognized_card(self):
        self.p.learned_detector.observed_enemies = []
        with patch.object(self.p.planner, "plan", return_value=self.result()) as plan:
            self.p.decide(self.image, None, now=100)
        observed = plan.call_args.args[0]
        self.assertTrue(observed.tracks[0].card_id.startswith("unknown:"))
        self.assertFalse(observed.enemy_seen)
        self.assertEqual(self.p.last_plan["perception_mode"], "lane_hypotheses")

    def test_prediction_confirmed_once_and_rejection_rolls_back(self):
        for status in ("rejected", "confirmed"):
            self.p.next_action_at = 0
            snapshot = self.p.snapshot_state()
            with patch.object(self.p.planner, "plan", return_value=self.result()):
                decision = self.p.decide(self.image, None, now=100)
            self.assertIsNotNone(decision)
            reserved = self.p.prepare_action(decision, snapshot)
            self.p.resolve_action(reserved.action_id, status, now=101)
            self.assertFalse(self.p.resolve_action(reserved.action_id, status, now=101))
            if status == "rejected":
                self.assertEqual(self.p.virtual_elixir, snapshot["virtual_elixir"])
                self.assertFalse(self.p.world.confirmed)
            else:
                self.assertIn(reserved.action_id, self.p.world.confirmed)

    def test_pending_action_blocks_new_search(self):
        self.p._pending_action_id = "pending"
        with patch.object(self.p.planner, "plan") as plan:
            self.assertIsNone(self.p.decide(self.image, None, now=100))
        plan.assert_not_called()
        self.assertEqual(self.p.world.revision, 1)

    def test_expired_search_never_falls_back_to_old_click(self):
        result = self.result()
        result.elapsed_ms = 1100
        with patch.object(self.p.planner, "plan", return_value=result), patch.object(BattlePolicy, "decide") as old:
            self.assertIsNone(self.p.decide(self.image, None, now=100))
        old.assert_not_called()
        self.assertEqual(self.p.last_plan["fallback_reason"], "plan_expired_recapture")

    def test_device_capture_latency_does_not_expire_new_default_plan(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.p.prediction_frame_age_s = 1.05
        result = self.result()
        result.valid_until = 100 + config["prediction"]["max_plan_age_s"]
        result.elapsed_ms = 100
        with patch.object(self.p.planner, "plan", return_value=result):
            decision = self.p.decide(self.image, None, now=100)
        self.assertIsNotNone(decision)
        self.assertGreater(decision.plan_valid_until, 101.2)

    def test_no_enemy_uses_normal_play_instead_of_permanent_wait(self):
        self.p.planning_config["fast_defense"] = True
        self.p.learned_detector.observed_enemies = []
        empty = {lane: LaneThreat(lane, 0, 0, 0, "none", (), ()) for lane in ("left", "right")}
        with patch.object(self.p, "_perceive_threats", return_value=empty), patch.object(self.p.planner, "plan") as plan, patch.object(BattlePolicy, "decide", return_value=None) as old:
            self.p.decide(self.image, None, now=100)
        plan.assert_not_called()
        old.assert_called_once()
        self.assertEqual(self.p.last_plan["fallback_reason"], "no_active_enemy_use_normal_play")

    def test_capture_age_counts_toward_expiration(self):
        self.p.prediction_frame_age_s = 1.1
        with patch.object(self.p.planner, "plan", return_value=self.result()), patch.object(BattlePolicy, "decide") as old:
            self.assertIsNone(self.p.decide(self.image, None, now=100))
        old.assert_not_called()
        self.assertEqual(self.p.last_plan["fallback_reason"], "plan_expired_recapture")

    def test_cooldown_updates_health_without_search_or_spending(self):
        self.p.next_action_at = 101
        self.p.learned_detector.observed_enemies[0]["hp_fraction"] = .1
        with patch.object(self.p.planner, "plan") as plan:
            self.assertIsNone(self.p.decide(self.image, None, now=100))
        plan.assert_not_called()
        self.assertEqual(next(iter(self.p.world.tracks.values())).hp_fraction, .1)
        self.assertEqual(self.p.action_sequence, 0)

    def test_unstable_raw_hand_is_not_used_for_prediction(self):
        self.p.hand_history.stability_frames = 2
        with patch.object(BattlePolicy, "decide", return_value=None), patch.object(self.p.planner, "plan") as planner:
            self.p.decide(self.image, None, now=100)
        planner.assert_not_called()
        self.assertEqual(self.p.last_plan["fallback_reason"], "hand_unavailable")

    def test_same_frame_is_not_counted_as_two_stability_observations(self):
        self.p._observed_image, self.p._observed_at = self.image, 100
        self.p._last_hand_observed_at = 100
        with patch.object(self.p.planner, "plan", return_value=self.result()):
            self.p.decide(self.image, None, now=100)
        self.observe.assert_not_called()

    def test_expired_plan_is_rejected_before_device_click(self):
        from crbot.engine import BotEngine
        before = self.p.snapshot_state()
        with patch.object(self.p.planner, "plan", return_value=self.result()):
            decision = self.p.decide(self.image, None, now=100)
        engine = object.__new__(BotEngine)
        engine.policy, engine.config = self.p, base_config()
        engine.recorder, engine.device, engine.dry_run = MagicMock(), MagicMock(), False
        with patch("crbot.engine.time.monotonic", return_value=200):
            result = engine._execute_action(self.image, decision, before)
        engine.device.tap_normalized.assert_not_called()
        self.assertEqual(result[0].action_status, "rejected")
        self.assertEqual(self.p.virtual_elixir, before["virtual_elixir"])


class AuditTests(unittest.TestCase):
    def test_audit_reads_real_flat_recorder_format(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "runs" / "sample" / "events.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text('\n'.join(json.dumps(e) for e in [
                {"event": "battle_prediction", "actual_engine": "predictive", "plan": {"elapsed_ms": 12}},
                {"event": "battle_action", "decision_engine": "predictive", "action_status": "confirmed"},
            ]), encoding="utf-8")
            result = audit_predictions(root)
            self.assertEqual(result["confirmed_predictive_actions"], 1)
            self.assertEqual(result["search_p95_ms"], 12)
            self.assertFalse(result["battle_acceptance"])


if __name__ == "__main__":
    unittest.main()
