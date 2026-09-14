from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from crbot.action_value import ActionValueHead, evaluate, fit, train_head
from crbot.predictive_planner import PlanResult, PredictivePlanner
from crbot.replay_learning import ReplayPolicyModel, train_replay_policy
from tests import test_prediction as prediction
from tests import test_replay as replay


def examples(prefix, n=16):
    return [SimpleNamespace(group_id=f"{prefix}{i}", card_id="knight",
                            deploy_point=[.25 if i % 2 else .75, .65], target=i % 2)
            for i in range(n)]


def feature(a):
    return np.zeros(4, dtype=np.float32)


class ActionValueTests(unittest.TestCase):
    def test_position_learns_different_outcomes_for_same_state_and_card(self):
        head = fit(examples("train"), feature)
        self.assertEqual(head.predict("knight", feature(None), [.25, .65])["value"], 1)
        self.assertEqual(head.predict("knight", feature(None), [.75, .65])["value"], 0)
        self.assertIsNone(head.predict("knight", feature(None), [.5, .2]))
        self.assertIsNone(head.predict("knight", np.ones(4) * 10, [.25, .65]))
        self.assertIsNone(head.predict("unknown", feature(None), [.25, .65]))

    def test_repeated_actions_do_not_replace_independent_battles(self):
        a = examples("one", 1)[0]
        head = fit([a] * 100, feature)
        self.assertIsNone(head.predict("knight", feature(a), a.deploy_point))

    def test_whole_battle_evaluation_and_temporal_gate(self):
        train, validation = examples("train"), examples("validation")
        head, report = train_head(train + validation, train, validation, train, validation,
                                  feature, require_temporal=True)
        self.assertTrue(head.enabled)
        self.assertEqual(report["random"]["brier"], 0)
        self.assertGreater(report["random"]["prior_brier"], 0)
        _, report = train_head(train, train, validation, [], [], feature, require_temporal=True)
        self.assertFalse(report["enabled"])
        with self.assertRaisesRegex(ValueError, "leakage"):
            evaluate(train, train, feature)

    def test_small_or_unsupported_validation_does_not_admit_head(self):
        self.assertFalse(evaluate(examples("t"), examples("v", 4), feature)["passed"])
        train = examples("t")
        for a in train:
            a.deploy_point = [.5, .1]
        self.assertFalse(evaluate(train, examples("v"), feature)["passed"])

    def test_roundtrip_and_invalid_artifact(self):
        head = fit(examples("t"), feature)
        head.enabled = True
        loaded = ActionValueHead.load(head.arrays())
        self.assertTrue(loaded.enabled)
        self.assertEqual(loaded.predict("knight", feature(None), [.25, .65])["value"], 1)
        self.assertIsNone(ActionValueHead.load({}))
        arrays = head.arrays()
        arrays["action_value_xy"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            ActionValueHead.load(arrays)

    def test_production_training_saves_head_without_promoting_candidate(self):
        fixtures = replay.ReplayPolicyLearningTests()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            catalog = fixtures._dataset(root)
            result = train_replay_policy(root, catalog, fixtures._config(True), candidate_only=True)["candidate"]
            self.assertFalse(result["promoted"])
            report = result["manifest"]["action_value"]
            self.assertFalse(report["enabled"])
            self.assertEqual(set(report["fit_battles"]),
                             set(result["manifest"]["train_battles"] + result["manifest"]["validation_battles"]))
            paths = [p for p in (root / "models/replay_policy").rglob("*.npz") if "temporary" not in p.parts]
            self.assertTrue(paths)
            with np.load(paths[0]) as data:
                self.assertIsNotNone(ActionValueHead.load(data))

    def test_runtime_champion_and_zero_influence(self):
        model = ReplayPolicyModel.__new__(ReplayPolicyModel)
        model.champion = {"version": "test-champion"}
        model.available, model.load_error, model.action_value_error = True, None, ""
        model.action_value_head = fit(examples("train"), feature)
        model.action_value_head.enabled = True
        model.influence_scale = .1
        model._feature = lambda *args: feature(None)
        scorer, status = model.action_scorer({"knight": object()}, 7, {}, None, 20)
        self.assertEqual(status["model_version"], "test-champion")
        self.assertAlmostEqual(scorer(dict(card_id="knight", x=.25, y=.65))["delta"], .1)
        model.influence_scale = 0
        self.assertIsNone(model.action_scorer({}, 7, {}, None, 20)[0])
        model.influence_scale = .1
        model.available = False
        self.assertIsNone(model.action_scorer({}, 7, {}, None, 20)[0])


class PlannerLearningTests(unittest.TestCase):
    def setUp(self):
        self.planner = PredictivePlanner(prediction.knowledge(), {"fast_defense": True, "positions_per_card": 2})
        self.planner.clock = lambda: 0.

    def result(self):
        return PlanResult(1, 0, 1, "ready", candidates=[
            dict(action=dict(card_id="knight", x=.25, y=.65), score=1.),
            dict(action=dict(card_id="knight", x=.75, y=.65), score=1.)])

    def test_timeout_discards_all_partial_corrections(self):
        result = self.result()
        before = copy.deepcopy(result.candidates)
        timer = [0.]
        self.planner.clock = lambda: timer[0]
        def score(action):
            timer[0] += .6
            return {"delta": .1}
        self.planner.apply_learning(result, score, 1)
        self.assertEqual(result.candidates, before)
        self.assertFalse(result.learning["applied"])
        self.assertEqual(result.learning["reason"], "learning_budget_exhausted")

    def test_invalid_and_disabled_scores_preserve_rows(self):
        for scorer in (None, lambda a: {"delta": float("nan")}, lambda a: None):
            result = self.result()
            before = copy.deepcopy(result.candidates)
            self.planner.apply_learning(result, scorer, 1)
            self.assertEqual(result.candidates, before)

    def test_real_planner_applies_bounded_score_and_records_changed_selection(self):
        scene = prediction.world(tracks=(), hand=((0, "knight"),))
        original = self.planner.sim.evaluate
        def equal_scores(state):
            _, parts = original(state)
            return 0., parts
        with patch.object(self.planner.sim, "evaluate", side_effect=equal_scores):
            baseline = self.planner.plan(scene)
            learned = self.planner.plan(scene, action_scorer=lambda a: {"delta": 100, "value": 1})
        self.assertIsNone(baseline.action.card_id)
        self.assertEqual(learned.action.card_id, "knight")
        self.assertTrue(learned.learning["changed_selection"])
        self.assertFalse(learned.learning["selected_for_execution"])
        self.assertTrue(all(abs(r["score"] - r.get("simulation_score", r["score"])) <= 1
                            for r in learned.candidates))

    def test_tower_and_cost_guard_still_overrides_learned_preference(self):
        scene = prediction.world(hand=((0, "knight"),))
        with patch.object(self.planner.sim, "evaluate", return_value=(0., dict(
                own_tower_damage=0, own_towers_remaining=2, imminent_tower_exposure=0))):
            learned = self.planner.plan(scene, action_scorer=lambda a: {"delta": 1})
        self.assertIsNone(learned.action.card_id)
        self.assertFalse(learned.learning["changed_selection"])


class RouterLearningTests(unittest.TestCase):
    setUp = prediction.RouterTests.setUp
    result = prediction.RouterTests.result

    def test_execution_and_shadow_participation_are_distinct(self):
        result = self.result()
        result.learning = dict(applied=True, selected_for_execution=False)
        with patch.object(self.p.planner, "plan", return_value=result):
            decision = self.p.decide(self.image, None, now=100)
        self.assertTrue(decision.replay_learning_used)
        self.assertTrue(self.p.last_plan["learning"]["selected_for_execution"])
        from crbot.replay import action_from_payload
        recorded = action_from_payload({**decision.to_dict(), "action_status": "confirmed"})
        self.assertTrue(recorded["learned_action_value"]["selected_for_execution"])
        self.assertEqual(recorded["action_status"], "confirmed")
        self.p.next_action_at = 0
        self.p.decision_engine = "shadow"
        with patch.object(self.p.planner, "plan", return_value=result), patch.object(prediction.BattlePolicy, "decide", return_value=None):
            self.p.decide(self.image, None, now=100)
        self.assertFalse(self.p.last_plan["learning"]["selected_for_execution"])

    def test_explicit_disable_never_calls_model(self):
        self.p.planning_config["learned_action_value"] = False
        self.p.replay_model = SimpleNamespace(action_scorer=lambda *args: self.fail("disabled scorer called"))
        with patch.object(self.p.planner, "plan", return_value=self.result()) as plan:
            self.p.decide(self.image, None, now=100)
        self.assertIsNone(plan.call_args.kwargs["action_scorer"])
