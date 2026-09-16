import unittest
from dataclasses import replace
from unittest.mock import patch

from crbot.card_matchup import matchup
from crbot.gpu_placement import PlacementBatch
from crbot.placement_search import placement_points
from crbot.predictive_planner import PredictivePlanner
from tests.test_prediction import knowledge, world


class CardMatchupTests(unittest.TestCase):
    def setUp(self):
        self.planner = PredictivePlanner(knowledge(), {})
        self.state = self.planner.initial(world(), 5, 1)
        self.enemy = next(e for e in self.state.entities if not e.tower)
        self.spec = self.planner.kb.roster('musketeer', 11)[0][0]

    def test_shields_require_separate_hits_and_overkill_is_not_dps(self):
        enemy = replace(self.enemy, hp=1, shield=1)
        a = matchup(replace(self.spec, damage=100), enemy)
        b = matchup(replace(self.spec, damage=1000), enemy)
        self.assertEqual(a['hits_to_kill'], 2)
        self.assertEqual(a['kill_s'], b['kill_s'])
        self.assertEqual(a['attack_weight'], b['attack_weight'])

    def test_attack_speed_count_and_enemy_splash_change_comparison(self):
        spec = replace(self.spec, damage=50, hp=100, shield=0, first_hit=.1)
        enemy = replace(self.enemy, hp=1000, shield=0,
                        spec=replace(self.enemy.spec, damage=100, first_hit=.5,
                                     targets=('ground',), building_only=False, splash=0))
        single = matchup(spec, enemy)
        swarm = matchup(spec, enemy, 3)
        splash = matchup(spec, replace(enemy, spec=replace(enemy.spec, splash=1)), 3)
        self.assertLess(swarm['kill_s'], single['kill_s'])
        self.assertGreater(swarm['survival_s'], splash['survival_s'])
        self.assertGreater(matchup(replace(spec, period=.1), enemy)['attack_weight'], single['attack_weight'])

    def test_target_constraints_override_high_damage(self):
        airborne = replace(self.enemy, spec=replace(self.enemy.spec, air=True))
        ground = replace(self.spec, targets=('ground',), damage=100000)
        self.assertIsNone(matchup(ground, airborne)['kill_s'])
        self.assertEqual(matchup(ground, airborne)['attack_weight'], 0)
        self.assertFalse(matchup(replace(ground, building_only=True), self.enemy)['can_hit'])

    def test_database_damage_changes_actual_position_ranking(self):
        # Same unit, positions and scene: editing the database-derived damage
        # must change the live screening values, rather than only the log.
        enemy = replace(self.enemy, hp=1000, shield=0,
                        spec=replace(self.enemy.spec, building_only=False))
        projected = [(enemy, (enemy.x, enemy.y))]
        points = [(enemy.x, enemy.y), (enemy.x+10, enemy.y)]
        batch = PlacementBatch('cpu')
        low = batch.score(points, replace(self.spec, damage=1), projected, [])
        high = batch.score(points, replace(self.spec, damage=1000), projected, [])
        self.assertGreater(high[0], low[0])
        self.assertNotAlmostEqual(high[0]-low[0], high[1]-low[1])

    def test_mixed_roster_uses_all_members_and_matches_scalar(self):
        kb = self.planner.kb
        original = kb.roster
        members = [(replace(self.spec, targets=('ground',), reach=1), 1),
                   (replace(self.spec, targets=('ground', 'air'), reach=6), 3)]
        class ScalarBatch:
            def score(self, *args):
                return None
        with patch.object(kb, 'roster', side_effect=lambda cid, level=11: members if cid == 'knight' else original(cid, level)):
            self.planner.sim.placement_batch = ScalarBatch()
            expected = placement_points(self.planner.sim, self.state, 'knight', 1, limit=24)
            batch = PlacementBatch('cpu')
            self.planner.sim.placement_batch = batch
            actual = placement_points(self.planner.sim, self.state, 'knight', 1, limit=24)
            self.assertEqual(expected, actual)
            self.assertEqual(batch.calls, 2)

    def test_damage_changes_which_enemy_position_is_preferred(self):
        spec = replace(self.spec, reach=1, hp=10000)
        small = replace(self.enemy, x=3, y=20, hp=100, shield=0, value=1)
        heavy = replace(self.enemy, x=15, y=20, hp=2000, shield=0, value=1.1)
        projected = [(small, (3, 20)), (heavy, (15, 20))]
        batch = PlacementBatch('cpu')
        low = batch.score([(3, 21), (15, 21)], replace(spec, damage=10), projected, [])
        high = batch.score([(3, 21), (15, 21)], replace(spec, damage=2000), projected, [])
        self.assertGreater(low[0], low[1])
        self.assertGreater(high[1], high[0])

    def test_decision_audit_contains_actual_database_stats(self):
        self.planner.config.update(budget_ms=2000, fast_defense=True)
        result = self.planner.plan(world())
        row = next(r for r in result.hand_evaluations if r['card_id'] == 'musketeer')
        evidence = row['database_comparison']
        self.assertEqual(evidence['knowledge_version'], self.planner.kb.version)
        self.assertEqual(evidence['units'][0]['damage'], self.spec.damage)
        self.assertTrue(evidence['units'][0]['matchups'])


if __name__ == '__main__':
    unittest.main()
