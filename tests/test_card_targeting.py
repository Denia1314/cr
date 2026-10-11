"""Public card properties must constrain choices and executable attacks."""
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from crbot.battle_perception import LaneThreat
from crbot.battle_simulation import Simulator, SimState
from crbot.card_matchup import matchup
from crbot.cards import CardCatalog
from crbot.emergency_defense import emergency_defense
from crbot.grid_world import target_for
from crbot.knowledge import KnowledgeBase
from crbot.policy import BattlePolicy
from crbot.tactics import card_tactics
from tests.test_tactics import formation_config, FakeRecognizer

ROOT = Path(__file__).resolve().parents[1]


class CardTargetingTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase.load(ROOT / 'data/battle_knowledge.json')
        self.sim = Simulator(self.kb)
        self.catalog = CardCatalog.load(ROOT / 'data/cards.json')

    def test_ice_golem_cannot_attack_or_pull_hog(self):
        state = SimState()
        hog = self.sim.add(state, self.kb.roster('hog_rider')[0][0], -1, 4, 24)
        ice = self.sim.add(state, self.kb.roster('ice_golem')[0][0], 1, 4, 25)
        tower = self.sim.add(state, self.kb.unit('PrincessTower'), 1, 4, 28, tower=True)
        comparison = matchup(ice.spec, hog)
        self.assertFalse(comparison['can_hit'])
        self.assertFalse(comparison['can_be_targeted'])
        self.assertIs(target_for(self.sim, state, hog), tower)
        state.hands[1], state.elixir[1] = [(0, 'ice_golem')], 10
        self.assertIsNone(emergency_defense(self.sim, state)[0])
        state.hands[1].append((1, 'mini_pekka'))
        self.assertEqual(emergency_defense(self.sim, state)[0].card_id, 'mini_pekka')

    def test_rule_policy_rejects_ice_golem_even_without_layer_classifier(self):
        policy = BattlePolicy(formation_config())
        policy.catalog = self.catalog
        policy.hand_recognizer = FakeRecognizer(['ice_golem', 'mini_pekka'])
        policy.imitation_model = policy.replay_model = None
        policy.reset_battle(now=0)
        policy.next_action_at = 0
        threats = {'left': LaneThreat('left', .8, 1, .6, 'heavy', ((.3, .6),), ('hog_rider',)),
                   'right': LaneThreat('right', 0, 0, 0, 'none', ())}
        self.assertEqual(policy._defense_score(self.catalog.get('ice_golem'), threats['left']), -1000)
        with patch('crbot.policy.detect_lane_threats', return_value=threats):
            decision = policy.decide(Image.new('RGB', (600, 1000)), None, now=1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.card_id, 'mini_pekka')

    def test_completed_planner_chooses_attacking_defender_against_hog(self):
        from crbot.battle_world import Track
        from crbot.predictive_planner import PredictivePlanner
        from tests.test_prediction import world
        planner = PredictivePlanner(self.kb, {'fast_defense': True, 'budget_ms': 2000})
        result = planner.plan(world(hand=((0, 'ice_golem'), (1, 'mini_pekka')), elixir=10,
            tracks=(Track(1, 'hog_rider', -1, .28, .71, 99, 100, .9),)))
        self.assertEqual(result.action.card_id, 'mini_pekka')
        self.assertTrue(result.candidates)
        self.assertFalse(result.compute.get('timeout_tower_defense'))

    def test_explicit_area_attack_changes_damage_to_compact_group(self):
        def damaged(splash):
            state = SimState()
            spec = self.kb.roster('valkyrie')[0][0]
            self.sim.add(state, replace(spec, splash=splash, speed=0), 1, 5, 20)
            victims = [self.sim.add(state, replace(self.kb.unit('Knight'), speed=0, damage=0),
                                   -1, 5.9, y) for y in (19.8, 20, 20.2)]
            self.sim.advance(state, .75)
            return sum(e.hp < e.spec.hp for e in victims)
        self.assertEqual(damaged(0), 1)
        self.assertEqual(damaged(self.kb.roster('valkyrie')[0][0].splash), 3)
        # Removing prose/role hints must not remove factual targeting or area damage.
        valkyrie = replace(self.catalog.get('valkyrie'), roles=(), notes='', description='', targets=())
        profile = card_tactics(valkyrie)
        self.assertEqual(profile.attack_targets, ('ground',))
        self.assertEqual(profile.splash_strength, 1)

    def test_death_damage_uses_separate_crown_tower_value(self):
        state = SimState()
        giant = self.sim.add(state, self.kb.roster('giant_skeleton')[0][0], 1, 5, 20)
        tower = self.sim.add(state, replace(self.kb.unit('PrincessTower'), damage=0), -1, 5, 20, tower=True)
        troop = self.sim.add(state, replace(self.kb.unit('Knight'), damage=0, speed=0), -1, 5, 20)
        giant.hp = 0
        self.sim.advance(state, 3)
        self.assertEqual(troop.hp, troop.spec.hp)
        self.sim.advance(state, .5)
        self.assertEqual(troop.spec.hp-troop.hp, giant.spec.death_damage)
        self.assertAlmostEqual(tower.spec.hp-tower.hp, 1068)


if __name__ == '__main__':
    unittest.main()
