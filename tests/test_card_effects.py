import json
from pathlib import Path
from dataclasses import replace
import unittest

from crbot.battle_simulation import Simulator, SimState, SimAction
from crbot.card_effects import mechanism_profile
from tests.test_prediction import knowledge, world
from crbot.predictive_planner import PredictivePlanner
from crbot.battle_world import Track


class CardEffectTests(unittest.TestCase):
    def setUp(self):
        self.kb=knowledge()
        self.sim=Simulator(self.kb)

    def state(self,cid):
        return SimState(elixir={1:10,-1:0},hands={1:[(0,cid)],-1:[]})

    def target(self,s,x=5,y=20,hp=5000,air=False,shield=0,side=-1):
        return self.sim.add(s,replace(self.kb.unit('Knight'),hp=hp,speed=0,damage=0,air=air,shield=shield),side,x,y)

    def cast(self,s,cid,x=5,y=20):
        self.assertTrue(self.sim.apply(s,SimAction(cid,0,.05+x*.05,.18+y*.02),1))

    def test_lightning_selects_three_highest_hp(self):
        s=self.state('lightning')
        targets=[self.target(s,hp=hp) for hp in (5000,6000,7000,8000)]
        self.cast(s,'lightning')
        self.sim.advance(s,1)
        self.assertEqual(sum(e.hp<e.spec.hp for e in targets),3)
        self.assertEqual(targets[0].hp,5000)

    def test_arrows_are_separate_hits_and_poison_is_not_total_per_tick(self):
        s=self.state('arrows');enemy=self.target(s,shield=1)
        spell=self.kb.spell('arrows');self.cast(s,'arrows');self.sim.advance(s,1.5)
        self.assertAlmostEqual(5000-enemy.hp,spell['damage']*2)
        s=self.state('poison');enemy=self.target(s)
        spell=self.kb.spell('poison');self.cast(s,'poison');self.sim.advance(s,8.5)
        self.assertAlmostEqual(5000-enemy.hp,spell['damage']*spell['pulses'])

    def test_log_line_does_not_hit_air_or_behind(self):
        s=self.state('the_log')
        targets=[self.target(s,5,18),self.target(s,8,18),self.target(s,5,18,air=True),self.target(s,5,24)]
        self.cast(s,'the_log',5,22);self.sim.advance(s,3)
        self.assertLess(targets[0].hp,5000)
        self.assertTrue(all(e.hp==5000 for e in targets[1:]))

    def test_graveyard_is_delayed_and_spawned_over_time(self):
        s=self.state('graveyard');self.cast(s,'graveyard')
        self.sim.advance(s,2)
        self.assertEqual(len(s.entities),0)
        self.sim.advance(s,1)
        self.assertGreater(len(s.entities),0)
        self.assertLess(len(s.entities),13)
        copy=s.clone();copy.effects[0]['at']+=1
        self.assertNotEqual(copy.effects[0]['at'],s.effects[0]['at'])

    def test_clone_does_not_clone_buildings_or_other_clones(self):
        s=self.state('clone');self.target(s,side=1)
        self.sim.add(s,self.kb.unit('Cannon'),1,5,20)
        self.cast(s,'clone');self.sim.advance(s,1)
        clones=[e for e in s.entities if e.cloned]
        self.assertEqual(len(clones),1)
        self.assertEqual(clones[0].hp,1)
        self.assertFalse(clones[0].spec.building)

    def test_rage_and_pull_change_state(self):
        s=self.state('rage');ally=self.target(s,side=1);enemy=self.target(s)
        self.cast(s,'rage');self.sim.advance(s,1)
        self.assertGreater(ally.haste,1)
        self.assertLess(enemy.hp,5000)
        s=self.state('tornado');enemy=self.target(s,7,20)
        self.cast(s,'tornado');self.sim.advance(s,2)
        self.assertLess(enemy.x,7)

    def test_all_cards_have_effect_records_and_hand_positions_are_counted(self):
        data=json.loads(Path('data/card_effects.json').read_text(encoding='utf8'))
        self.assertEqual(set(data['cards']),set(self.kb.cards))
        self.assertEqual(mechanism_profile(self.kb,'poison')['pattern'],'periodic')
        planner=PredictivePlanner(self.kb,{'budget_ms':2000})
        result=planner.plan(world(hand=((0,'knight'),(1,'musketeer'),(2,'cannon'),(3,'poison'))))
        self.assertEqual(len(result.hand_evaluations),4)
        self.assertTrue(all(r['evaluated']==r['positions'] and r['evaluated']>0 for r in result.hand_evaluations))
        self.assertTrue(all(r['positions']==16 for r in result.hand_evaluations[:3]))

    def test_early_sighting_gets_a_forecast_even_on_enemy_half(self):
        result=PredictivePlanner(self.kb,{'budget_ms':2000}).plan(world(
            tracks=(Track(1,'giant',-1,.28,.25,99,100,.9,3),)))
        self.assertTrue(result.enemy_forecast)
        self.assertGreater(result.enemy_forecast[0]['unopposed_tower_eta_s'],10)
        self.assertEqual(result.enemy_forecast[0]['status'],'hypothesis')

    def test_unmapped_card_does_not_remove_other_hand_options(self):
        result=PredictivePlanner(self.kb,{'budget_ms':2000}).plan(world(
            hand=((0,'goblinstein'),(1,'knight'),(2,'musketeer'),(3,'poison'))))
        self.assertEqual(result.hand_evaluations[0]['status'],'mechanism_or_target_unavailable')
        self.assertTrue(all(e['evaluated']>0 for e in result.hand_evaluations[1:]))

    def test_collector_produces_resources_and_berserker_has_explicit_stats(self):
        s=self.state('elixir_collector');s.seconds_per_elixir=1e9
        self.cast(s,'elixir_collector',5,22)
        before=s.elixir[1];self.sim.advance(s,11)
        self.assertGreater(s.elixir[1],before+.9)
        unit=self.kb.roster('berserker',11)[0][0]
        self.assertEqual(unit.period,.6)
        self.assertGreater(unit.damage,0)


if __name__=='__main__':
    unittest.main()
