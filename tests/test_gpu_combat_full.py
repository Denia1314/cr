import time
import unittest
from dataclasses import replace
from crbot.battle_simulation import SimState, SimAction
from crbot.gpu_combat import GPUCombat
from tests import test_gpu_combat as reference


class FullCombatGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):reference.GPUCombatTests.setUpClass.__func__(cls)
    require_gpu=reference.GPUCombatTests.require_gpu
    scene=reference.GPUCombatTests.scene
    assert_state=reference.GPUCombatTests.assert_state
    def compare(self,states,duration):
        self.require_gpu()
        refs=[s.clone() for s in states]
        for s in refs:self.sim.advance(s,duration)
        self.gpu.advance(states,duration,time.perf_counter()+30)
        for a,b in zip(refs,states):
            self.assert_state(a,b)
            self.assertEqual(a.next_uid,b.next_uid)
            self.assertEqual(a.uncertainties,b.uncertainties)
            self.assertEqual(len(a.effects),len(b.effects))
            for x,y in zip(a.entities,b.entities):
                for f in ('spawn_at','resource_at','dash_phase','dash_target','dash_at','dash_ready_at','dash_x','dash_y','cloned'):
                    self.assertEqual(getattr(x,f),getattr(y,f),f)
            for x,y in zip(a.effects,b.effects):
                self.assertEqual(x['pattern'],y['pattern'])
                for f in x:
                    if isinstance(x[f],float):self.assertAlmostEqual(x[f],y[f],places=7,msg=f)
                    else:self.assertEqual(x[f],y[f],f)

    def test_periodic_and_death_spawn_resources(self):
        states=[]
        for card in ('witch','night_witch','goblin_giant','golem','elixir_golem','elixir_collector','goblin_cage','tombstone'):
            s=self.scene(card,6,21,'knight')
            if card in ('goblin_giant','golem','elixir_golem','goblin_cage','tombstone'):
                e=s.entities[-2];s.impacts.append((.5,-1,e.uid,e.x,e.y,100000.,0.,True,1.))
            states.append(s)
        self.compare(states,12.)

    def test_dash_flight_immunity_and_reflection(self):
        states=[]
        for card in ('mega_knight','bandit','electro_giant'):
            s=self.scene(card,5,21,'knight')
            enemy=s.entities[-1];enemy.x,enemy.y=9.,21.
            enemy.spec=replace(enemy.spec,hp=10000,speed=0,damage=100,reach=8,first_hit=0,period=.2)
            enemy.hp=10000
            states.append(s)
        self.compare(states,5.)

    def test_all_spell_patterns_and_pending_events(self):
        states=[]
        for card in ('arrows','lightning','poison','the_log','barbarian_barrel','royal_delivery',
                     'goblin_barrel','graveyard','rage','clone','tornado','earthquake'):
            s=SimState(elixir={1:10.,-1:10.},hands={1:[(0,card)],-1:[]})
            base=replace(self.sim.kb.unit('Knight'),hp=10000,speed=0,damage=0,shield=1.)
            for side in (1,-1):
                for x,y in ((5.,20.),(5.,18.),(7.,20.)):
                    self.sim.add(s,base,side,x,y)
            self.sim.add(s,self.sim.kb.unit('Cannon'),-1,5,21)
            x,y=self.sim.screen(5.,22.)
            self.assertTrue(self.sim.apply(s,SimAction(card,0,x,y),1))
            states.append(s)
        self.compare(states,2.75)
        self.compare(states,6.)

    def test_cloned_spawner_death_children_and_entity_budget(self):
        s=self.scene('witch',5,22,'knight')
        e=s.entities[-2];e.cloned=True;e.hp=1.;e.shield=1.
        s.impacts.append((3.,-1,e.uid,e.x,e.y,10000.,0.,True,1.))
        crowded=SimState()
        spec=replace(self.sim.kb.unit('Knight'),speed=0,damage=0)
        for i in range(94):self.sim.add(crowded,spec,1,5+i*.001,22)
        witch=self.sim.add(crowded,self.sim.kb.roster('witch')[0][0],1,6,22)
        witch.spawn_at=0
        self.compare([s,crowded],4.)

    def test_every_current_simulatable_card_has_cuda_coverage(self):
        states=[]
        for card,data in self.sim.kb.cards.items():
            if data.get('elixir') is None or not (self.sim.kb.roster(card) or self.sim.kb.spell(card)):continue
            s=SimState(elixir={1:10.,-1:10.},hands={1:[(0,card)],-1:[]})
            target=replace(self.sim.kb.unit('Knight'),hp=10000,speed=0,damage=0)
            self.sim.add(s,target,-1,5.,21.)
            x,y=self.sim.screen(5.,22.)
            if not self.sim.apply(s,SimAction(card,0,x,y),1):continue
            self.assertTrue(self.gpu.supports(s),card)
            states.append(s)
        self.assertGreaterEqual(len(states),119)
        self.compare(states,4.)

    def test_dash_actual_windup_flight_and_stun_interruption(self):
        for card in ('mega_knight','bandit'):
            s=SimState()
            u=replace(self.sim.kb.roster(card)[0][0],damage=0,speed=0)
            self.sim.add(s,u,1,5.,20.)
            target=replace(self.sim.kb.unit('Knight'),hp=10000,damage=0,speed=0,radius=.3)
            self.sim.add(s,target,-1,9+u.radius+.3,20.)
            self.compare([s],.3)
            self.assertEqual(s.entities[0].dash_phase,'windup')
            self.compare([s],s.entities[0].dash_at-s.time+.01)
            self.assertEqual(s.entities[0].dash_phase,'flight')
            s.impacts.append((s.time,-1,s.entities[0].uid,5.,20.,100.,0.,True,1.,.2,0.))
            self.compare([s],2.)

    def test_conditional_combinations_match_reference_for_all_tactical_phases(self):
        self.require_gpu()
        from tests.test_prediction import world
        from crbot.predictive_planner import PredictivePlanner
        planner=PredictivePlanner(self.sim.kb,{'grid_world':True,'combo_positions_per_card':2,'defense_pipeline_s':.35})
        gpu=GPUCombat(planner.sim)
        w=world(elixir=10.,hand=((0,'witch'),(1,'clone'),(2,'mega_knight'),(3,'rage')))
        roots=[SimAction(),SimAction('witch',0,.3,.7)]
        scenarios=[(0.,.65,False),(7.,1.,True)]
        for phase in ('defend','develop','counterpush','prepare'):
            expected=[planner.evaluate_combination(w,r,scenarios,6.,phase,time.perf_counter()+30) for r in roots]
            actual=list(gpu.combination_rows(planner,w,roots,scenarios,6.,phase,time.perf_counter()+30))
            self.assertEqual(actual,expected,phase)
        self.assertEqual(gpu.completed_combinations,8)

    def test_mixed_mechanics_segmented_battles_match_reference(self):
        import random
        rng=random.Random(77182)
        states=[]
        cards=('witch','night_witch','golem','elixir_golem','goblin_giant','mega_knight',
               'bandit','electro_giant','electro_dragon','electro_wizard','ice_golem','balloon')
        spells=('clone','rage','poison','lightning','tornado','earthquake')
        for n in range(12):
            s=SimState(elixir={1:10.,-1:10.},hands={1:[(0,spells[n%6])],-1:[]})
            for side in (1,-1):
                for x in (3.,15.):
                    self.sim.add(s,self.sim.kb.unit('PrincessTower'),side,x,28. if side==1 else 4.,tower=True)
                for k in range(3):
                    card=cards[(n+side+k*3)%len(cards)]
                    for spec,count in self.sim.kb.roster(card):
                        for j in range(count):self.sim.add(s,spec,side,rng.uniform(3,15),rng.uniform(17,23) if side==1 else rng.uniform(9,17),deployed=True)
            x,y=self.sim.screen(9.,20.)
            self.assertTrue(self.sim.apply(s,SimAction(spells[n%6],0,x,y),1))
            states.append(s)
        for duration in (.35,1.5,4.15):self.compare(states,duration)

    def test_cuda_failure_preserves_complete_cpu_combo_fallback(self):
        self.require_gpu()
        from unittest.mock import patch
        from tests.test_prediction import world
        from crbot.predictive_planner import PredictivePlanner
        p=PredictivePlanner(self.sim.kb,{'grid_world':True,'combo_positions_per_card':1})
        w=world(elixir=10.,hand=((0,'witch'),(1,'rage')))
        roots=[SimAction(),SimAction('witch',0,.3,.7)];scenarios=[(7.,1.,False)]
        expected=[p.evaluate_combination(w,r,scenarios,6.,'defend',time.perf_counter()+30) for r in roots]
        gpu=GPUCombat(p.sim);p.gpu_combat=gpu
        with patch.object(gpu,'advance',side_effect=RuntimeError('device lost')):
            actual=list(gpu.combination_rows(p,w,roots,scenarios,6.,'defend',time.perf_counter()+30))
        self.assertEqual(actual,expected)
        self.assertFalse(gpu.available);self.assertEqual(gpu.error,'device lost')

    def test_single_phase_device_failure_keeps_reference_state(self):
        self.require_gpu()
        from unittest.mock import patch
        from crbot.predictive_planner import PredictivePlanner
        p=PredictivePlanner(self.sim.kb,{'grid_world':True})
        s=self.scene('witch');expected=s.clone();p.sim.advance(expected,6.)
        gpu=GPUCombat(p.sim);p.gpu_combat=gpu
        with patch.object(gpu,'advance',side_effect=RuntimeError('device lost')):
            p.advance_state(s,6.,deadline=time.perf_counter()+30,clock=time.perf_counter)
        self.assert_state(s,expected)
        self.assertFalse(gpu.available)
