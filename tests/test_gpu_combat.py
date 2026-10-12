import time
import unittest
from dataclasses import replace
from pathlib import Path

from crbot.battle_simulation import Simulator, SimState, SimAction
from crbot.gpu_combat import GPUCombat
from crbot.knowledge import KnowledgeBase

ROOT=Path(__file__).resolve().parents[1]


class GPUCombatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim=Simulator(KnowledgeBase.load(ROOT/'data/battle_knowledge.json'))
        cls.sim.grid_navigation=True
        cls.gpu=GPUCombat(cls.sim)

    def require_gpu(self):
        if not self.gpu.available:self.skipTest(self.gpu.error or 'CUDA unavailable')

    def assert_state(self,a,b):
        self.assertAlmostEqual(a.time,b.time,places=9)
        self.assertEqual([e.uid for e in a.entities],[e.uid for e in b.entities])
        for x,y in zip(a.entities,b.entities):
            for field in ('x','y','hp','shield','ready_at','walked','locked_at','stunned_until'):
                self.assertAlmostEqual(getattr(x,field),getattr(y,field),places=7,msg=f'{x.spec.name}.{field}')
            self.assertEqual(x.target,y.target)
            self.assertEqual(x.winding_target,y.winding_target)
            self.assertEqual(x.active,y.active)
        for side in (1,-1):
            self.assertAlmostEqual(a.elixir[side],b.elixir[side],places=9)
            self.assertAlmostEqual(a.damage[side],b.damage[side],places=7)
            self.assertEqual(a.tower_shots[side],b.tower_shots[side])
        self.assertEqual(len(a.impacts),len(b.impacts))
        for x,y in zip(a.impacts,b.impacts):
            for v,w in zip(x,y):
                if v is None:self.assertIsNone(w)
                else:self.assertAlmostEqual(float(v),float(w),places=7)

    def scene(self,card='knight',x=3.,y=21.,enemy='giant'):
        s=SimState()
        self.sim.add(s,self.sim.kb.unit('PrincessTower',11),1,3,28,tower=True)
        self.sim.add(s,self.sim.kb.unit('PrincessTower',11),-1,3,4,tower=True)
        self.sim.add(s,self.sim.kb.roster(card,11)[0][0],1,x,y,deployed=True)
        self.sim.add(s,self.sim.kb.roster(enemy,11)[0][0],-1,3,12)
        return s

    def test_supported_full_horizon_matches_reference(self):
        self.require_gpu()
        scenes=[self.scene(c,x,y,e) for c,x,y,e in (
            ('knight',3.,21.,'giant'),('musketeer',7.,20.,'hog_rider'),
            ('baby_dragon',14.,23.,'knight'),('cannon',3.,20.,'giant'),
            ('dark_prince',3.,20.,'knight'),('inferno_tower',7.,20.,'giant'))]
        scenes=[s for s in scenes if self.gpu.supports(s)]
        self.assertGreaterEqual(len(scenes),5)
        references=[s.clone() for s in scenes]
        for s in references:self.sim.advance(s,8)
        self.gpu.advance(scenes,8,time.perf_counter()+20)
        for a,b in zip(references,scenes):self.assert_state(a,b)

    def test_spell_shield_control_and_death_matches_reference(self):
        self.require_gpu()
        a=self.scene('dark_prince',.1,.1,'knight')
        a.impacts=[(.25,-1,None,.1,.1,1500.,4.,True,.3,1.,2.),(.5,1,None,3.,12.,200.,4.,True,1.)]
        b=a.clone();self.sim.advance(a,3)
        self.gpu.advance([b],3,time.perf_counter()+10)
        self.assert_state(a,b)

    def test_all_known_mechanics_supported_unknown_effect_not_simplified(self):
        state=SimState()
        self.sim.add(state,self.sim.kb.roster('witch',11)[0][0],1,3,20)
        self.assertTrue(self.gpu.supports(state))
        state=SimState(effects=[{'pattern':'unit_pulse'}])
        self.assertTrue(self.gpu.supports(state))
        state.effects=[{'pattern':'future_unimplemented_mechanism'}]
        self.assertFalse(self.gpu.supports(state))

    def test_many_positions_and_target_locks_match_reference(self):
        self.require_gpu()
        import random
        rng=random.Random(731)
        scenes=[]
        cards=['knight','musketeer','baby_dragon','cannon','dark_prince','inferno_tower']
        for i in range(48):
            s=self.scene(cards[i%len(cards)],rng.uniform(1,17),rng.uniform(17,28),
                         ['giant','hog_rider','knight'][i%3])
            s.entities[-1].x=rng.uniform(1,17)
            s.entities[-1].y=rng.uniform(7,17)
            if i%4==0:s.entities[-1].target=s.entities[0].uid
            if self.gpu.supports(s):scenes.append(s)
        refs=[s.clone() for s in scenes]
        for s in refs:self.sim.advance(s,6.)
        self.gpu.advance(scenes,6.,time.perf_counter()+20)
        for a,b in zip(refs,scenes):self.assert_state(a,b)

    def test_expired_deadline_leaves_input_unchanged(self):
        self.require_gpu()
        a=self.scene();b=a.clone()
        with self.assertRaises(TimeoutError):self.gpu.advance([a],6,time.perf_counter()-1)
        self.assert_state(a,b)

    def test_independent_branches_and_segmented_calls(self):
        self.require_gpu()
        a=self.scene('musketeer');b=self.scene('cannon',8,20)
        refs=[a.clone(),b.clone()]
        for duration in (.35,1.5,4.15):
            for state in refs:self.sim.advance(state,duration)
            self.gpu.advance([a,b],duration,time.perf_counter()+10)
        for x,y in zip(refs,[a,b]):self.assert_state(x,y)

    def test_complete_root_scores_and_responses_match_reference(self):
        self.require_gpu()
        from tests.test_prediction import world
        from crbot.predictive_planner import PredictivePlanner
        planner=PredictivePlanner(self.sim.kb,{'grid_world':True,'fast_defense':True,'defense_pipeline_s':.35})
        gpu=GPUCombat(planner.sim)
        self.assertTrue(gpu.available,gpu.error)
        w=world(elixir=10.)
        scenarios=[(0.,.65,False),(4.,.85,True),(7.,1.,True)]
        states=[planner.initial(w,c,h) for c,h,_ in scenarios]
        roots=[SimAction(),SimAction('knight',0,.3,.65),SimAction('musketeer',1,.7,.65),
               SimAction('cannon',2,.5,.6),SimAction('fireball',3,.3,.5)]
        expected=[planner.evaluate_root(w,r,scenarios,6.,'legacy',time.perf_counter()+20,states) for r in roots]
        actual=list(gpu.rows(planner,w,roots,scenarios,6.,'legacy',time.perf_counter()+20,states))
        self.assertEqual({r['label']:r for r in actual if r},{r['label']:r for r in expected if r})
        self.assertGreater(gpu.completed,0)

    def test_special_root_uses_complete_cuda_and_matches_reference(self):
        self.require_gpu()
        from tests.test_prediction import world
        from crbot.predictive_planner import PredictivePlanner
        planner=PredictivePlanner(self.sim.kb,{'grid_world':True,'fast_defense':True})
        gpu=GPUCombat(planner.sim)
        w=world(elixir=10.,hand=((0,'witch'),))
        scenarios=[(7.,1.,False)];states=[planner.initial(w,7.,1.)]
        root=SimAction('witch',0,.3,.7)
        expected=planner.evaluate_root(w,root,scenarios,6.,'legacy',time.perf_counter()+20,states)
        rows=list(gpu.rows(planner,w,[root],scenarios,6.,'legacy',time.perf_counter()+20,states))
        self.assertEqual(rows,[expected]);self.assertEqual(gpu.completed,1);self.assertEqual(gpu.fallbacks,0)

    def test_planner_uses_real_cuda_and_keeps_action_and_fair_coverage(self):
        self.require_gpu()
        from tests.test_prediction import world
        from crbot.predictive_planner import PredictivePlanner
        config={'grid_world':True,'fast_defense':True,'budget_ms':2000,'positions_per_card':4,
                'defense_pipeline_s':.35,'parallel_combat':False,'gpu_placement':False}
        cpu=PredictivePlanner(self.sim.kb,config)
        gpu=PredictivePlanner(self.sim.kb,{**config,'gpu_combat':True})
        a=cpu.plan(world(elixir=10.));b=gpu.plan(world(elixir=10.))
        self.assertEqual(a.action,b.action)
        self.assertEqual({r['label']:r['score'] for r in a.candidates},
                         {r['label']:r['score'] for r in b.candidates})
        self.assertGreater(b.compute['gpu_combat']['frame_gpu_roots'],4)
        self.assertEqual(b.compute['combat_device'],'cuda:0')
        self.assertTrue(all(e['evaluated']>1 for e in b.hand_evaluations))

    def test_delayed_death_damage_and_tower_multiplier_match_reference(self):
        self.require_gpu()
        from dataclasses import replace
        state=self.scene(enemy='balloon')
        balloon=state.entities[-1];balloon.x,balloon.y=3.,27.;balloon.hp=1.
        balloon.spec=replace(balloon.spec,death_tower_multiplier=.35)
        state.impacts=[(.25,1,balloon.uid,3.,27.,100.,0.,True,1.)]
        expected=state.clone()
        self.assertTrue(self.gpu.supports(state))
        self.sim.advance(expected,6.)
        self.gpu.advance([state],6.,time.perf_counter()+20)
        self.assert_state(expected,state)
        self.assertGreater(state.damage[1],0)

    def test_chain_and_multiple_bolts_match_reference(self):
        self.require_gpu()
        states=[]
        for card in ('electro_spirit','electro_wizard'):
            state=self.scene(card,3.,21.,'knight')
            state.entities[-1].x,state.entities[-1].y=3.,20.
            for x,y in ((4.,20.),(5.,20.),(5.5,21.)):
                self.sim.add(state,self.sim.kb.roster('knight')[0][0],-1,x,y)
            states.append(state)
        refs=[s.clone() for s in states]
        self.assertTrue(all(self.gpu.supports(s) for s in states))
        for s in refs:self.sim.advance(s,5.)
        self.gpu.advance(states,5.,time.perf_counter()+20)
        for a,b in zip(refs,states):
            self.assert_state(a,b)
            self.assertEqual(a.uncertainties,b.uncertainties)

    def test_wait_only_log_explains_unaffordable_hand(self):
        from tests.test_prediction import world
        from crbot.predictive_planner import PredictivePlanner
        planner=PredictivePlanner(self.sim.kb,{'fast_defense':True,'budget_ms':2000})
        result=planner.plan(world(elixir=1.4,hand=((0,'mega_knight'),(1,'goblin_giant'),
                                                  (2,'elite_barbarians'),(3,'valkyrie'))))
        self.assertEqual(len(result.candidates),1)
        self.assertEqual(result.compute['affordable_card_count'],0)
        self.assertEqual(result.compute['planned_hand_count'],4)
        self.assertIn('1.4',result.reason)
        self.assertIn('最低 4',result.reason)
        self.assertIn('无可支付出牌候选',result.reason)


if __name__=='__main__':unittest.main()
