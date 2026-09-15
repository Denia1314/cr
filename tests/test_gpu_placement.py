import unittest
import numpy as np
from crbot.gpu_placement import PlacementBatch
from crbot.predictive_planner import PredictivePlanner
from tests.test_prediction import knowledge, world


class PlacementBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = knowledge()

    def test_cpu_cuda_rank_parity_and_real_planner_calls(self):
        try:
            import torch
            available = torch.cuda.is_available()
        except ImportError:
            available = False
        if not available:
            self.skipTest("CUDA not available")
        cpu = PredictivePlanner(self.kb,dict(gpu_placement=True,compute_device="cpu",fast_defense=True,budget_ms=2000))
        gpu = PredictivePlanner(self.kb,dict(gpu_placement=True,compute_device="cuda",fast_defense=True,budget_ms=2000))
        a,b = cpu.plan(world()),gpu.plan(world())
        self.assertEqual(a.action,b.action)
        self.assertEqual([c['label'] for c in a.candidates],[c['label'] for c in b.candidates])
        np.testing.assert_allclose([c['score'] for c in a.candidates],[c['score'] for c in b.candidates],atol=1e-4)
        self.assertGreater(b.compute['calls'],0)
        self.assertGreater(b.compute['positions'],500)
        self.assertTrue(b.compute['device'].startswith('cuda'))
        self.assertFalse(b.budget_exhausted)

    def test_vector_rank_matches_scalar_shortlist(self):
        from crbot.placement_search import placement_points
        p = PredictivePlanner(self.kb,{})
        state = p.initial(world(),7,1)
        batch = PlacementBatch('cpu')
        # Capture the full dense-grid scalar scores through a no-op batch, then
        # compare ranking of that same grid using the vector kernel.
        class ScalarBatch:
            def score(self,*args): return None
        p.sim.placement_batch = ScalarBatch()
        expected = placement_points(p.sim,state,'musketeer',1,limit=24)
        p.sim.placement_batch = batch
        actual = placement_points(p.sim,state,'musketeer',1,limit=24)
        self.assertEqual(expected,actual)

    def test_endangered_tower_is_saved_instead_of_choosing_cheap_loss(self):
        from dataclasses import replace
        w = world()
        w = replace(w,tower_health=(.03,1,1,1),tracks=(replace(w.tracks[0],card_id='hog_rider',y=.65,hp_fraction=1),))
        p = PredictivePlanner(self.kb,dict(gpu_placement=True,compute_device='cpu',fast_defense=True,
                                         budget_ms=2000,positions_per_card=24,defense_damage_tolerance=5))
        result = p.plan(w)
        chosen = next(c for c in result.candidates if c['label']==result.action.label)
        waiting = next(c for c in result.candidates if c['label']=='WAIT')
        self.assertEqual(min(b['own_towers_remaining'] for b in chosen['branches']),2)
        self.assertLess(min(b['own_towers_remaining'] for b in waiting['branches']),2)

    def test_trajectory_cpu_cuda_parity(self):
        try:
            import torch
            if not torch.cuda.is_available():self.skipTest('CUDA not available')
        except ImportError:self.skipTest('CUDA not available')
        from crbot.placement_search import placement_points
        p=PredictivePlanner(self.kb,{})
        state=p.initial(world(),7,1)
        p.sim.placement_batch=PlacementBatch('cpu',trajectory=True)
        expected=placement_points(p.sim,state,'knight',1,limit=None)
        p.sim.placement_batch=PlacementBatch('cuda',trajectory=True)
        actual=placement_points(p.sim,state,'knight',1,limit=None)
        self.assertEqual(expected,actual)
        self.assertEqual(p.sim.placement_batch.status()['trajectory_steps'],33)


class ExpandedGpuPlacementTests(unittest.TestCase):
    def setUp(self):
        self.p=PredictivePlanner(knowledge(),{})
        self.s=self.p.initial(world(),7,1)

    def devices(self):
        yield 'cpu'
        try:
            import torch
            if torch.cuda.is_available():
                yield 'cuda'
        except ImportError:
            pass

    def test_legal_masks_match_scalar_for_sides_spells_and_custom_arena(self):
        from crbot.arena_geometry import ArenaGeometry
        from crbot.placement_domain import deployment_domain
        for geometry in (ArenaGeometry(),ArenaGeometry.from_config(dict(bounds=[.075,.097,.925,.757],tower_points=[[.24,.625],[.76,.625],[.24,.219],[.76,.219]]))):
            self.p.sim.geometry=geometry
            for side in (1,-1):
                for cid in ('knight','cannon','fireball','royal_delivery'):
                    self.p.sim.placement_batch=None
                    expected=deployment_domain(self.p.sim,self.s,cid,side,step=.5)
                    for device in self.devices():
                        self.p.sim.placement_batch=PlacementBatch(device)
                        self.assertEqual(deployment_domain(self.p.sim,self.s,cid,side,step=.5),expected)

    def test_spell_shields_air_mask_and_quiet_board_parity(self):
        import math
        from dataclasses import replace
        enemy=next(e for e in self.s.entities if not e.tower)
        enemy.hp=20;enemy.shield=200
        air=replace(enemy,spec=replace(enemy.spec,air=True),x=12,uid=99)
        enemies=[enemy,air];projected=[(e.x,e.y) for e in enemies]
        points=[(x,y) for x in range(19) for y in range(33)]
        spell=dict(radius=2,damage=100,air=False)
        expected=[sum(min(e.hp+e.shield,100)*(1+e.value) for e,q in zip(enemies,projected)
                      if not e.spec.air and math.dist(p,q)<=2+e.spec.radius) for p in points]
        towers=[e for e in self.s.entities if e.tower and e.side==1]
        quiet=[-.2*min(math.dist(p,(t.x,t.y-2)) for t in towers) for p in points]
        for device in self.devices():
            b=PlacementBatch(device)
            np.testing.assert_allclose(b.spell_score(points,enemies,projected,spell),expected,atol=1e-9)
            np.testing.assert_allclose(b.quiet_score(points,towers,1),quiet,atol=1e-9)
            self.assertEqual(b.status()['spell_positions'],len(points))

    def test_dormant_tower_provides_no_fire_and_building_only_cannot_hit_troops(self):
        from dataclasses import replace
        enemy=next(e for e in self.s.entities if not e.tower)
        spec=self.p.kb.roster('giant',11)[0][0]
        tower=next(e for e in self.s.entities if e.tower and e.side==1)
        tower=replace(tower,x=enemy.x,y=enemy.y,active=False)
        points=[(enemy.x,enemy.y),(enemy.x+2,enemy.y)]
        projected=[(enemy,(enemy.x,enemy.y))]
        for device in self.devices():
            b=PlacementBatch(device,trajectory=True)
            inactive=b.score(points,spec,projected,[tower])
            zero_fire=b.score(points,spec,projected,[replace(tower,spec=replace(tower.spec,damage=0),active=True)])
            np.testing.assert_allclose(inactive,zero_fire,atol=1e-9)
            harmless=b.score(points,replace(spec,damage=0),projected,[tower])
            np.testing.assert_allclose(inactive,harmless,atol=1e-9)

    def test_cuda_failure_uses_same_cpu_domain(self):
        from unittest.mock import patch
        b=PlacementBatch('cpu')
        expected=b.legal_points(self.p.sim,self.s,'knight',1,[(.3,.6),(.3,.2)])
        arrays=b._arrays
        with patch.object(b,'_arrays',side_effect=[RuntimeError('device lost'),arrays()]):
            self.assertEqual(b.legal_points(self.p.sim,self.s,'knight',1,[(.3,.6),(.3,.2)]),expected)
        self.assertEqual(b.device,'cpu')
        self.assertEqual(b.error,'device lost')
