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
