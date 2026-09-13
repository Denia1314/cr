import unittest
from dataclasses import replace

from crbot.battle_simulation import SimAction, SimState
from crbot.placement_search import placement_points, screen
from tests.test_prediction import knowledge, world
from crbot.predictive_planner import PredictivePlanner
from crbot.battle_world import Track


class PlacementSearchTests(unittest.TestCase):
    def setUp(self):
        self.planner = PredictivePlanner(knowledge(), {'budget_ms': 2000})
        self.sim = self.planner.sim

    def state(self, x=.28, y=.58, **kwargs):
        return self.planner.initial(world(tracks=(Track(1,'giant',-1,x,y,99,100,.9,3),), **kwargs), 5, 1)

    def test_full_region_and_enemy_motion_change_points(self):
        near = placement_points(self.sim, self.state(), 'knight', 1)
        moved = placement_points(self.sim, self.state(.39,.68), 'knight', 1)
        self.assertGreater(len(near), 4)
        self.assertNotEqual(near[:3], moved[:3])
        old = {(.28,.67),(.72,.67),(.28,.55),(.72,.55)}
        self.assertTrue(any(p not in old for p in near))

    def test_legality_is_shared_by_search_and_simulation(self):
        s = self.state()
        for cid in ('knight','cannon','musketeer'):
            for x,y in placement_points(self.sim,s,cid,1):
                self.assertTrue(self.sim.legal_placement(s,cid,1,x,y))
        tower = next(e for e in s.entities if e.tower and e.side == 1)
        x,y = screen(tower.x,tower.y)
        self.assertFalse(self.sim.apply(s, SimAction('cannon',2,x,y),1))
        self.assertFalse(self.sim.legal_placement(s,'knight',1,*screen(9,30)))
        self.assertFalse(self.sim.legal_placement(s,'knight',1,.3,.4))
        self.assertTrue(self.sim.legal_placement(s,'fireball',1,x,y))

    def test_destroyed_tower_changes_defensive_shortlist(self):
        live = placement_points(self.sim,self.state(),'cannon',1)
        destroyed = placement_points(self.sim,self.state(tower_health=(0,1,1,1)),'cannon',1)
        self.assertNotEqual(live[:3],destroyed[:3])

    def test_princess_crossfire_and_range(self):
        tower = self.sim.kb.unit('PrincessTower')
        for y, expected in [(24,True),(5,False)]:
            s=SimState()
            left=self.sim.add(s,tower,1,4,28,tower=True)
            right=self.sim.add(s,tower,1,14,28,tower=True)
            enemy=self.sim.add(s,replace(self.sim.kb.unit('Giant'),speed=0,damage=0),-1,9,y)
            self.sim.advance(s,3)
            self.assertEqual(s.tower_shots[1]>0,expected)
            if expected:
                self.assertEqual(left.target,enemy.uid)
                self.assertEqual(right.target,enemy.uid)
                self.assertLess(enemy.hp,enemy.spec.hp)
            self.assertEqual((left.x,left.y,right.x,right.y),(4,28,14,28))

    def test_spell_candidates_include_group_centers(self):
        s=self.state()
        s.entities=[e for e in s.entities if e.tower]
        spec=self.sim.kb.unit('Skeleton')
        self.sim.add(s,replace(spec,speed=0),-1,5,20)
        self.sim.add(s,replace(spec,speed=0),-1,9,20)
        points=placement_points(self.sim,s,'fireball',1)
        self.assertIn(screen(7,20),points)

    def test_tower_exposure_and_clone_counters(self):
        s=self.state(y=.74)
        self.sim.advance(s,1)
        _,parts=self.sim.evaluate(s)
        self.assertIn('imminent_tower_exposure',parts)
        copy=s.clone()
        copy.tower_shots[1]+=1
        self.assertNotEqual(copy.tower_shots,s.tower_shots)

    def test_dynamic_defense_reduces_late_hog_tower_damage(self):
        snapshot=world(tracks=(Track(1,'hog_rider',-1,.72,.7,99,100,.9,3),))
        state=self.planner.initial(snapshot,5,1)
        def damage(action):
            branch=state.clone()
            self.assertTrue(self.sim.apply(branch,action,1))
            self.sim.advance(branch,8)
            return branch.damage[1]
        old=[]
        for slot,cid in state.hands[1]:
            points=[(.43,.59),(.57,.59)] if cid=='cannon' else ([(.72,.7)] if cid=='fireball' else [(.28,.67),(.72,.67),(.28,.55),(.72,.55)])
            old.extend(damage(SimAction(cid,slot,x,y)) for x,y in points)
        plan=self.planner.plan(snapshot)
        self.assertEqual(plan.status,'ready')
        self.assertLess(damage(plan.action),min(old))


if __name__ == '__main__':
    unittest.main()
