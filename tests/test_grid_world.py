import math
import time
import unittest
from dataclasses import replace

from crbot.battle_world import BattleWorld, Track
from crbot.battle_simulation import SimState, SimAction
from crbot.grid_navigation import route, cell, passable, obstacles
from crbot.grid_world import GridWorldAudit, target_for, along
from crbot.grid_world_view import render_grid, current_entities
from crbot.parallel_combat import CombatPool
from crbot.predictive_planner import PredictivePlanner, PlanResult
from tests.test_prediction import knowledge, world


class GridWorldTests(unittest.TestCase):
    def test_current_view_excludes_stale_unknown_and_deployment_predictions(self):
        snapshot = world(tracks=(Track(7,'unknown:left:single',-1,.2,.22,97,98.016,.5),
                                Track(9,'unknown:right:single',-1,.7,.6,100,100,.5)))
        audit = GridWorldAudit()
        packet = audit.update(snapshot,self.p.initial(snapshot,7,1),self.p.sim,PlanResult(1,100,101,'wait'))
        stale = next(e for e in packet['entities'] if e.get('track_id')==7)
        self.assertEqual(stale['source'], 'occluded')
        self.assertAlmostEqual(stale['age_s'], 1.984)
        packet['entities'].append(dict(id='predicted',side=1,x=4,y=20,source='deployment_hypothesis'))
        visible = list(current_entities(packet))
        self.assertFalse(any(e.get('track_id')==7 or e.get('id')=='predicted' for e in visible))
        self.assertTrue(any(e.get('track_id')==9 for e in visible))
        _, boxes = render_grid(packet,(450,700))
        self.assertEqual([e['id'] for _,e in boxes],[e['id'] for e in visible])

    def setUp(self):
        self.kb=knowledge();self.p=PredictivePlanner(self.kb,dict(grid_world=True,fast_defense=True))

    def test_ground_route_crosses_bridge_air_ignores_river(self):
        state=SimState();sim=self.p.sim
        ground=sim.add(state,self.kb.unit('Knight'),1,9,23)
        target=sim.add(state,self.kb.unit('PrincessTower'),-1,9,7,tower=True)
        points=route(state,ground,target,sim.geometry)
        self.assertGreater(len(points),2)
        for a,b in zip(points,points[1:]):
            for n in range(11):
                x=a[0]+(b[0]-a[0])*n/10;y=a[1]+(b[1]-a[1])*n/10
                self.assertTrue(passable(*cell(x,y),sim.geometry.bridges,()))
        air=sim.add(state,self.kb.roster('minions')[0][0],1,9,23)
        self.assertEqual(route(state,air,target,sim.geometry),[(9,23),(9,7)])

    def test_ground_path_goes_around_building_and_clone_is_isolated(self):
        state=SimState();sim=self.p.sim
        mover=sim.add(state,self.kb.unit('Knight'),1,4,29)
        target=sim.add(state,self.kb.unit('PrincessTower'),-1,4,18,tower=True)
        sim.add(state,self.kb.roster('cannon')[0][0],1,4,24)
        blocked=obstacles(state,mover,target)
        points=route(state,mover,target,sim.geometry)
        self.assertTrue(any(abs(x-4)>1 for x,y in points))
        self.assertTrue(all(cell(x,y) not in blocked for x,y in points))
        cloned=state.clone();sim.advance(cloned,.5)
        self.assertEqual((mover.x,mover.y),(4,29))
        self.assertNotEqual((cloned.entities[0].x,cloned.entities[0].y),(4,29))

    def test_targeting_uses_air_and_building_only_attributes(self):
        sim=self.p.sim;state=SimState()
        giant=sim.add(state,self.kb.roster('giant')[0][0],1,4,23)
        sim.add(state,self.kb.unit('Knight'),-1,4,24)
        tower=sim.add(state,self.kb.unit('PrincessTower'),-1,4,20,tower=True)
        self.assertIs(target_for(sim,state,giant),tower)
        state=SimState();knight=sim.add(state,self.kb.unit('Knight'),1,4,23)
        sim.add(state,self.kb.roster('minions')[0][0],-1,4,23)
        self.assertIsNone(target_for(sim,state,knight))

    def test_unknown_identity_upgrades_without_duplicate_and_occlusion_expires(self):
        battle=BattleWorld()
        def update(cid,x,now,visible=True):
            return battle.update([dict(card_id=cid,x=x,y=.6,side=-1,confidence=.9)] if visible else [],
                now=now,elapsed=now,elixir=7,hand=[],costs={},seconds_per_elixir=2.8)
        first=update('unknown:left:heavy',.3,1)
        second=update('giant',.32,1.3)
        third=update('unknown:left:health',.34,1.6)
        self.assertEqual(len(third.tracks),1)
        self.assertEqual(first.tracks[0].track_id,second.tracks[0].track_id)
        self.assertEqual(third.tracks[0].card_id,'giant')
        hidden=update('',0,2,False)
        self.assertEqual(len(hidden.tracks),1)
        self.assertEqual(len(update('',0,5,False).tracks),0)

    def test_two_separate_units_keep_distinct_track_ids(self):
        battle=BattleWorld()
        snap=battle.update([dict(card_id='knight',x=x,y=.6,side=-1,confidence=.9) for x in (.3,.33)],
                          now=1,elapsed=1,elixir=7,hand=[],costs={},seconds_per_elixir=2.8)
        self.assertEqual(len(snap.tracks),2)
        self.assertEqual(len({t.track_id for t in snap.tracks}),2)

    def test_king_towers_are_present_and_activate_after_damage(self):
        p=PredictivePlanner(self.kb,dict(king_towers=True,grid_world=True))
        state=p.initial(world(tracks=()),7,1)
        kings=[e for e in state.entities if e.tower_kind=='king']
        self.assertEqual(len(kings),2);self.assertFalse(any(e.active for e in kings))
        kings[0].hp-=10;p.sim.advance(state,.25)
        self.assertTrue(kings[0].active);self.assertFalse(kings[1].active)

    def test_packet_keeps_unknown_allies_and_compares_next_observation(self):
        audit=GridWorldAudit();plan=PlanResult(1,100,101,'wait')
        track=Track(1,'knight',1,.3,.7,100,100,.9,hp_fraction=.8,hp_observed_at=100)
        snapshot=world(tracks=(track,Track(2,'unknown:right:health',1,.7,.6,100,100,.8)))
        first=audit.update(snapshot,self.p.initial(snapshot,7,1),self.p.sim,plan)
        unknown=next(e for e in first['entities'] if e['track_id']==2)
        self.assertIsNone(unknown['attributes'])
        entity=next(e for e in first['entities'] if e['track_id']==1)
        xy=along(entity['path'],entity['attributes']['speed']*.5) or [entity['x'],entity['y']]
        nx,ny=self.p.sim.screen(*xy)
        next_world=replace(snapshot,at=100.5,tracks=(replace(track,x=nx,y=ny,last_seen=100.5,hp_fraction=.7,hp_observed_at=100.5),))
        second=audit.update(next_world,self.p.initial(next_world,7,1),self.p.sim,plan)
        self.assertEqual(second['calibration']['samples'],1)
        self.assertLess(second['calibration']['mean_position_error_tiles'],.01)
        self.assertAlmostEqual(second['calibration']['errors'][0]['hp_change'],-.1)
        image,boxes=render_grid(second,(380,600));self.assertEqual(image.size,(380,600));self.assertTrue(boxes)

    def test_grid_simulation_parallel_matches_serial(self):
        config=dict(grid_world=True,king_towers=True,fast_defense=True,combat_workers=2)
        p=PredictivePlanner(self.kb,config);pool=CombatPool(self.kb.payload,config)
        try:
            snap=world();roots=[SimAction(),SimAction('knight',0,.3,.7)];scenarios=[(7,1,False)]
            serial=[p.evaluate_root(snap,r,scenarios,4,'defend',time.perf_counter()+20) for r in roots]
            parallel=list(pool.rows(snap,roots,scenarios,4,'defend',time.perf_counter()+20))
            self.assertEqual(serial,parallel)
        finally:pool.close()


if __name__=='__main__':unittest.main()
