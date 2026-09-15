from dataclasses import asdict,replace
from pathlib import Path
import json
import unittest
from PIL import Image,ImageDraw
from crbot.arena_geometry import ArenaGeometry
from crbot.battle_lab import coverage,synthetic_scenarios
from crbot.tower_observation import observe_tower_health
from crbot.trajectory_audit import trajectory_errors
from crbot.predictive_planner import PredictivePlanner
from tests.test_prediction import knowledge,world


class BattleLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb=knowledge()
        cls.config=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text(encoding='utf-8'))['prediction']

    def test_geometry_round_trip_and_river_match_reference(self):
        g=ArenaGeometry.from_config(self.config['arena_geometry'])
        self.assertAlmostEqual(g.screen(9,16)[1],.427)
        for x in range(19):
            for y in range(33):
                a,b=g.xy(*g.screen(x,y))
                self.assertAlmostEqual(a,x)
                self.assertAlmostEqual(b,y)

    def test_calibrated_bridge_deployment_is_legal(self):
        p=PredictivePlanner(self.kb,{'arena_geometry':self.config['arena_geometry']})
        s=p.initial(world(),7,1)
        self.assertTrue(p.sim.legal_placement(s,'knight',1,*p.sim.screen(3.5,17)))
        self.assertFalse(p.sim.legal_placement(s,'knight',1,*p.sim.screen(3.5,14)))
        self.assertEqual(len([e for e in s.entities if e.tower]),4)

    def test_coverage_never_marks_unvalidated_cards_as_validated(self):
        rows=coverage(self.kb)
        self.assertEqual(len(rows),132)
        self.assertTrue(all(v['verified_game_cases']==0 for v in rows.values()))
        self.assertEqual(rows['mirror']['status'],'missing')
        self.assertEqual(len(synthetic_scenarios()),30)

    def test_tower_bars_are_observations_and_absence_is_unknown(self):
        image=Image.new('RGB',(200,100),(30,30,30));draw=ImageDraw.Draw(image)
        draw.rectangle((20,10,49,14),fill=(60,190,240))
        draw.rectangle((20,30,79,34),fill=(230,60,100))
        rois=[[.1,.1,.7,.15],[.1,.5,.7,.55],[.1,.3,.7,.35],[.1,.7,.7,.75]]
        hp=observe_tower_health(image,rois)
        self.assertEqual(hp,(.25,None,.5,None))

    def test_latency_compensates_observed_motion_in_correct_coordinate_system(self):
        p=PredictivePlanner(self.kb,{'arena_geometry':self.config['arena_geometry']})
        w=world();w=replace(w,tracks=(replace(w.tracks[0],vy=.02),),observation_delay_s=.5)
        s=p.initial(w,7,1);entity=next(e for e in s.entities if not e.tower)
        self.assertAlmostEqual(entity.y,p.sim.xy(w.tracks[0].x,w.tracks[0].y+.01)[1])

    def test_attack_retains_reserve_and_does_not_require_legacy(self):
        p=PredictivePlanner(self.kb,dict(unified_tactics=True,fast_defense=True,budget_ms=2000,attack_reserve=3))
        low=p.plan(world(tracks=(),elixir=2))
        self.assertEqual(low.tactical_phase,'develop')
        self.assertEqual(low.status,'wait')
        high=p.plan(world(tracks=(),elixir=10))
        self.assertEqual(high.status,'ready')
        self.assertTrue(high.combo_candidates)
        self.assertEqual(high.completed_depth,2)
        for candidate in high.combo_candidates:
            self.assertEqual(candidate['scope'],'conditional_two_card_dynamic_positions')
            for branch in candidate['branches']:
                self.assertEqual(branch['followup_after_s'],1.5)
                self.assertIn('enemy_followup_response',branch)
                self.assertIn('followup_action',branch)
        self.assertLessEqual(self.kb.cards[high.action.card_id]['elixir'],7)

    def test_trajectory_report_requires_future_matching_observation(self):
        g=ArenaGeometry();x,y=g.screen(5,20)
        events=[dict(event='battle_prediction',world=dict(at=1,revision=1),plan=dict(compute=dict(arena_geometry=asdict(g)),enemy_forecast=[dict(track_id=1,linear_samples=[dict(dt=1,x=5,y=20)])])),
                dict(event='battle_prediction',world=dict(at=2,revision=2,observations=[dict(track_id=1,card_id='knight',last_seen=2,x=x,y=y)]),plan={})]
        result=trajectory_errors(events)
        self.assertEqual(result['samples'],1)
        self.assertAlmostEqual(result['mean_error_tiles'],0)
        self.assertFalse(result['battle_acceptance'])

    def test_attack_cooldown_does_not_freeze_chasing_movement(self):
        from crbot.battle_simulation import Simulator,SimState
        sim=Simulator(self.kb,step=.05);state=SimState()
        unit=replace(self.kb.unit('Knight'),speed=1.,reach=1.,first_hit=.1)
        actor=sim.add(state,unit,1,4,24)
        enemy=sim.add(state,unit,-1,4,20)
        actor.ready_at=10
        before=actor.y
        sim.advance(state,.2)
        self.assertLess(actor.y,before)

    def test_deploy_delay_still_blocks_movement(self):
        from crbot.battle_simulation import Simulator,SimState
        sim=Simulator(self.kb,step=.05);state=SimState()
        unit=replace(self.kb.unit('Knight'),deploy=1.,speed=1.)
        actor=sim.add(state,unit,1,4,24,deployed=True)
        sim.add(state,unit,-1,4,20)
        sim.advance(state,.5)
        self.assertEqual(actor.y,24)

    def test_first_hit_windup_begins_only_when_in_range(self):
        from crbot.battle_simulation import Simulator,SimState
        sim=Simulator(self.kb,step=.05);state=SimState()
        attacker=replace(self.kb.unit('Knight'),speed=2.,reach=1.,radius=0.,first_hit=.5,damage=100)
        victim=replace(attacker,speed=0.,damage=0.)
        actor=sim.add(state,attacker,1,4,24)
        enemy=sim.add(state,victim,-1,4,21)
        sim.advance(state,1.1)
        self.assertEqual(enemy.hp,victim.hp)
        sim.advance(state,.7)
        self.assertLess(enemy.hp,victim.hp)
