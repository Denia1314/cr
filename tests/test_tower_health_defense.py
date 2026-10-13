import unittest
from dataclasses import replace
from unittest.mock import patch
from crbot.battle_world import BattleWorld,Track
from crbot.predictive_planner import PredictivePlanner
from crbot.battle_simulation import SimAction
from crbot.combat_health import CombatHealth
from crbot.tower_threat import assess
from crbot.emergency_defense import emergency_defense
from crbot.tactical_objective import avoid_late_defense
from tests.test_prediction import knowledge,world
from tests.test_urgent_tower_defense import row


class TowerHealthDefenseTests(unittest.TestCase):
    def setUp(self):
        self.p=PredictivePlanner(knowledge(),{})
        self.w=world(elixir=4,hand=((0,'valkyrie'),),tracks=(Track(1,'hog_rider',-1,.28,.60,99,100,.9,
                     hp_fraction=.02,hp_observed_at=100),))

    def state(self,w=None):
        s=self.p.initial(w or self.w,7,1)
        enemy=next(e for e in s.entities if not e.tower and e.side==-1)
        tower=min((e for e in s.entities if e.side==1 and e.tower),key=lambda t:self.p.sim.distance(t,enemy))
        return s,enemy,tower

    def test_low_hp_hog_dies_to_princess_before_hit_no_emergency(self):
        s,e,t=self.state()
        threat=assess(s,e,t,window=6)
        self.assertEqual(threat['tower_damage'],0)
        self.assertLess(threat['estimated_kill_s'],threat['first_tower_hit_s'])
        actual=s.clone();self.p.sim.advance(actual,6)
        self.assertEqual(actual.damage[1],0)
        self.assertFalse(any(e.track_id==1 for e in actual.entities))
        self.assertIsNone(emergency_defense(self.p.sim,s)[0])
        self.assertEqual(self.p.sim.evaluate(s)[1]['imminent_tower_exposure'],0)

    def test_full_hp_hog_and_distracted_tower_are_real_threats(self):
        s,e,t=self.state(replace(self.w,tracks=(replace(self.w.tracks[0],hp_fraction=1,y=.70),)))
        self.assertGreater(assess(s,e,t,window=6)['tower_damage'],0)
        s,e,t=self.state()
        distract=self.p.sim.add(s,self.p.kb.unit('Giant',11),-1,t.x,t.y-2)
        t.target=distract.uid
        self.assertGreater(assess(s,e,t,window=6)['tower_damage'],0)

    def test_existing_troop_contributes_damage_without_borrowing_air_damage(self):
        s,e,t=self.state()
        t.active=False;t.tower_kind='king'
        self.assertGreater(assess(s,e,t,window=6)['tower_damage'],0)
        ally=self.p.sim.add(s,self.p.kb.unit('Musketeer',11),1,e.x+1,e.y+1)
        self.assertEqual(assess(s,e,t,window=6)['tower_damage'],0)
        ally.ready_at=100
        self.assertGreater(assess(s,e,t,window=6)['tower_damage'],0)

    def test_late_local_defense_cancelled_but_cap_and_needed_defense_preserved(self):
        from crbot.defense_timing import forecast
        s,e,t=self.state();forecasts=forecast(self.p.sim,s,self.w.hand)
        wait=row(SimAction(),0);play=row(SimAction('valkyrie',0,.28,.62),0)
        chosen,audit=avoid_late_defense(play,[wait,play],self.w,self.p.sim,forecasts)
        self.assertIs(chosen,wait);self.assertTrue(audit['changed'])
        self.assertIs(avoid_late_defense(play,[wait,play],replace(self.w,elixir=10),self.p.sim,forecasts)[0],play)
        danger=row(SimAction(),500)
        self.assertIs(avoid_late_defense(play,[danger,play],self.w,self.p.sim,forecasts)[0],play)

    def test_stale_visual_anchor_survives_and_combat_interval_tracks_damage(self):
        b=BattleWorld()
        args=dict(elapsed=20,elixir=4,hand=self.w.hand,costs={'hog_rider':4},seconds_per_elixir=2.8)
        d=dict(card_id='hog_rider',side=-1,x=.28,y=.60,confidence=.9,hp_fraction=.5,hp_confidence=.9)
        first=b.update([d],now=100,**args)
        h=CombatHealth();h.update(self.p.sim,first);h.remember(self.p.initial(first,7,1),first)
        d.pop('hp_fraction')
        second=b.update([d],now=101,**args)
        audit=h.update(self.p.sim,second,budget_s=1)
        track=second.tracks[0]
        self.assertEqual(track.hp_fraction,.5)
        self.assertEqual(track.hp_observed_at,100)
        self.assertTrue(audit['complete'])
        self.assertLess(track.hp_estimate,.5)
        self.assertLess(track.hp_estimate_high,1)
        self.assertEqual(track.hp_estimate_source,'combat_projection')
        entity=next(e for e in self.p.initial(second,7,1).entities if not e.tower)
        self.assertAlmostEqual(entity.hp/entity.spec.hp,track.hp_estimate_high)

    def test_fresh_bar_overrides_projection_and_timeout_never_publishes_partial_damage(self):
        h=CombatHealth();h.update(self.p.sim,self.w);h.remember(self.p.initial(self.w,7,1),self.w)
        w=replace(self.w,revision=2,at=101,tracks=(replace(self.w.tracks[0],last_seen=101,hp_fraction=.8,hp_observed_at=101),))
        with patch.object(self.p.sim,'advance',side_effect=TimeoutError):
            audit=h.update(self.p.sim,w)
        self.assertFalse(audit['complete'])
        self.assertEqual(w.tracks[0].hp_estimate,.8)
        self.assertEqual(w.tracks[0].hp_estimate_source,'visual_health')

    def test_predicted_death_does_not_remove_visible_body_and_reset_drops_history(self):
        h=CombatHealth();h.update(self.p.sim,self.w);h.remember(self.p.initial(self.w,7,1),self.w)
        w=replace(self.w,revision=2,at=101,tracks=(replace(self.w.tracks[0],last_seen=101),))
        h.update(self.p.sim,w,budget_s=1)
        self.assertGreater(w.tracks[0].hp_estimate_high,0)
        self.assertTrue(any(e.track_id==1 for e in self.p.initial(w,7,1).entities))
        h.reset();self.assertIsNone(h.state)

    def test_measured_send_latency_updates_all_simulation_paths(self):
        self.p.update_execution_delay(.60)
        self.assertAlmostEqual(self.p.config['defense_pipeline_s'],.78)
        self.assertAlmostEqual(self.p.sim.defense_pipeline_s,.78)
        self.p.update_execution_delay(.1)
        self.assertAlmostEqual(self.p.sim.defense_pipeline_s,.35)

    def test_health_projection_preserves_cooldowns_and_in_flight_shots(self):
        h=CombatHealth()
        w=replace(self.w,tracks=(replace(self.w.tracks[0],hp_fraction=None,hp_observed_at=None),))
        h.update(self.p.sim,w);s=self.p.initial(w,7,1)
        tower=next(e for e in s.entities if e.tower and e.side==1)
        enemy=next(e for e in s.entities if e.track_id==1)
        tower.ready_at=1.2;tower.target=enemy.uid;tower.winding_target=enemy.uid
        s.impacts.append((.8,1,enemy.uid,enemy.x,enemy.y,100.,0.,True,1.))
        h.remember(s,w)
        w2=replace(w,revision=2,at=100.25)
        h.update(self.p.sim,w2,budget_s=1)
        h.remember(self.p.initial(w2,7,1),w2)
        t2=next(e for e in h.state.entities if e.tower and e.side==1)
        self.assertAlmostEqual(t2.ready_at,.95)
        self.assertEqual(len(h.state.impacts),1)
        self.assertAlmostEqual(h.state.impacts[0][0],.55)
        w3=replace(w2,revision=3,at=101.,tracks=tuple(replace(t) for t in w2.tracks))
        h.update(self.p.sim,w3,budget_s=1)
        self.assertLess(w3.tracks[0].hp_estimate,w2.tracks[0].hp_estimate)

    def test_ground_only_defender_cannot_cancel_air_threat(self):
        w=replace(self.w,tracks=(replace(self.w.tracks[0],card_id='balloon',hp_fraction=.02),))
        s,e,t=self.state(w)
        t.tower_kind='king';t.active=False
        self.p.sim.add(s,self.p.kb.unit('Knight',11),1,e.x,e.y)
        self.assertGreater(assess(s,e,t,window=12)['tower_damage'],0)

    def test_actual_gpu_low_hp_no_play_matches_reference(self):
        from crbot.gpu_combat import GPUCombat
        import time
        gpu=GPUCombat(self.p.sim)
        if not gpu.available:
            self.skipTest(gpu.error or 'CUDA unavailable')
        s,e,t=self.state();reference=s.clone()
        self.p.sim.advance(reference,6)
        gpu.advance([s],6,time.perf_counter()+10)
        self.assertEqual(self.p.sim.evaluate(reference),self.p.sim.evaluate(s))
        self.assertEqual(s.damage[1],0)


if __name__=='__main__':unittest.main()
