import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from PIL import Image

from crbot.unit_badges import find_level_badges, associate_badges
from crbot.pretrained_battlefield import classify_side
from crbot.tower_observation import TowerHealthTracker, observe_tower_health
from crbot.grid_world import GridWorldAudit
from crbot.grid_world_view import render_grid
from crbot.predictive_planner import PredictivePlanner, PlanResult
from crbot.battle_world import BattleWorld
from crbot.live_grid import GridTracker
from crbot.frame_stream import Frame
from tests.battlefield_fixtures import badge_image, scene_with_badge
from tests.test_prediction import knowledge, world


class BadgeEvidenceTests(unittest.TestCase):
    def test_red_eleven_is_enemy_and_blue_eleven_is_ally_even_with_opposite_model(self):
        for side,scores in ((-1,[.99,.01]),(1,[.01,.99])):
            crop=find_level_badges(badge_image(side),arena_only=False)
            self.assertEqual([(b['side'],b['level']) for b in crop],[(side,11)])
            image=scene_with_badge(side=side)
            self.assertEqual(classify_side(image,[.2,.40,.3,.49],scores),(side,.95,'level_badge'))

    def test_colored_body_without_digits_is_not_a_team_observation(self):
        for color in ('red','blue','gray'):
            image=Image.new('RGB',(540,960),color)
            self.assertIsNone(classify_side(image,[.2,.4,.3,.5],[.99,.01]))

    def test_overlapping_bodies_cannot_share_a_badge_and_conflicting_teams_abstain(self):
        badges=find_level_badges(scene_with_badge())
        self.assertEqual(len(associate_badges([[.2,.4,.3,.5],[.19,.405,.31,.51]],badges)),1)
        opposite=dict(badges[0],side=1,x=badges[0]['x']+.001)
        self.assertEqual(associate_badges([[.2,.4,.3,.5]],badges+[opposite]),{})

    def test_badge_below_body_is_not_assigned_to_that_unit(self):
        self.assertEqual(associate_badges([[.2,.30,.3,.5]],find_level_badges(scene_with_badge())),{})

    def test_enemy_at_king_and_both_teams_near_princess_towers_are_not_masked(self):
        for side in (-1,1):
            for x,y in ((.5,.67),(.24,.23),(.76,.62)):
                badges=find_level_badges(scene_with_badge(x=x,y=y,side=side))
                self.assertEqual([(b['side'],b['level']) for b in badges],[(side,11)])


class TowerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.blank=Image.new('RGB',(540,960),(40,70,50))
        self.points=[(.24,.625),(.76,.625),(.24,.219),(.76,.219)]
        self.tracker=TowerHealthTracker()

    def test_absence_and_flat_floor_never_mean_destroyed(self):
        for image in (self.blank,Image.new('RGB',(540,960),(185,150,105))):
            for now in (1,2,3):
                self.assertEqual(self.tracker.observe(image,[],self.points,now=now),(None,)*4)
            self.tracker.reset()

    def test_red_carpet_exposed_after_enemy_tower_collapse_is_not_health_fill(self):
        rois=[[.1,.1,.3,.14]]*4
        self.assertEqual(observe_tower_health(Image.new('RGB',(540,960),(195,100,72)),rois),(None,)*4)

    def test_rubble_requires_multiple_fresh_frames_then_survives_missing_bar_and_resets(self):
        # Independently exercise temporal confirmation, occlusion and next battle.
        with patch('crbot.tower_observation.princess_rubble',side_effect=lambda image,p:p==self.points[1]):
            self.assertIsNone(self.tracker.observe(self.blank,[],self.points,now=1)[1])
            self.assertIsNone(self.tracker.observe(self.blank,[],self.points,now=1)[1])
            self.assertEqual(self.tracker.observe(self.blank,[],self.points,now=1.3)[1],0)
        self.assertEqual(self.tracker.observe(self.blank,[],self.points,now=2)[1],0)
        self.tracker.reset()
        self.assertIsNone(self.tracker.observe(self.blank,[],self.points,now=3)[1])

    def test_transient_effect_or_long_gap_does_not_latch(self):
        with patch('crbot.tower_observation.princess_rubble',return_value=True):
            self.tracker.observe(self.blank,[],self.points,now=1)
            self.assertEqual(self.tracker.observe(self.blank,[],self.points,now=10),(None,)*4)
        self.tracker.observe(self.blank,[],self.points,now=10.1)
        with patch('crbot.tower_observation.princess_rubble',return_value=True):
            self.assertEqual(self.tracker.observe(self.blank,[],self.points,now=10.5),(None,)*4)

    def test_unknown_tower_hp_is_not_published_as_full_and_destroyed_tower_is_not_simulated(self):
        planner=PredictivePlanner(knowledge(),{})
        snap=replace(world(tracks=()),tower_health=(None,0,.5,1.))
        state=planner.initial(snap,7,1)
        self.assertEqual(len([e for e in state.entities if e.tower]),3)
        packet=GridWorldAudit().update(snap,state,planner.sim,PlanResult(1,100,101,'wait'))
        rows={e['tower_index']:e for e in packet['entities'] if e.get('tower')}
        self.assertIsNone(rows[0]['hp']);self.assertIsNone(rows[0]['hp_fraction'])
        self.assertTrue(rows[1]['destroyed']);self.assertEqual(rows[1]['hp_fraction'],0)
        self.assertEqual(rows[2]['hp_fraction'],.5)
        # Even legacy packets with guessed full HP must not paint a green bar.
        for size in ((240,220),(380,600)):
            rendered,_=render_grid(dict(entities=[dict(rows[0],hp_fraction=1)]),size)
            self.assertFalse(np.any(np.all(np.asarray(rendered)==[181,239,137],axis=2)))

    def test_world_latches_destroyed_and_explicit_reset_restores_unknown(self):
        battle=BattleWorld()
        def update(now,hp):
            return battle.update([],now=now,elapsed=now,elixir=7,hand=[],costs={},seconds_per_elixir=2.8,tower_health=hp)
        update(1,(1,0,1,1))
        self.assertEqual(update(2,(None,1,None,None)).tower_health[1],0)
        self.assertEqual(update(3,None).tower_health[1],0)
        battle.reset();self.assertIsNone(update(4,None).tower_health[1])

    def test_live_preview_confirms_rubble_without_new_plan_and_new_battle_resets(self):
        base=dict(at=1,revision=1,battle_started_at=0,geometry=dict(bounds=[0,0,1,1],tower_points=self.points),
                  entities=[dict(id='tower',tower=True,tower_index=1,side=1,x=13,y=25,hp=100,hp_fraction=1)])
        tracker=GridTracker({'elixir_roi':[0,0,1,1]})
        with patch('crbot.live_grid.find_level_badges',return_value=[]),patch('crbot.tower_observation.princess_rubble',return_value=True):
            _,first=tracker.update(Frame(self.blank,1,1,1),(self.blank,base))
            self.assertIsNone(first['entities'][0]['hp_fraction'])
            _,second=tracker.update(Frame(self.blank,1.3,1.3,2),(self.blank,base))
            self.assertTrue(second['entities'][0]['destroyed'])
        with patch('crbot.live_grid.find_level_badges',return_value=[]):
            _,third=tracker.update(Frame(self.blank,1.5,1.5,3),(self.blank,dict(base,revision=2)))
            self.assertEqual(third['entities'][0]['hp_fraction'],0)
            _,fourth=tracker.update(Frame(self.blank,2,2,4),(self.blank,dict(base,battle_started_at=2)))
            self.assertIsNone(fourth['entities'][0]['hp_fraction'])


if __name__=='__main__':unittest.main()
