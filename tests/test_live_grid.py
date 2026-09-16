import copy
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from crbot.live_grid import GridTracker
from crbot.frame_stream import Frame


class LiveGridTests(unittest.TestCase):
    def test_scaled_badges_keep_normalized_location(self):
        from PIL import ImageDraw
        from crbot.battle_perception import _level_badge_candidates
        from tests.battlefield_fixtures import scene_with_badge
        image=scene_with_badge(x=.4,y=.44,size=(1080,1920))
        original=_level_badge_candidates(image)
        small=GridTracker.small(image)
        scaled=_level_badge_candidates(small,pixel_scale=small.width/image.width)
        self.assertEqual(len(original),1)
        self.assertEqual(len(scaled),1)
        self.assertAlmostEqual(original[0][0],scaled[0][0],delta=.005)
        self.assertAlmostEqual(original[0][1],scaled[0][1],delta=.005)

    def setUp(self):
        rng=np.random.default_rng(2)
        self.pixels=rng.integers(0,256,(320,180),dtype=np.uint8)
        self.image=Image.fromarray(self.pixels).convert('RGB')
        self.base=dict(at=1.,revision=7,geometry=dict(bounds=[0,0,1,1]),bridges=[3.5,14.5],
                       width=18,height=32,entities=[dict(id='unit',x=9,y=16,side=-1,age_s=0,
                       card_id='knight',hp_fraction=.8,path=[[1,2],[3,4]])])
        self.tracker=GridTracker({'elixir_roi':[0,0,1,1]})

    def test_new_pixels_update_position_without_new_plan_and_do_not_mutate_model(self):
        original=copy.deepcopy(self.base)
        moved=Image.fromarray(np.roll(self.pixels,3,axis=1)).convert('RGB')
        with patch('crbot.live_grid.find_level_badges',return_value=[]):
            _,packet=self.tracker.update(Frame(moved,1.033,1.04,12),(self.image,self.base))
        self.assertEqual(packet['revision'],12)
        self.assertEqual(packet['model_revision'],7)
        row=packet['entities'][0]
        self.assertAlmostEqual(row['x'],9.3,delta=.06)
        self.assertEqual(row['source'],'optical_flow')
        self.assertIsNone(row['hp_fraction'])
        self.assertEqual(row['path'],[])
        self.assertEqual(self.base,original)

    def test_missing_features_and_expired_identity_are_not_repeated_as_detection(self):
        blank=Image.new('RGB',(180,320))
        with patch('crbot.live_grid.find_level_badges',return_value=[]):
            _,packet=self.tracker.update(Frame(blank,1.1,1.11,2),(blank,self.base))
            self.assertEqual(packet['entities'],[])
            _,packet=self.tracker.update(Frame(self.image,4,4.01,3),(self.image,self.base))
            self.assertEqual(packet['entities'],[])

    def test_new_badge_enters_without_waiting_for_planner(self):
        with patch('crbot.live_grid.find_level_badges',return_value=[dict(x=.2,y=.3,side=-1,level=11)]):
            _,packet=self.tracker.update(Frame(self.image,4,4.01,3),(self.image,self.base))
        row=packet['entities'][0]
        self.assertEqual(row['card_id'],'unknown')
        self.assertEqual(row['source'],'current_frame_badge')
        self.assertEqual(row['level'],11)
        self.assertEqual(row['side_evidence'],'level_badge')
        self.assertAlmostEqual(row['x'],3.6)
