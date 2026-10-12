import unittest
from dataclasses import replace
from unittest.mock import MagicMock,patch
from PIL import Image
from crbot.slice_detector import SliceBattlefield
from crbot.battle_world import Track
from crbot.grid_world_view import compose_yolo,current_entities
from crbot.grid_world import GridWorldAudit
from crbot.predictive_planner import PredictivePlanner,PlanResult
from tests.test_prediction import knowledge,world


class UnconfirmedUnitsTests(unittest.TestCase):
    def setUp(self):
        self.p=PredictivePlanner(knowledge(),{'grid_world':True})
        self.candidate=dict(bbox=[.6911,.4570,.8381,.5325],confidence=.36,
            candidate_card_id='ice_golem',side=None,observation_status='unconfirmed')
        self.event=dict(id='confirmed-1',card_id='mega_knight',x=.765,y=.6,at=98.)
        self.w=world(tracks=(),confirmed_placements=(self.event,),unconfirmed_units=(self.candidate,))

    def allies(self,w):return [e for e in self.p.initial(w,7,1).entities if not e.tower and e.side==1]

    def test_wrong_class_body_locates_existing_hypothesis_without_creating_track(self):
        state=self.p.initial(self.w,7,1)
        entity=next(e for e in state.entities if not e.tower)
        expected=self.p.sim.xy(.7646,.49475)
        self.assertEqual(entity.spec,self.p.kb.roster('mega_knight',self.p.sim.level)[0][0])
        self.assertAlmostEqual(entity.x,expected[0]);self.assertAlmostEqual(entity.y,expected[1])
        self.assertIsNone(entity.track_id)
        self.assertIn('unconfirmed_ally_position_hypothesis',state.uncertainties)
        self.assertIn('unobserved_ally_survival_hypothesis',state.uncertainties)
        packet=GridWorldAudit().update(self.w,state,self.p.sim,PlanResult(1,100,101,'wait'))
        row=next(r for r in packet['entities'] if not r['tower'])
        self.assertEqual(row['source'],'deployment_hypothesis');self.assertTrue(row['identity_estimated'])
        self.assertFalse(any(not e.get('tower') for e in current_entities(packet)))

    def test_no_confirmed_event_never_invents_ally(self):
        self.assertFalse(self.allies(replace(self.w,confirmed_placements=())))

    def test_ambiguous_enemy_old_or_weak_body_does_not_change_position(self):
        baseline=self.allies(replace(self.w,unconfirmed_units=()))[0]
        for candidates in ((dict(self.candidate,side=-1),),(dict(self.candidate,confidence=.1),),
                           (dict(self.candidate,confidence=float('nan')),),(self.candidate,self.candidate),):
            e=self.allies(replace(self.w,unconfirmed_units=candidates))[0]
            self.assertEqual((e.x,e.y),(baseline.x,baseline.y))
        old=replace(self.w,at=102.)
        self.assertEqual([(e.x,e.y) for e in self.allies(old)],[(e.x,e.y) for e in self.allies(replace(old,unconfirmed_units=()))])
        ambiguous=replace(self.w,confirmed_placements=(self.event,dict(self.event,id='second')))
        self.assertEqual([(e.x,e.y) for e in self.allies(ambiguous)],[(e.x,e.y) for e in self.allies(replace(ambiguous,unconfirmed_units=()))])

    def test_observed_enemy_or_ally_owns_its_body(self):
        for side in (-1,1):
            track=Track(1,'mega_knight',side,.7646,.49475,98,100,.9)
            w=replace(self.w,tracks=(track,))
            self.assertNotIn('unconfirmed_ally_position_hypothesis',self.p.initial(w,7,1).uncertainties)

    def test_preview_preserves_unknown_status_and_never_labels_wrong_candidate_as_ally(self):
        image=Image.new('RGB',(400,800))
        packet=dict(revision=1,detections=[],unconfirmed_units=[self.candidate])
        _,boxes=compose_yolo(image,packet,400,800,'yolo')
        self.assertEqual(len(boxes),1)
        self.assertEqual(boxes[0][1]['observation_status'],'unconfirmed')
        self.assertIsNone(boxes[0][1]['side']);self.assertNotIn('card_id',boxes[0][1])

    def test_low_box_cannot_steal_badge_from_strong_detection(self):
        d=SliceBattlefield.__new__(SliceBattlefield)
        d.catalog=object();d.metadata={};d.config={};d.device='cpu';d.calls=0
        d.model=MagicMock();d.model.names={0:'ice-golem',1:'knight'}
        boxes=d.model.predict.return_value[0].boxes
        boxes.cls.cpu.return_value.tolist.return_value=[0,1]
        boxes.conf.cpu.return_value.tolist.return_value=[.36,.9]
        boxes.xyxyn.cpu.return_value.tolist.return_value=[[.69,.45,.83,.53],[.69,.45,.83,.53]]
        badge=dict(side=1,confidence=.95,evidence='level_badge',level=11,bbox=[.7,.43,.72,.45])
        with patch('crbot.slice_detector.unit_card_id',side_effect=lambda name,c:name), \
             patch('crbot.slice_detector.find_level_badges',return_value=[badge]), \
             patch('crbot.slice_detector.associate_badges',side_effect=lambda boxes,b:{0:badge} if boxes else {}):
            result=d.detect(Image.new('RGB',(400,800)))
        self.assertEqual([x['card_id'] for x in result],['knight'])
        self.assertEqual(d.last_rejections['low_confidence'],1)
        self.assertFalse(d.unconfirmed_units)
