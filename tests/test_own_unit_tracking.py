import unittest
from dataclasses import replace

from PIL import Image

from crbot.battle_world import BattleWorld
from crbot.own_unit_tracking import OwnUnitTracker
from crbot.predictive_planner import PredictivePlanner
from crbot.grid_world_view import compose_yolo
from tests.test_prediction import knowledge, world


class OwnUnitTrackingTests(unittest.TestCase):
    def setUp(self):
        self.planner=PredictivePlanner(knowledge(),{})
        self.tracker=OwnUnitTracker(self.planner.sim)
        self.event=dict(id='play1',card_id='mega_knight',x=.75,y=.6,at=100.)

    def body(self, x=.75, y=.59, **fields):
        return dict(bbox=[x-.04,y-.035,x+.04,y+.035],confidence=.36,
                    candidate_card_id='ice_golem',side=None,observation_status='unconfirmed',**fields)

    def update(self, now, bodies=None, events=None, detections=(), generation=1):
        return self.tracker.update(list(detections),[self.body()] if bodies is None else bodies,
            [self.event] if events is None else events,now=now,generation=generation)

    def bind(self):
        self.assertFalse(self.update(100.1)[2])
        observations,pending,locks=self.update(100.3)
        self.assertEqual(len(locks),1)
        self.assertFalse(pending)
        self.assertEqual(observations[0]['card_id'],'mega_knight')
        self.assertEqual(observations[0]['visual_card_id'],'ice_golem')
        self.assertEqual(observations[0]['confidence'],.36)
        return observations,locks[0]['identity_lock_id']

    def test_class_flips_preserve_identity_and_world_track(self):
        first,key=self.bind()
        w=BattleWorld();w.confirm('play1','mega_knight',.75,.6,100.)
        def observe(rows,at):
            return w.update(rows,now=at,elapsed=at-100,elixir=5,hand=[],costs={},seconds_per_elixir=2.8)
        before=observe(first,100.3)
        raw=dict(self.body(y=.52),card_id='knight',side=1,confidence=.9,x=.75,y=.52)
        rows,_,locks=self.update(100.6,bodies=[],detections=[raw])
        after=observe(rows,100.6)
        self.assertEqual(after.tracks[0].card_id,'mega_knight')
        self.assertEqual(after.tracks[0].track_id,before.tracks[0].track_id)
        self.assertEqual(locks[0]['identity_lock_id'],key)
        self.assertTrue(after.confirmed_placements[0]['identity_bound'])

    def test_short_occlusion_and_reappearance_keep_id_but_expire(self):
        _,key=self.bind()
        _,_,locks=self.update(101.,bodies=[])
        self.assertEqual(locks[0]['position_state'],'occluded')
        rows,_,locks=self.update(101.3,bodies=[self.body(y=.56)])
        self.assertEqual(rows[0]['identity_lock_id'],key)
        self.assertFalse(self.update(104.4,bodies=[])[2])
        self.assertFalse(self.update(104.6,bodies=[self.body()],events=[self.event])[2])

    def test_duplicate_capture_keeps_cached_bound_identity(self):
        _,key=self.bind()
        rows,_,locks=self.update(100.3,bodies=[dict(self.body(),candidate_card_id='knight')])
        self.assertEqual(rows[0]['card_id'],'mega_knight')
        self.assertEqual(rows[0]['identity_lock_id'],key)
        self.assertEqual(locks[0]['identity_lock_id'],key)

    def test_pending_visual_association_does_not_create_duplicate_unit(self):
        raw=dict(self.body(),card_id='ice_golem',side=1,confidence=.9,x=.75,y=.59)
        rows,_,locks=self.update(100.1,bodies=[],detections=[raw])
        self.assertFalse(locks)
        self.assertTrue(rows[0]['identity_ambiguous'])
        w=BattleWorld();w.confirm('play1','mega_knight',.75,.6,100.)
        snapshot=w.update(rows,now=100.1,elapsed=.1,elixir=5,hand=[],costs={},seconds_per_elixir=2.8)
        self.assertFalse(snapshot.tracks)
        state=self.planner.initial(snapshot,5,1.)
        self.assertEqual(len([e for e in state.entities if not e.tower and e.side==1]),1)
        rows,_,locks=self.update(100.3,bodies=[],detections=[raw])
        self.assertFalse(rows[0].get('identity_ambiguous'))
        self.assertEqual(rows[0]['card_id'],'mega_knight')

    def test_confirmed_event_publication_is_an_independent_snapshot(self):
        w=BattleWorld();generation,_=w.placement_state()
        w.confirm('play1','mega_knight',.75,.6,100.)
        _,events=w.placement_state()
        events[0]['card_id']='knight'
        self.assertEqual(w.placement_state()[1][0]['card_id'],'mega_knight')
        w.reset()
        self.assertGreater(w.placement_state()[0],generation)
        self.assertFalse(w.placement_state()[1])

    def test_unconfirmed_or_repeated_capture_cannot_bind(self):
        self.update(100.1,events=[])
        self.assertFalse(self.update(100.3,events=[])[2])
        self.tracker.reset()
        self.update(100.1)
        self.assertFalse(self.update(100.1)[2])
        self.assertFalse(self.update(100.0)[2])

    def test_tracking_only_low_score_needs_confirmed_action_and_two_frames(self):
        weak=dict(self.body(),confidence=.12,tracking_only=True)
        self.assertFalse(self.update(100.1,bodies=[weak],events=[])[2])
        self.tracker.reset()
        self.assertFalse(self.update(100.1,bodies=[weak])[2])
        rows,_,locks=self.update(100.3,bodies=[weak])
        self.assertEqual(len(locks),1)
        self.assertEqual(rows[0]['confidence'],.12)
        w=BattleWorld()
        args=dict(now=100.3,elapsed=.3,elixir=5,hand=[],costs={},seconds_per_elixir=2.8)
        self.assertFalse(w.update(rows,**args).tracks)
        w.confirm('play1','mega_knight',.75,.6,100.)
        args['now']=100.4
        self.assertEqual(w.update(rows,**args).tracks[0].card_id,'mega_knight')
        self.tracker.reset()
        unmarked=dict(weak,tracking_only=False)
        self.update(100.1,bodies=[unmarked])
        self.assertFalse(self.update(100.3,bodies=[unmarked])[2])

    def test_weak_health_reading_does_not_terminate_identity(self):
        self.bind()
        _,_,locks=self.update(100.5,bodies=[self.body(hp_fraction=0,hp_confidence=.2)])
        self.assertEqual(len(locks),1)

    def test_preexisting_body_and_explicit_enemy_are_not_claimed(self):
        self.update(99.5,events=[])
        self.assertFalse(self.update(100.1)[2])
        self.assertFalse(self.update(100.3)[2])
        self.tracker.reset()
        self.update(100.1,bodies=[dict(self.body(),side=-1)])
        self.assertFalse(self.update(100.3,bodies=[dict(self.body(),side=-1)])[2])

    def test_delayed_confirmation_uses_issued_time(self):
        self.update(100.1,events=[])
        event=dict(self.event,at=100.2,issued_at=100.)
        self.update(100.3,events=[event])
        self.assertEqual(len(self.update(100.5,events=[event])[2]),1)

    def test_multiple_bodies_or_competing_deployments_do_not_guess(self):
        bodies=[self.body(x=.73),self.body(x=.77)]
        self.update(100.1,bodies=bodies)
        self.assertFalse(self.update(100.3,bodies=bodies)[2])
        self.tracker.reset()
        events=[self.event,dict(self.event,id='play2')]
        self.update(100.1,events=events)
        self.assertFalse(self.update(100.3,events=events)[2])

    def test_homogeneous_group_uses_separate_ids_without_multiplying_entities(self):
        event=dict(self.event,card_id='archers',x=.5,y=.6)
        bodies=[self.body(x=.46),self.body(x=.54)]
        self.update(100.1,bodies=bodies,events=[event])
        observations,_,locks=self.update(100.3,bodies=bodies,events=[event])
        self.assertEqual(len(locks),2)
        self.assertEqual(len({r['identity_lock_id'] for r in observations}),2)
        keys={r['identity_lock_id'] for r in observations}
        observations,_,locks=self.update(100.5,bodies=list(reversed(bodies)),events=[event])
        self.assertEqual({r['identity_lock_id'] for r in observations},keys)

    def test_death_and_battle_reset_end_binding(self):
        self.bind()
        rows,_,locks=self.update(100.5,bodies=[self.body(hp_fraction=0,hp_confidence=1.)])
        self.assertEqual(rows[0]['card_id'],'mega_knight')
        self.assertFalse(locks)
        self.assertFalse(self.update(100.7)[2])
        self.assertFalse(self.update(100.9,events=[],generation=2)[2])

    def test_far_teleport_and_ambiguous_crossing_do_not_reassign(self):
        _,key=self.bind()
        rows,_,locks=self.update(100.5,bodies=[self.body(x=.25,y=.3)])
        self.assertFalse(rows)
        self.assertEqual(locks[0]['identity_lock_id'],key)
        self.assertEqual(locks[0]['position_state'],'occluded')
        rows,_,locks=self.update(100.7,bodies=[self.body(x=.73),self.body(x=.77)])
        self.assertFalse(rows)
        self.assertEqual(locks[0]['position_state'],'occluded')

    def test_planner_keeps_locked_identity_during_occlusion_without_respawn(self):
        observations,_=self.bind()
        w=BattleWorld();w.confirm('play1','mega_knight',.75,.6,100.)
        snapshot=w.update(observations,now=100.3,elapsed=.3,elixir=5,hand=[],costs={},seconds_per_elixir=2.8)
        state=self.planner.initial(replace(snapshot,at=102.),5,1.)
        allies=[e for e in state.entities if e.side==1 and not e.tower]
        self.assertEqual(len(allies),1)
        self.assertEqual(allies[0].spec,self.planner.kb.roster('mega_knight',self.planner.sim.level)[0][0])
        state=self.planner.initial(replace(snapshot,at=104.),5,1.)
        self.assertFalse([e for e in state.entities if e.side==1 and not e.tower])

    def test_preview_distinguishes_locked_visible_and_estimated_body(self):
        rows,key=self.bind()
        frame=Image.new('RGB',(400,800))
        packet=dict(revision=1,detections=rows,unconfirmed_units=[],identity_tracks=self.tracker.rows(100.3))
        _,boxes=compose_yolo(frame,packet,400,800,'yolo')
        self.assertEqual(len(boxes),1)
        self.assertEqual(boxes[0][1]['id'],key)
        packet.update(detections=[],identity_tracks=self.tracker.rows(101.))
        _,boxes=compose_yolo(frame,packet,400,800,'yolo')
        self.assertEqual(boxes[0][1]['position_state'],'occluded')
        self.assertEqual(boxes[0][1]['id'],key)


if __name__=='__main__':unittest.main()
