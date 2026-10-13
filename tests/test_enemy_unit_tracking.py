import unittest
from crbot.enemy_unit_tracking import EnemyUnitTracker
from crbot.battle_world import BattleWorld
from crbot.predictive_planner import PredictivePlanner
from tests.test_prediction import knowledge,world


class EnemyTrackingTests(unittest.TestCase):
    def setUp(self):
        self.p=PredictivePlanner(knowledge(),{})
        self.t=EnemyUnitTracker(self.p.sim)

    def body(self,card='hog_rider',confidence=.8,x=.28,side=-1,**extra):
        return dict(card_id=card,confidence=confidence,side=side,x=x,y=.55,
                    bbox=[x-.035,.52,x+.035,.58],**extra)

    def update(self,rows=(),candidates=(),at=100,generation=1):
        return self.t.update(list(rows),list(candidates),now=at,generation=generation)

    def test_threshold_and_low_confidence_class_flip_keep_identity(self):
        self.assertFalse(self.update([self.body(confidence=.799)])[2])
        first=self.update([self.body()],at=100.1)[0][0]
        second=self.update([self.body('valkyrie',.72,x=.29)],at=100.2)[0][0]
        self.assertEqual(second['card_id'],'hog_rider')
        self.assertEqual(second['identity_lock_id'],first['identity_lock_id'])
        self.assertEqual(second['visual_card_id'],'valkyrie')
        self.assertEqual(second['confidence'],.72)

    def test_weak_body_updates_existing_lock_but_never_starts_one(self):
        candidate=dict(candidate_card_id='giant',confidence=.3,side=-1,bbox=[.25,.52,.32,.58])
        self.assertFalse(self.update(candidates=[candidate])[2])
        self.update([self.body()],at=100.1)
        rows,remaining,tracks=self.update(candidates=[candidate],at=100.2)
        self.assertEqual(rows[0]['card_id'],'hog_rider')
        self.assertEqual(rows[0]['confidence'],.3)
        self.assertFalse(remaining)
        self.assertEqual(tracks[0]['position_state'],'visible')

    def test_occlusion_is_preview_only_and_expires_without_new_evidence(self):
        self.update([self.body()])
        rows,_,tracks=self.update(at=101)
        self.assertFalse(rows)
        self.assertEqual(tracks[0]['position_state'],'occluded')
        self.assertEqual(tracks[0]['last_visual_at'],100)
        self.assertFalse(self.update(at=103.01)[2])

    def test_allies_and_unknown_sides_cannot_extend_enemy(self):
        self.update([self.body()])
        self.update([self.body(side=1)],at=102)
        self.update(candidates=[dict(candidate_card_id='hog_rider',confidence=.9,side=None,bbox=[.25,.52,.32,.58])],at=102.5)
        self.assertFalse(self.update(at=103.1)[2])

    def test_reset_and_dead_health_release_locks(self):
        old=self.update([self.body()])[0][0]['identity_lock_id']
        self.assertFalse(self.update(at=100.1,generation=2)[2])
        new=self.update([self.body()],at=100.2,generation=2)[0][0]['identity_lock_id']
        self.assertNotEqual(old,new)
        rows,_,locks=self.update([self.body(hp_fraction=0,hp_confidence=.9)],at=100.3,generation=2)
        self.assertFalse(locks)
        self.assertEqual(rows[0]['hp_fraction'],0)

    def test_two_nearby_enemies_stay_separate_and_crossing_is_ambiguous(self):
        rows,_,locks=self.update([self.body(x=.25),self.body('valkyrie',x=.36)])
        self.assertEqual(len(locks),2)
        rows,_,locks=self.update([self.body(x=.305)],at=100.1)
        self.assertTrue(rows[0].get('identity_ambiguous'))
        self.assertTrue(all(t['position_state']=='occluded' for t in locks))

    def test_cached_frame_does_not_age_or_double_confirm_identity(self):
        first=self.update([self.body()])
        self.assertEqual(first,self.update(at=100))
        first[0][0]['card_id']='bad'
        self.assertEqual(self.update(at=100)[0][0]['card_id'],'hog_rider')

    def test_motion_gate_supports_fast_hog_across_planner_frames(self):
        first=self.update([self.body()])[0][0]
        body=self.body();body['y']=.60;body['bbox']=[.245,.57,.315,.63]
        second=self.update([body],at=101)[0][0]
        self.assertEqual(second['identity_lock_id'],first['identity_lock_id'])

    def test_world_retains_same_track_through_weak_class_and_occlusion(self):
        b=BattleWorld()
        args=dict(elapsed=20,elixir=5,hand=[(0,'knight')],costs={'hog_rider':4},seconds_per_elixir=2.8)
        rows,_,_=self.update([self.body(confidence=.79)])
        before=b.update(rows,now=100,**args)
        rows,_,_=self.update([self.body()],at=100.1)
        locked=b.update(rows,now=100.1,**args)
        self.assertEqual(len(locked.tracks),1)
        self.assertEqual(locked.tracks[0].track_id,before.tracks[0].track_id)
        rows,_,_=self.update(candidates=[dict(candidate_card_id='giant',confidence=.3,side=-1,bbox=[.25,.52,.32,.58])],at=100.2)
        weak=b.update(rows,now=100.2,**args)
        self.assertEqual(weak.tracks[0].card_id,'hog_rider')
        missing=b.update([],now=102,**args)
        scene=self.p.initial(missing,7,1)
        self.assertTrue(any(e.track_id==locked.tracks[0].track_id for e in scene.entities))
        self.assertEqual(missing.tracks[0].last_seen,100.2)


if __name__=='__main__':unittest.main()
