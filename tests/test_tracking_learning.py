import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image,ImageDraw
from crbot.tracking_learning import train,pairs,load_model,validate_rows,prepare
from crbot.enemy_unit_tracking import EnemyUnitTracker
from crbot.predictive_planner import PredictivePlanner
from tests.test_prediction import knowledge


class TrackingLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.rows=[]
        for group,split in enumerate(('train','val','test')):
            for i in range(25):
                path=self.root/f'{group}-{i}.png'
                image=Image.new('RGB',(100,100),(group+3,i+3,20));draw=ImageDraw.Draw(image)
                swapped=(i//2)%2==1
                draw.rectangle((25,50,30,60),fill='blue' if swapped else 'red');draw.rectangle((33,50,38,60),fill='red' if swapped else 'blue');image.save(path)
                self.rows.append(dict(battle=str(group),split=split,at=i*.1,image=str(path),reviewed=True,
                    units=[dict(track_id=str(j),card_id='hog_rider',side=-1,bbox=b)
                           for j,b in enumerate(([.33,.50,.38,.60],[.25,.50,.30,.60]) if swapped else ([.25,.50,.30,.60],[.33,.50,.38,.60]))]))
    def tearDown(self):self.temp.cleanup()

    def test_actual_supervised_training_and_frozen_holdouts(self):
        output=self.root/'candidate.json';model=train(self.rows,output,epochs=1000)
        self.assertTrue(model['qualified'])
        self.assertEqual(model['test']['fp'],0)
        self.assertGreaterEqual(model['test']['recall'],.5)
        self.assertFalse(model['battle_accepted'])
        self.assertEqual(load_model(output)['weights'],model['weights'])

    def test_unreviewed_identity_change_and_battle_leakage_rejected(self):
        for mutation in ('review','identity','split'):
            rows=json.loads(json.dumps(self.rows))
            if mutation=='review':rows[0]['reviewed']=False
            if mutation=='identity':rows[1]['units'][0]['card_id']='valkyrie'
            if mutation=='split':rows[0]['split']='test'
            with self.assertRaises(ValueError):validate_rows(rows)

    def test_image_leakage_and_insufficient_samples_rejected(self):
        rows=json.loads(json.dumps(self.rows));rows[25]['image']=rows[0]['image']
        with self.assertRaises(ValueError):validate_rows(rows)
        rows=[r for r in self.rows if r['at']<.2]
        with self.assertRaises(ValueError):train(rows,self.root/'too-small.json')

    def test_unqualified_candidate_not_loaded(self):
        path=self.root/'bad.json';path.write_text(json.dumps(dict(schema=1,qualified=False)),encoding='utf-8')
        with self.assertRaises(ValueError):load_model(path)
        p=PredictivePlanner(knowledge(),dict(tracking_association_model=str(path)))
        self.assertIsNone(p.sim.tracking_association_model)
        self.assertTrue(p.tracking_model_error)

    def test_continuous_capture_is_bounded_and_never_auto_reviewed(self):
        import time
        from crbot.tracking_capture import TrackingCapture
        capture=TrackingCapture(self.root/'capture',fps=2,limit=2)
        image=Image.new('RGB',(20,20),'red')
        try:
            capture.submit(image,100,-1,[])
            capture.submit(image,100,1,[dict(card_id='hog_rider')])
            capture.submit(image,100.1,1,[])
            deadline=time.monotonic()+2
            while capture.saved<1 and time.monotonic()<deadline:time.sleep(.01)
            capture.submit(image,101,1,[])
            capture.submit(image,102,1,[])
        finally:capture.close()
        self.assertEqual(capture.saved,2)
        rows=[json.loads(l) for l in (capture.root/'1/frames.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(all(not r['reviewed'] and r['units']==[] for r in rows))
        with Image.open(rows[0]['image']) as saved:self.assertGreater(saved.getpixel((0,0))[0],240)
        self.assertFalse((capture.root/'-1').exists())

    def test_prepare_creates_reviewable_page_without_overwriting_labels(self):
        captures=self.root/'captures'
        for group in range(3):
            folder=captures/'session'/str(group);folder.mkdir(parents=True)
            row=dict(self.rows[group*25],reviewed=False,units=[],proposals=[])
            (folder/'frames.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
        output=self.root/'review'
        report=prepare(self.root/'no-runs',output,captures)
        self.assertEqual(report['battles'],3)
        self.assertFalse(report['ready_for_training'])
        self.assertIn('data:image/jpeg;base64,',(output/'review.html').read_text(encoding='utf-8'))
        with self.assertRaises(ValueError):prepare(self.root/'no-runs',output,captures)

    def test_loaded_association_disambiguates_same_class_crossing(self):
        p=PredictivePlanner(knowledge(),{});path=self.root/'model.json'
        train(self.rows,path,epochs=1000);p.sim.tracking_association_model=load_model(path)
        tracker=EnemyUnitTracker(p.sim)
        def row(box):return dict(card_id='hog_rider',side=-1,confidence=.8,bbox=box,x=(box[0]+box[2])/2,y=.55)
        a,b=self.rows[0]['units'];image=Image.open(self.rows[0]['image'])
        first=tracker.update([row(a['bbox']),row(b['bbox'])],[],now=100,generation=1,image=image)[0]
        # Swap appearance positions: geometric distances are ambiguous, but the
        # trained scorer should follow the original appearance identities.
        swapped=image.copy();draw=ImageDraw.Draw(swapped)
        draw.rectangle((25,50,30,60),fill='blue');draw.rectangle((33,50,38,60),fill='red')
        second=tracker.update([row(a['bbox']),row(b['bbox'])],[],now=100.3,generation=1,image=swapped)[0]
        keyed={r['identity_lock_id']:r for r in second if 'identity_lock_id' in r}
        self.assertEqual(len(keyed),2)
        self.assertGreater(keyed[first[0]['identity_lock_id']]['x'],first[0]['x'])
        image.close()


if __name__=='__main__':unittest.main()
