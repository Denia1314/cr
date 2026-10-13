import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from PIL import Image

from crbot.frame_stream import LatestFrameStream
from crbot.yolo_stream import YoloStream


class Detector:
    available = True
    error = ''
    last_detection_succeeded = True
    observed_allies = []
    observed_enemies = []

    def __init__(self):
        self.calls = 0

    def detect(self, image, fallback, *, include_health=True):
        self.include_health = include_health
        self.calls += 1
        self.observed_allies = [dict(card_id='knight', side=1, bbox=[.2,.3,.4,.5], pixel=image.getpixel((0,0))[0])]
        self.unconfirmed_units=[dict(bbox=[.6,.4,.8,.5],confidence=.36,side=None,
            observation_status='unconfirmed',pixel=image.getpixel((0,0))[0])]

    def status(self):
        return dict(inference_calls=self.calls, version='fixture')


class YoloStreamTests(unittest.TestCase):
    def test_enemy_identity_stays_locked_in_continuous_preview(self):
        from crbot.own_unit_tracking import OwnUnitTracker
        from crbot.predictive_planner import PredictivePlanner
        from tests.test_prediction import knowledge
        class EnemyDetector(Detector):
            def detect(self,image,fallback,*,include_health=True):
                self.calls+=1
                self.observed_allies=[];self.unconfirmed_units=[]
                self.observed_enemies=[dict(card_id='hog_rider' if self.calls==1 else 'valkyrie',side=-1,
                    confidence=.8 if self.calls==1 else .72,bbox=[.25,.52,.32,.58],x=.285,y=.55,
                    pixel=image.getpixel((0,0))[0])]
        counter=[0]
        def capture():
            counter[0]=(counter[0]+1)%255
            return Image.new('RGB',(8,8),(counter[0],0,0))
        frames=LatestFrameStream(capture,.01).start()
        detector=EnemyDetector()
        tracker=OwnUnitTracker(PredictivePlanner(knowledge(),{}).sim)
        worker=YoloStream(frames,lambda:detector,None,20,identity_tracker=tracker,deployment_source=lambda:(1,())).start()
        try:
            self.wait_for(lambda:worker.processed>=3)
            image,packet=worker.latest()
            row=packet['detections'][0]
            self.assertEqual(row['card_id'],'hog_rider')
            self.assertEqual(row['visual_card_id'],'valkyrie')
            self.assertEqual(row['confidence'],.72)
            self.assertEqual(row['pixel'],image.getpixel((0,0))[0])
            self.assertEqual(detector.observed_enemies[0]['card_id'],'valkyrie')
        finally:
            worker.close();frames.close()

    def wait_for(self, check):
        deadline = time.monotonic() + 3
        while not check() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(check())

    def test_planning_does_not_block_new_detections_and_frames_stay_paired(self):
        counter = 0
        def capture():
            nonlocal counter
            counter = (counter+1) % 255
            return Image.new('RGB', (8,8), (counter,0,0))
        frames = LatestFrameStream(capture, .01).start()
        detector = Detector()
        worker = YoloStream(frames, lambda: detector, None, 20).start()
        try:
            self.wait_for(lambda: worker.processed >= 6)
            image, packet = worker.latest()
            self.assertEqual(packet['detections'][0]['pixel'], image.getpixel((0,0))[0])
            self.assertEqual(packet['unconfirmed_units'][0]['pixel'],image.getpixel((0,0))[0])
            self.assertEqual(packet['model']['inference_calls'], packet['revision'])
            self.assertGreater(packet['capture_sequence'], packet['revision'])
            self.assertGreater(worker.status()['skipped_capture_frames'], 0)
            self.assertGreater(worker.status()['actual_fps'], 0)
            self.assertEqual(worker.status()['queue_capacity'], 1)
            self.assertFalse(detector.include_health)
            previous = packet['revision']
            self.wait_for(lambda: worker.processed >= previous+3)
            self.assertGreater(worker.latest()[1]['revision'], previous)
        finally:
            worker.close(); frames.close()
        self.assertFalse(worker.thread.is_alive())

    def test_failed_inference_is_not_counted_as_a_new_successful_frame(self):
        frames = LatestFrameStream(lambda: Image.new('RGB',(8,8)), .01).start()
        detector = Detector()
        detector.last_detection_succeeded = False
        detector.error = 'fixture failure'
        worker = YoloStream(frames, lambda: detector, None).start()
        try:
            self.wait_for(lambda: bool(worker.error))
            self.assertEqual(worker.processed, 0)
            self.assertIsNone(worker.latest())
            self.assertEqual(worker.status()['actual_fps'], 0)
            self.assertIn('fixture failure', worker.status()['error'])
        finally:
            worker.close(); frames.close()

    def test_old_or_failed_results_do_not_look_live(self):
        worker = YoloStream(None, None, None)
        worker.result = (object(), dict(at=time.monotonic()-1, finished_at=time.monotonic()-1))
        self.assertIsNone(worker.latest())
        self.assertEqual(worker.status()['actual_fps'], 0)
        worker.error = 'capture stopped'
        self.assertIsNone(worker.latest())

    def test_continuous_identity_binding_keeps_current_frame_and_raw_model_data(self):
        from crbot.own_unit_tracking import OwnUnitTracker
        from crbot.predictive_planner import PredictivePlanner
        from tests.test_prediction import knowledge
        counter=0
        def capture():
            nonlocal counter
            counter=(counter+1)%255
            return Image.new('RGB',(8,8),(counter,0,0))
        event=dict(id='confirmed-stream',card_id='mega_knight',x=.7,y=.45,at=time.monotonic()-.02)
        frames=LatestFrameStream(capture,.01).start()
        detector=Detector()
        tracker=OwnUnitTracker(PredictivePlanner(knowledge(),{}).sim)
        worker=YoloStream(frames,lambda:detector,None,20,identity_tracker=tracker,
            deployment_source=lambda:(1,(event,))).start()
        try:
            self.wait_for(lambda:worker.processed>=4)
            image,packet=worker.latest()
            bound=next(d for d in packet['detections'] if d.get('identity_lock_id'))
            self.assertEqual(bound['card_id'],'mega_knight')
            self.assertEqual(bound['pixel'],image.getpixel((0,0))[0])
            key=bound['identity_lock_id']
            self.assertEqual(detector.observed_allies[0]['card_id'],'knight')
            self.assertEqual(len(detector.unconfirmed_units),1)
            previous=packet['revision']
            self.wait_for(lambda:worker.processed>=previous+2)
            self.assertEqual(worker.latest()[1]['identity_tracks'][0]['identity_lock_id'],key)
        finally:
            worker.close();frames.close()

    def test_gui_uses_continuous_actual_results_instead_of_decision_snapshots(self):
        from crbot.gui import RoyalTrainerApp
        app = RoyalTrainerApp.__new__(RoyalTrainerApp)
        paired = (object(), dict(revision=100))
        app.preview_mode = Mock(); app.preview_mode.get.return_value = 'YOLO识别'
        app.current_image = object()
        app.engine = SimpleNamespace(yolo_stream=Mock(), policy=SimpleNamespace(yolo_frame=(object(), dict(revision=1))))
        app.engine.yolo_stream.latest.return_value = paired
        self.assertIs(app._grid_preview_source(), paired)
        app.engine.yolo_stream.latest.return_value = None
        self.assertEqual(app._grid_preview_source(), (app.current_image, None))
