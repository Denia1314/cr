import struct
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PIL import Image
from crbot.adb import MumuDevice,DeviceError
from crbot.frame_stream import LatestFrameStream,Frame
from crbot.engine import BotEngine
from crbot.battle_perception import HandCardMatch,LaneThreat


class FrameTests(unittest.TestCase):
    def test_capture_period_subtracts_capture_work_and_never_adds_backlog(self):
        for duration,expected_wait in ((.1,.15),(.4,0.)):
            stream=LatestFrameStream(lambda:Image.new('RGB',(2,2)),interval=.25)
            stop=Mock()
            stop.is_set.side_effect=[False,False,True]
            stream.stopped=stop
            starts=.25 if duration<.25 else duration
            with patch('crbot.frame_stream.time.monotonic',side_effect=[0.,duration,duration,starts,starts+duration,starts+duration]):
                stream._run()
            self.assertAlmostEqual(stop.wait.call_args.args[0],expected_wait)
            self.assertAlmostEqual(stream.last_start_interval,starts)
            self.assertEqual(stream.produced,2)
            self.assertAlmostEqual(stream.status()['actual_fps'],1/starts)

    def test_raw_headers_and_truncated_frame(self):
        for extra in (b'',struct.pack('<I',1)):
            data=struct.pack('<III',2,1,1)+extra+bytes([255,0,0,255,0,255,0,255])
            image=MumuDevice.decode_raw_screenshot(data)
            self.assertEqual(list(image.getdata()),[(255,0,0),(0,255,0)])
        with self.assertRaises(DeviceError):
            MumuDevice.decode_raw_screenshot(data[:-1])

    def test_raw_failure_uses_bounded_png_fallback(self):
        import io
        encoded=io.BytesIO()
        Image.new('RGB',(2,2),'red').save(encoded,format='PNG')
        device=object.__new__(MumuDevice)
        device.adb=Mock(side_effect=[b'unsupported raw',encoded.getvalue(),encoded.getvalue()])
        self.assertEqual(device.screenshot_fast().getpixel((0,0)),(255,0,0))
        device.screenshot_fast()
        self.assertEqual(device.capture_backend,'adb_png_fallback')
        self.assertEqual(device.adb.call_args.kwargs['timeout'],5)
        self.assertEqual(device.adb.call_args.args[0],['exec-out','screencap','-p'])

    def test_stream_returns_new_capture_after_click_and_discards_backlog(self):
        stream=LatestFrameStream(lambda:Image.new('RGB',(2,2)),interval=.005).start()
        try:
            first=stream.get()
            clicked=time.monotonic()
            after=stream.get(sequence=first.sequence,after=clicked)
            self.assertGreaterEqual(after.started,clicked)
            self.assertGreater(after.sequence,first.sequence)
            self.assertEqual(stream.status()['queue_capacity'],1)
        finally:
            stream.close()
        self.assertFalse(stream.thread.is_alive())

    def test_capture_error_is_propagated_not_replaced_by_stale_frame(self):
        def fail():raise DeviceError('lost transport')
        stream=LatestFrameStream(fail).start()
        try:
            with self.assertRaises(DeviceError):stream.get()
        finally:stream.close()

    def test_timeout_does_not_return_pre_click_frame(self):
        stream=LatestFrameStream(lambda:Image.new('RGB',(2,2))).start()
        try:
            with self.assertRaises(TimeoutError):stream.get(after=time.monotonic()+10,timeout=.02)
        finally:stream.close()


class RevalidationTests(unittest.TestCase):
    def setUp(self):
        self.engine=object.__new__(BotEngine)
        self.image=Image.new('RGB',(10,10))
        self.engine.frame_stream=SimpleNamespace(latest=lambda:Frame(self.image,time.monotonic(),time.monotonic(),2))
        self.engine._frame_sequence=1
        self.engine.config={'vision':{'elixir_roi':[0,0,1,1]}}
        self.engine.policy=SimpleNamespace(hand_recognizer=SimpleNamespace(recognize=lambda _: [HandCardMatch(0,'knight',.9,60,1,100,False)]),
            _perceive_threats=lambda *args:{'left':LaneThreat('left',.5,1,.6,'heavy',(),())})
        self.decision=SimpleNamespace(decision_engine='predictive',slot_index=0,card_id='knight',card_cost=3,lane='left',left_threat=.5,right_threat=0)

    def check(self,elixir=5):
        with patch('crbot.engine.estimate_elixir',return_value=(elixir,.9)):
            return self.engine._revalidate_prediction(Image.new('RGB',(10,10)),self.decision)[1]

    def test_unchanged_valid_action_survives(self):self.assertIsNone(self.check())
    def test_insufficient_elixir_is_rejected(self):self.assertEqual(self.check(2),'elixir_changed_before_send')
    def test_hand_change_is_rejected(self):
        self.decision.card_id='fireball'
        self.assertEqual(self.check(),'hand_changed_before_send')
    def test_disappeared_threat_is_rejected(self):
        self.engine.policy._perceive_threats=lambda *args:{'left':LaneThreat('left',0,0,0,'none',(),())}
        self.assertEqual(self.check(),'threat_disappeared_before_send')

    def test_full_elixir_development_survives_disappeared_threat(self):
        self.decision.action_intent='overflow_development'
        self.engine.policy._perceive_threats=lambda *args:{'left':LaneThreat('left',0,0,0,'none',(),())}
        self.assertIsNone(self.check(9.9))
        self.assertEqual(self.check(8),'development_resource_changed_before_send')
        self.assertEqual(self.check(2),'elixir_changed_before_send')
        self.decision.card_id='fireball'
        self.assertEqual(self.check(10),'hand_changed_before_send')

    def test_preparation_keeps_its_separate_resource_threshold(self):
        self.decision.action_intent='preparation'
        self.engine.policy._perceive_threats=lambda *args:{'left':LaneThreat('left',0,0,0,'none',(),())}
        self.assertIsNone(self.check(7))
        self.assertEqual(self.check(6),'development_resource_changed_before_send')

    def test_rejected_recheck_never_reaches_device_taps(self):
        from crbot.policy import BattleDecision
        decision=BattleDecision(0,[.1,.9],[.3,.6],'left','test',5,'vision',0,0,
                               card_id='fireball',card_cost=4,decision_engine='predictive',action_id='test')
        self.engine.device=Mock()
        self.engine.policy.prepare_action=lambda decision,snapshot:decision
        self.engine.policy.resolve_action=Mock()
        self.engine.response_timing=Mock()
        result=self.engine._execute_action(self.image,decision,{})
        self.assertEqual(result[-1],'hand_changed_before_send')
        self.engine.device.tap_normalized.assert_not_called()
        self.engine.policy.resolve_action.assert_called_once()


class HandCacheTests(unittest.TestCase):
    def test_changed_slots_share_gpu_submission_and_keep_slot_order(self):
        import numpy as np
        from crbot.battle_perception import UniversalHandRecognizer
        h=object.__new__(UniversalHandRecognizer)
        h.available=True
        h.templates={'knight':(SimpleNamespace(card_id='knight'), np.zeros((2,32),dtype=np.uint8))}
        h.vision={'card_slot_centers':[[0],[1],[2],[3]], 'card_empty_max_keypoints':0}
        h.orb=SimpleNamespace(detectAndCompute=Mock(return_value=([1,2],np.zeros((2,32),dtype=np.uint8))))
        h.gpu_matcher=SimpleNamespace(counts_many=Mock(side_effect=lambda queries,ratio:[[50] for q in queries]))
        h._slot_image=lambda image,center:np.full((2,2),image.getpixel((center[0],0))[0],dtype=np.uint8)
        image=Image.new('RGB',(4,2))
        first=h.recognize(image)
        self.assertEqual([m.slot_index for m in first],[0,1,2,3])
        self.assertEqual(len(h.gpu_matcher.counts_many.call_args.args[0]),4)
        self.assertEqual(h.recognize(image.copy()),first)
        self.assertEqual(h.gpu_matcher.counts_many.call_count,1)
        image.putpixel((2,0),(255,0,0))
        self.assertEqual(h.recognize(image),first)
        self.assertEqual(len(h.gpu_matcher.counts_many.call_args.args[0]),1)
        self.assertEqual(h.gpu_matcher.counts_many.call_count,2)

    def test_exact_slot_cache_invalidates_only_changed_slot(self):
        import numpy as np
        from crbot.battle_perception import UniversalHandRecognizer
        h=object.__new__(UniversalHandRecognizer)
        h.available=True
        h.templates={'dummy':object()}
        h.vision={'card_slot_centers':[[0],[1],[2],[3]]}
        h.orb=SimpleNamespace(detectAndCompute=Mock(return_value=([],None)))
        h._slot_image=lambda image,center:np.full((2,2),image.getpixel((center[0],0))[0],dtype=np.uint8)
        image=Image.new('RGB',(4,2))
        first=h.recognize(image)
        self.assertEqual(h.recognize(image.copy()),first)
        self.assertEqual(h.orb.detectAndCompute.call_count,4)
        image.putpixel((1,0),(255,0,0))
        h.recognize(image)
        self.assertEqual(h.orb.detectAndCompute.call_count,5)


class EngineStreamTests(unittest.TestCase):
    def test_run_closes_producer_on_error(self):
        from threading import Event
        engine=object.__new__(BotEngine)
        engine.stop_event=Event()
        engine.recognizer=SimpleNamespace(calibrated_names=lambda:{'offline_ai_marker'})
        engine.config={'game':{'package':'test'},'automation':{'latest_frame_capture':True}}
        engine.device=SimpleNamespace(launch_package=Mock(),screenshot=lambda:Image.new('RGB',(2,2)))
        engine.recorder=SimpleNamespace(run_dir='test')
        engine._sleep=lambda _:False
        def fail(_):
            engine.frame_stream.get()
            raise DeviceError('test failure')
        engine._run_single_marker=fail
        with self.assertRaises(DeviceError):engine.run()
        self.assertFalse(engine.frame_stream.thread.is_alive())


class DefenseCriticalPathTests(unittest.TestCase):
    def test_diagnostics_and_sample_encoding_follow_ready_action(self):
        from crbot.policy import BattleDecision
        engine = object.__new__(BotEngine)
        engine.completed_battles = 0
        engine.previous_battle_frame = None
        engine._last_screenshot_elapsed_s = .03
        engine.dry_run = True
        engine.policy = Mock()
        engine.policy.observe_replay_state.return_value = dict(allies_observed=False, observed_allies=[])
        decision = BattleDecision(0,[.1,.9],[.3,.6], 'left','test',5,'vision',0,0)
        engine.policy.decide.return_value = decision
        engine.response_timing = Mock()
        engine.response_timing_summary = Mock(return_value={})
        engine.recorder = Mock()
        order = []
        def execute(*args, **kwargs):
            order.append('action')
            return decision, None, None, None, None, None
        engine._execute_action = execute
        engine.recorder.record_battle_sample.side_effect = lambda *a, **k: order.append('sample')
        engine._record_prediction_status = lambda: order.append('diagnostics')
        with patch('crbot.engine.time.time', return_value=1234):
            engine._play_battle(Image.new('RGB', (10,10)))
        self.assertEqual(order, ['action','sample','diagnostics'])
        self.assertEqual(engine.recorder.record_battle_sample.call_args.kwargs['observed_at_unix'], 1234)
