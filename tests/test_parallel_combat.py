import time
import unittest
from dataclasses import replace
from concurrent.futures import Future
from unittest.mock import patch
import multiprocessing
from types import SimpleNamespace

from crbot.parallel_combat import CombatPool
from crbot.predictive_planner import PredictivePlanner
from tests.test_prediction import knowledge, world


class ParallelCombatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = knowledge()
        cls.config = dict(fast_defense=True, unified_tactics=True, combat_workers=2, combat_batch_size=3)
        cls.pool = CombatPool(cls.kb.payload, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()

    def test_real_workers_match_all_serial_branches_and_order(self):
        planner = PredictivePlanner(self.kb, self.config)
        for tracks in (world().tracks, ()):
            snapshot = world(tracks=tracks)
            phase = 'defend' if tracks else 'develop'
            scenarios = [(0,.65,False),(5,.85,True),(7,1.,False)]
            roots = planner.candidates(planner.initial(snapshot,7,1),1,limit=13)
            serial = [planner.evaluate_root(snapshot,r,scenarios,8,phase,time.perf_counter()+30) for r in roots]
            parallel = list(self.pool.rows(snapshot, roots, scenarios, 8, phase, time.perf_counter()+30))
            self.assertEqual(serial, parallel)
        self.assertTrue(self.pool.available, self.pool.error)
        self.assertEqual(len(self.pool.executing_pids), 2)
        self.assertLessEqual(self.pool.max_pending,2)

    def test_first_hand_round_is_returned_without_waiting_for_a_batch(self):
        planner=PredictivePlanner(self.kb,self.config)
        snapshot=world()
        roots=planner.candidates(planner.initial(snapshot,7,1),1,limit=9)
        first_round=1+len({r.card_id for r in roots if r.card_id})
        with patch.object(self.pool.executor,'submit',wraps=self.pool.executor.submit) as submit:
            list(self.pool.rows(snapshot,roots,[(7,1,False)],4,'defend',time.perf_counter()+30))
        sizes=[len(call.args[3]) for call in submit.call_args_list]
        self.assertEqual(sizes[:first_round],[1]*first_round)
        self.assertTrue(any(size>1 for size in sizes[first_round:]))

    def test_completion_order_does_not_wait_for_slow_first_future(self):
        from unittest.mock import Mock
        from crbot.battle_simulation import SimAction
        slow, fast = Future(), Future()
        fast.set_result((1, [{'label':'finished'}], True, 999))
        pool=object.__new__(CombatPool)
        pool.__dict__.update(workers=2,batch_size=4,pending=[],calls=0,timeouts=0,
                             completed=0,completed_combinations=0,max_pending=0,
                             pids=set(),executing_pids=set(),logged=True,error='')
        pool.executor=Mock()
        pool.executor.submit.side_effect=[slow,fast]
        stream=pool.rows(world(),[SimAction(),SimAction('knight',0,.3,.7)],[(7,1,False)],6,
                         'defend',time.perf_counter()+2,completion_order=True)
        try:
            self.assertEqual(next(stream),{'label':'finished'})
            self.assertFalse(slow.done())
        finally:
            stream.close()
        self.assertTrue(slow.cancelled())

    def test_completed_work_is_drained_even_after_another_job_times_out(self):
        from unittest.mock import Mock
        from crbot.battle_simulation import SimAction
        failed, finished=Future(),Future()
        failed.set_result((1,[],False,998))
        finished.set_result((1,[{'label':'usable'}],True,999))
        pool=object.__new__(CombatPool)
        pool.__dict__.update(workers=2,batch_size=4,pending=[],calls=0,timeouts=0,
                             completed=0,completed_combinations=0,max_pending=0,
                             pids=set(),executing_pids=set(),logged=True,error='')
        pool.executor=Mock()
        pool.executor.submit.side_effect=[failed,finished]
        collected=[]
        with patch('crbot.parallel_combat.time.perf_counter',side_effect=[99,99,101,101,101,101]):
            with self.assertRaises(TimeoutError):
                for item in pool.rows(world(),[SimAction(),SimAction('knight',0,.3,.7)],[],6,'defend',100,completion_order=True):
                    collected.append(item)
        self.assertEqual(collected,[{'label':'usable'}])

    def test_unordered_real_workers_preserve_serial_values(self):
        planner=PredictivePlanner(self.kb,self.config)
        snapshot=world()
        roots=planner.candidates(planner.initial(snapshot,7,1),1,limit=9)
        scenarios=[(7,1,False)]
        serial=[planner.evaluate_root(snapshot,r,scenarios,6,'defend',time.perf_counter()+30) for r in roots]
        parallel=list(self.pool.rows(snapshot,roots,scenarios,6,'defend',time.perf_counter()+30,completion_order=True))
        key=lambda r:(r['action']['slot'],r['action']['x'],r['action']['y'])
        self.assertEqual(sorted(serial,key=key),sorted(parallel,key=key))

    def test_expired_batch_recovers_for_next_revision(self):
        planner = PredictivePlanner(self.kb,self.config)
        snapshot = world()
        roots = planner.candidates(planner.initial(snapshot,7,1),1,limit=5)
        with self.assertRaises(TimeoutError):
            list(self.pool.rows(snapshot,roots,[(7,1.,False)],8,'defend',time.perf_counter()-1))
        later = replace(snapshot,revision=2)
        rows = list(self.pool.rows(later,roots,[(7,1.,False)],8,'defend',time.perf_counter()+30))
        self.assertEqual(len(rows),len(roots))

    def test_busy_old_frame_is_not_queued(self):
        pending = Future()
        self.pool.pending = [pending]
        try:
            with self.assertRaises(TimeoutError):
                list(self.pool.rows(world(),[],[],8,'defend',time.perf_counter()+30))
        finally:
            pending.cancel()
            self.pool.pending=[]

    def test_planner_integration_and_shutdown(self):
        planner = PredictivePlanner(self.kb,{**self.config,'parallel_combat':True,'budget_ms':2000,'positions_per_card':3})
        try:
            result = planner.plan(world())
            self.assertIn(result.status,{'ready','wait'})
            self.assertGreater(result.compute['parallel_combat']['completed_roots'],0)
            self.assertTrue(result.compute['parallel_combat']['available'])
        finally:
            planner.close()
        self.assertFalse(planner.combat_pool.available)
        self.assertFalse(set(planner.combat_pool.pids) & {p.pid for p in multiprocessing.active_children()})

    def test_startup_failure_uses_serial_without_losing_decisions(self):
        planner = PredictivePlanner(self.kb,{**self.config,'parallel_combat':True,'budget_ms':2000,'positions_per_card':3})
        with patch('crbot.parallel_combat.ProcessPoolExecutor', side_effect=OSError('process creation failed')):
            result = planner.plan(world())
        self.assertIn(result.status,{'ready','wait'})
        self.assertEqual(result.compute['parallel_combat']['last_mode'],'serial')
        self.assertIn('process creation failed',result.compute['parallel_combat']['error'])

    def test_engine_closes_pool_on_early_stop_and_error(self):
        from crbot.engine import BotEngine
        from unittest.mock import Mock
        for error in (None, ValueError('launch failed')):
            engine = object.__new__(BotEngine)
            planner = Mock()
            engine.policy = SimpleNamespace(planner=planner)
            engine._run_session = Mock(side_effect=error)
            if error:
                with self.assertRaises(ValueError):
                    engine.run()
            else:
                engine.run()
            planner.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
