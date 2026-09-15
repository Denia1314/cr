"""Persistent, bounded CPU workers; only the parent selects or executes actions."""
from __future__ import annotations

import os
import time
import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool

_planner = None


def initialize(payload, config, barrier):
    global _planner
    from .knowledge import KnowledgeBase
    from .predictive_planner import PredictivePlanner
    _planner = PredictivePlanner(KnowledgeBase(payload), {**config, 'parallel_combat': False, 'gpu_placement': False})
    if config.get('gpu_placement', False):
        # Enemy-response shortlists retain the same batched ranking semantics,
        # without creating a CUDA context in each CPU worker.
        from .gpu_placement import PlacementBatch
        _planner.sim.placement_batch = PlacementBatch('cpu', trajectory=bool(config.get('gpu_trajectory_screening',False)))
    # Populate immutable mechanics caches before any frame deadline starts.
    for name in _planner.kb.units:
        _planner.kb.unit(name, _planner.sim.level)
    barrier.wait(timeout=30)


def warm():
    # Start all workers before a frame's short decision deadline begins.
    time.sleep(.03)
    return os.getpid()


def evaluate_batch(revision, world, roots, scenarios, horizon, phase, deadline):
    rows = []
    states = None
    if time.perf_counter() < deadline:
        states = [_planner.initial(world, cost, hp) for cost, hp, _ in scenarios]
    for root in roots:
        try:
            rows.append(_planner.evaluate_root(world, root, scenarios, horizon, phase, deadline, states))
        except TimeoutError:
            return revision, rows, False, os.getpid()
    return revision, rows, True, os.getpid()


class CombatPool:
    def __init__(self, payload, config):
        self.workers = max(1, min(8, int(config.get('combat_workers', 4)), max(1,(os.cpu_count() or 2)-1)))
        self.batch_size = max(1, min(16, int(config.get('combat_batch_size', 4))))
        self.executor = None
        self.pending = []
        self.error = ''
        self.pids = set()
        self.executing_pids = set()
        self.logged = False
        self.completed = 0
        self.calls = 0
        self.timeouts = 0
        self.max_pending = 0
        if self.workers < 2:
            self.error = 'insufficient_cpu_cores_use_serial'
            return
        try:
            context = multiprocessing.get_context('spawn')
            barrier = context.Barrier(self.workers)
            self.executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=context,
                                                initializer=initialize, initargs=(payload, config, barrier))
            futures = [self.executor.submit(warm) for _ in range(self.workers)]
            self.pids.update(f.result(timeout=30) for f in futures)
        except (OSError, RuntimeError, FutureTimeout) as exc:
            self.error = str(exc)
            self.close()

    @property
    def available(self):
        return self.executor is not None and not self.error

    def rows(self, world, roots, scenarios, horizon, phase, deadline):
        self.calls += 1
        if any(not future.done() for future in self.pending):
            # Never queue a new frame behind obsolete work.
            self.timeouts += 1
            raise TimeoutError('previous frame still draining')
        self.pending = []
        queue = deque()
        offset = 0
        try:
            while offset < len(roots) or queue:
                while offset < len(roots) and len(queue) < self.workers:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError()
                    batch = roots[offset:offset+self.batch_size]
                    offset += len(batch)
                    future = self.executor.submit(evaluate_batch, world.revision, world, batch, scenarios, horizon, phase, deadline)
                    queue.append(future)
                    self.pending.append(future)
                    self.max_pending = max(self.max_pending, len(queue))
                # Submission order preserves the serial prefix and spatial/card fairness.
                future = queue.popleft()
                revision, rows, complete, pid = future.result(timeout=max(0, deadline-time.perf_counter()))
                if revision != world.revision:
                    raise ValueError('worker returned another observation revision')
                self.pids.add(pid)
                self.executing_pids.add(pid)
                self.completed += sum(row is not None for row in rows)
                if len(self.executing_pids) > 1 and not self.logged:
                    print(f'[并行推演] 已由 {len(self.executing_pids)} 个 CPU 工作进程返回详细战斗结果。')
                    self.logged = True
                yield from rows
                if not complete:
                    raise TimeoutError()
        except FutureTimeout:
            self.timeouts += 1
            raise TimeoutError('parallel combat deadline') from None
        except (BrokenProcessPool, OSError, RuntimeError) as exc:
            self.error = str(exc)
            # The caller discards this incomplete frame; next frame uses serial.
            raise TimeoutError('parallel combat unavailable') from exc
        finally:
            for future in self.pending:
                if not future.done():
                    future.cancel()

    def status(self):
        return dict(enabled=True, available=self.available, workers=self.workers, batch_size=self.batch_size,
                    worker_pids=sorted(self.executing_pids), warmed_pids=sorted(self.pids), calls=self.calls, completed_roots=self.completed,
                    deadline_exits=self.timeouts, max_pending_batches=self.max_pending, error=self.error,
                    scope='coarse_detailed_combat', refinement_device='parent_cpu')

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
