"""Read-only packaged runtime checks; artifacts stay in the selected data directory."""
from __future__ import annotations

import importlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor


def worker_probe():
    from .parallel_combat import warm
    return {"pid": warm(), "frozen": bool(getattr(sys, "frozen", False))}


def run_checks(report: Path) -> int:
    result = {"ok": False, "frozen": bool(getattr(sys, "frozen", False)), "executable": sys.executable}
    try:
        from .app_paths import data_root, resource_root, learning_command
        from .config import load_config
        from .cards import CardCatalog
        from . import release_label
        result['release'] = release_label()
        modules = ['gui', 'engine', 'calibrate', 'annotate', 'demonstration', 'learning',
                   'imitation', 'replay_learning', 'training_sync', 'self_learning', 'grid_world_view']
        for name in modules:
            importlib.import_module('crbot.'+name)
        result['modules'] = modules
        # The interactive path must collect before spending minutes on a candidate.
        from .self_learning import SelfLearningService
        with tempfile.TemporaryDirectory(prefix='royal-start-check-') as tmp:
            service = SelfLearningService(Path(tmp), {'self_learning': {'enabled': True}},
                                          defer_initial_training=True)
            started = time.perf_counter()
            assert not service.boundary()
            result['first_battle_gate'] = {'candidate_training_deferred': True,
                'milliseconds': round((time.perf_counter() - started) * 1000, 3),
                'audit_every_battles': service.settings['audit_every_battles']}
        result['gui_module'] = importlib.import_module('crbot.gui').__file__
        config, _ = load_config(data_root()/'config.json')
        result['cards'] = len(CardCatalog.load(data_root()/'data/cards.json').cards)
        import torch
        import numpy as np
        import cv2
        result['torch'] = torch.__version__
        result['opencv'] = cv2.__version__
        assert not cv2.ORB_create().detectAndCompute(np.zeros((64,64),np.uint8),None)[0]
        device = 'cuda' if os.environ.get('CRBOT_COMPUTE_DEVICE') != 'cpu' and torch.cuda.is_available() else 'cpu'
        x = torch.ones((32,32), device=device)
        assert (x@x).sum().item() == 32768
        model = torch.nn.Linear(4,1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        model(torch.ones((2,4),device=device)).sum().backward()
        optimizer.step()
        result['compute'] = {'device': device, 'matmul': True, 'gradient_step': True}
        from ultralytics import YOLO
        detector = YOLO('yolo11n.yaml')
        outputs = detector.predict(np.zeros((64,64,3),np.uint8), imgsz=64, device=device, verbose=False)
        assert len(outputs) == 1
        result['yolo_inference'] = True
        from .battlefield_assets import assets_ready
        from .pretrained_battlefield import PretrainedBattlefield
        from PIL import Image
        assert assets_ready(data_root()), 'Missing or invalid pinned public battlefield baseline'
        battlefield = PretrainedBattlefield(data_root(), CardCatalog.load(data_root()/'data/cards.json'), {'device': device})
        battlefield.detect(Image.new('RGB', (540,960)))
        result['battlefield'] = battlefield.status()
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            workers = [f.result(timeout=90) for f in [pool.submit(worker_probe), pool.submit(worker_probe)]]
        assert all(w['pid'] != os.getpid() for w in workers)
        result['workers'] = workers
        from .parallel_combat import CombatPool
        from .battle_world import WorldSnapshot, Track
        from .knowledge import KnowledgeBase
        from .predictive_planner import PredictivePlanner
        kb = KnowledgeBase.load(data_root()/'data/battle_knowledge.json')
        settings = {'fast_defense': True, 'unified_tactics': True, 'combat_workers': 2, 'combat_batch_size': 2}
        snapshot = WorldSnapshot(revision=1, at=100., elapsed=20., elixir=7., enemy_elixir=(0.,7.),
            hand=((0,'knight'),(1,'musketeer'),(2,'cannon'),(3,'fireball')),
            tracks=(Track(1,'giant',-1,.28,.5,99,100,.9,3),), enemy_seen=('giant',), enemy_recent=(), uncertainty=())
        planner = PredictivePlanner(kb, settings)
        roots = planner.candidates(planner.initial(snapshot,7,1.),1,limit=3)
        scenarios = [(7,1.,False)]
        pool = CombatPool(kb.payload, settings)
        try:
            assert pool.available, pool.error
            serial = [planner.evaluate_root(snapshot, r, scenarios, 2, 'defend', time.perf_counter()+30) for r in roots]
            parallel = list(pool.rows(snapshot, roots, scenarios, 2, 'defend', time.perf_counter()+30))
            assert serial == parallel
            result['combat'] = {'serial_parallel_equal': True, 'branches': len(serial), 'worker_pids': sorted(pool.pids)}
        finally:
            pool.close()
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        for tool in ('git', 'gh'):
            checked = subprocess.run([tool, '--version'], capture_output=True, text=True, creationflags=flags, timeout=30)
            assert checked.returncode == 0, checked.stderr
            result[tool] = checked.stdout.splitlines()[0]
        if getattr(sys, 'frozen', False):
            # Execute the real autonomous-learning dispatch in a disposable paused store.
            with tempfile.TemporaryDirectory(prefix='royal-learning-check-') as tmp:
                root = Path(tmp)
                from .learning_store import LearningStore, atomic_json
                store = LearningStore(root)
                atomic_json(store.root/'control.json', {'paused': True})
                request = root/'request.json'
                request.write_text('{}', encoding='utf-8')
                worker_log = root/'worker.log'
                run = subprocess.run(learning_command(root, request, worker_log), creationflags=flags, timeout=90)
                assert run.returncode == 0, worker_log.read_text(encoding='utf-8')
                state = store.current_status()
                assert state['phase'] in ('paused', 'collector'), state
                result['learning_worker'] = state['phase']
        result['bundle_root'] = str(resource_root())
        result['ok'] = True
    except Exception:
        result['error'] = traceback.format_exc()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if result['ok'] else 1
