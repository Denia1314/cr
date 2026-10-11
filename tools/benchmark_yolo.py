"""Read-only live YOLO throughput acceptance; never taps or starts a battle."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--windows', type=int, default=4)
    parser.add_argument('--window-s', type=float, default=5)
    parser.add_argument('--minimum-fps', type=float, default=10)
    args = parser.parse_args()
    if args.windows < 1 or args.window_s < 1 or args.minimum_fps <= 0:
        parser.error('Use positive windows/minimum FPS and a window of at least one second.')
    import cv2
    from crbot.adb import MumuDevice
    from crbot.cards import CardCatalog
    from crbot.frame_stream import LatestFrameStream
    from crbot.learned_perception import LearnedBattlefieldDetector
    from crbot.yolo_stream import YoloStream
    config = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    config['mumu']['auto_launch'] = False
    cv2.setNumThreads(max(1, int(config.get('vision', {}).get('opencv_threads', 1))))
    device = MumuDevice(config)
    device.connect()
    catalog_path = Path(config.get('dataset', {}).get('card_catalog', 'data/cards.json'))
    catalog = CardCatalog.load(catalog_path if catalog_path.is_absolute() else ROOT/catalog_path)
    frames = LatestFrameStream(device.screenshot_fast,
        config.get('timing', {}).get('capture_interval_s', 1/40)).start()
    worker = YoloStream(frames, lambda: LearnedBattlefieldDetector(ROOT, catalog, config.get('training', {})),
        catalog, config.get('automation', {}).get('yolo_target_fps', 20)).start()
    try:
        deadline = time.monotonic()+60
        while worker.processed < 10:
            if worker.error:
                raise RuntimeError(worker.error)
            if time.monotonic() >= deadline:
                raise TimeoutError('YOLO warm-up did not produce 10 successful detections.')
            time.sleep(.1)
        rows = []
        for index in range(args.windows):
            count, started = worker.processed, time.perf_counter()
            time.sleep(args.window_s)
            elapsed = time.perf_counter()-started
            completed = worker.processed-count
            row = dict(window=index+1, seconds=elapsed, completed_inferences=completed,
                measured_fps=completed/elapsed, **worker.status())
            rows.append(row)
            print(f"Window {index+1}: {row['measured_fps']:.2f} FPS; {completed} real inferences; "
                  f"last {row['inference_ms']} ms", flush=True)
        passed = all(r['measured_fps'] >= args.minimum_fps and not r['error'] for r in rows)
        report = ROOT/'reports/yolo-speed/acceptance.json'
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(dict(passed=passed, minimum_fps=args.minimum_fps,
            scope='live_captures_and_real_models_with_team_evidence; health_reading_in_policy',
            capture=frames.status(), windows=rows), ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"{'PASS' if passed else 'FAIL'}: {report}")
        return 0 if passed else 1
    finally:
        worker.close()
        frames.close()
        device.close_capture()


if __name__ == '__main__':
    raise SystemExit(main())
