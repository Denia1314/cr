"""Read-only live capture/recognition benchmark. Never launches battles or taps."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crbot.adb import MumuDevice
from crbot.battle_perception import UniversalHandRecognizer
from crbot.cards import CardCatalog
from crbot.config import load_config, resolve_project_path
from crbot.frame_stream import LatestFrameStream
from crbot.perception_stream import PerceptionStream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    config, path = load_config(args.config)
    config['mumu']['auto_launch'] = False
    device = MumuDevice(config)
    device.connect()
    catalog = CardCatalog.load(resolve_project_path(path, config['dataset']['card_catalog']))
    recognizer = UniversalHandRecognizer(catalog, config['vision'])
    frames = LatestFrameStream(device.screenshot_fast, config['timing'].get('capture_interval_s', 1/30)).start()
    perception = PerceptionStream(frames, recognizer).start()
    try:
        first = perception.get(timeout=30)
        started = time.monotonic()
        initial_frames, initial_processed = frames.produced, perception.processed
        sequence = first.frame.sequence
        ages, work = [], []
        while time.monotonic() - started < max(1, args.seconds):
            result = perception.get(sequence=sequence)
            sequence = result.frame.sequence
            ages.append(time.monotonic()-result.frame.started)
            work.append(perception.status()['work_s'])
        elapsed = time.monotonic()-started
        ages.sort()
        print(json.dumps(dict(duration_s=elapsed, capture_fps=(frames.produced-initial_frames)/elapsed,
                             recognition_fps=(perception.processed-initial_processed)/elapsed,
                             recognition_work_ms=sum(work)/len(work)*1000,
                             frame_age_p95_ms=ages[min(len(ages)-1, int(len(ages)*.95))]*1000,
                             capture_backend=device.capture_backend,
                             perception=perception.status(), hand=recognizer.compute_status()), indent=2))
    finally:
        perception.close()
        frames.close()
        device.close_capture()


if __name__ == '__main__':
    main()
