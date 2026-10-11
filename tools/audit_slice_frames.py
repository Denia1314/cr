"""Read-only local frame audit: counts are observations, not measured accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from crbot.cards import CardCatalog
from crbot.slice_detector import SliceBattlefield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--reference-audit', type=Path, help='Reuse exact frozen frame names and hashes')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    adapter = SliceBattlefield(root, CardCatalog.load(root / 'data/cards.json'),
                              {'slice_model_path': args.model, 'device': 'auto', 'runtime_confidence': .7,
                               'slice_fallback_enabled': False})
    frames = sorted(args.frames.glob('*.jpg'))
    if not frames or args.limit < 1:
        raise ValueError('Need local frames and positive limit')
    chosen = [frames[min(len(frames)-1, int((i+.5)*len(frames)/min(args.limit,len(frames))))]
              for i in range(min(args.limit,len(frames)))]
    if args.reference_audit:
        reference = json.loads(args.reference_audit.read_text(encoding='utf-8'))
        chosen = []
        for record in reference['records']:
            frame = args.frames / record['frame']
            if hashlib.sha256(frame.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Reference frame changed')
            chosen.append(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for frame in chosen:
        with Image.open(frame) as image:
            observations = adapter.detect(image)
            arena = image.crop((0,0,image.width,round(image.height*.8))) if adapter.metadata.get('training_dataset_kind') == 'real_battle_bootstrap' else image
            raw = adapter.model.predict(arena, conf=.7, device=adapter.device, verbose=False)[0]
            Image.fromarray(raw.plot()[..., ::-1]).save(args.output / frame.name)
        records.append(dict(frame=frame.name, sha256=hashlib.sha256(frame.read_bytes()).hexdigest(),
                            raw_boxes=len(raw.boxes), accepted_units=len(observations),
                            observations=observations))
    result = dict(model=args.model, confidence=.7, frames=len(records),
                  frames_with_raw_boxes=sum(r['raw_boxes']>0 for r in records),
                  frames_with_accepted_units=sum(r['accepted_units']>0 for r in records),
                  total_accepted_units=sum(r['accepted_units'] for r in records),
                  accuracy_measured=False, battle_acceptance=False, records=records)
    (args.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__ == '__main__':
    main()
