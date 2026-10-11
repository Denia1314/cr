"""Import pinned KataCR sprites and export isolated synthetic YOLO datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

from PIL import Image

from .atomic_file import atomic_write

SOURCE = 'https://github.com/wty-yy/Clash-Royale-Detection-Dataset'
REVISION = '31b4151fedb1b914e99c3c122c16dca61cb2b905'
DEFAULT_SOURCE = 'training/external/clash-royale-detection-dataset'


def import_slices(source: Path, output: Path) -> dict:
    segment = source / 'images/segment'
    if not segment.is_dir():
        raise ValueError(f'Missing slice directory: {segment}')
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Source revision differs from the pinned dataset')
    records = []
    auxiliary = []
    for path in sorted(p for p in segment.glob('*/*') if p.suffix.lower() in {'.png', '.jpg'}):
        name = path.parent.name
        prefix = name + '_'
        is_auxiliary = name in {'background-items', 'backgrounds', 'small-text', 'big-text'}
        if not is_auxiliary and not path.stem.startswith(prefix):
            raise ValueError(f'Invalid slice name: {path}')
        side = None if is_auxiliary else path.stem[len(prefix):].split('_')[0]
        if not is_auxiliary and side not in {'0', '1'}:
            raise ValueError(f'Invalid affiliation: {path}')
        data = path.read_bytes()
        with Image.open(path) as image:
            image.load()
            if not is_auxiliary and ('A' not in image.getbands() or image.getchannel('A').getbbox() is None):
                raise ValueError(f'Slice has no nonempty alpha mask: {path}')
            size = list(image.size)
        digest = hashlib.sha256(data).hexdigest()
        target = output / 'sprites' / name / path.name
        atomic_write(target, data)
        (auxiliary if is_auxiliary else records).append(dict(path=target.relative_to(output).as_posix(), unit=name,
                            affiliation=None if side is None else int(side), size=size, sha256=digest))
    if not records:
        raise ValueError('No slices found')
    atomic_write(output / 'LICENSE', (source / 'LICENSE').read_bytes())
    result = dict(source=SOURCE, revision=revision, kind='external_slices',
                  count=len(records), auxiliary_count=len(auxiliary), total_images=len(records)+len(auxiliary),
                  classes=sorted({r['unit'] for r in records}),
                  battle_acceptance=False, promoted=False, records=records, auxiliary=auxiliary)
    atomic_write(output / 'manifest.json', (json.dumps(result, indent=2) + '\n').encode())
    return result


def export_synthetic(source: Path, output: Path, *, seed=20261011) -> dict:
    """One masked sprite per canvas; split by content hash before compositing.

    This is a bootstrap dataset, never a real-battle validation set.
    """
    manifest = json.loads((source / 'manifest.json').read_text())
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    classes = manifest['classes']
    ids = {name: i for i, name in enumerate(classes)}
    counts = {'train': 0, 'val': 0}
    splits = {'train': [], 'val': []}
    rng = random.Random(seed)
    for record in manifest['records']:
        path = source / record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Corrupt slice: {path}')
        split = 'val' if int(record['sha256'][:8], 16) % 5 == 0 else 'train'
        # Duplicate slice bytes always belong to the same split.
        splits[split].append(record['sha256'])
        with Image.open(path) as raw:
            sprite = raw.convert('RGBA')
        sprite = sprite.crop(sprite.getchannel('A').getbbox())
        scale = min(1.0, 400 / sprite.width, 400 / sprite.height)
        sprite = sprite.resize((max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale))))
        canvas = Image.new('RGB', (640, 640), tuple(rng.randrange(50, 150) for _ in range(3)))
        x, y = rng.randrange(641 - sprite.width), rng.randrange(641 - sprite.height)
        canvas.paste(sprite, (x, y), sprite)
        index = counts[split]
        for folder in ('images', 'labels'):
            (output / folder / split).mkdir(parents=True, exist_ok=True)
        canvas.save(output / 'images' / split / f'{index:06d}.jpg')
        row = f"{ids[record['unit']]} {(x+sprite.width/2)/640:.8f} {(y+sprite.height/2)/640:.8f} {sprite.width/640:.8f} {sprite.height/640:.8f}\n"
        (output / 'labels' / split / f'{index:06d}.txt').write_text(row)
        counts[split] += 1
    if not all(counts.values()):
        raise ValueError('Both training and validation slices are required')
    (output / 'data.yaml').write_text(f'path: {json.dumps(output.resolve().as_posix())}\ntrain: images/train\nval: images/val\nnames: {json.dumps(classes)}\n')
    result = dict(source=SOURCE, revision=manifest['revision'], kind='synthetic_bootstrap',
                  classes=classes, counts=counts, seed=seed, source_hashes=splits,
                  local_quality_validated=False, battle_acceptance=False, promoted=False)
    atomic_write(output / 'manifest.json', json.dumps(result, indent=2).encode())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('import', 'export', 'train'))
    parser.add_argument('--source', type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument('--output', type=Path, default=Path('training/external/katacr-slices'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--base-model', default='yolo11n.pt')
    args = parser.parse_args(argv)
    if args.action == 'import':
        result = import_slices(args.source, args.output)
        result = {k: v for k, v in result.items() if k not in {'records', 'auxiliary'}}
    elif args.action == 'export':
        result = export_synthetic(args.source, args.output)
        result.pop('source_hashes')
    else:
        from ultralytics import YOLO
        from .gpu import vision_device
        manifest = json.loads((args.source / 'manifest.json').read_text())
        if manifest.get('kind') != 'synthetic_bootstrap':
            raise ValueError('Train requires an exported synthetic dataset')
        model = YOLO(args.base_model)
        model.train(data=str((args.source / 'data.yaml').resolve()), epochs=args.epochs,
                    batch=args.batch, device=vision_device(args.device), workers=0,
                    project=str(args.output.resolve()), name='candidate', exist_ok=False)
        result = dict(model=str(model.trainer.best), source=SOURCE, revision=REVISION,
                      kind='external_slice_candidate', classes=manifest['classes'],
                      model_sha256=hashlib.sha256(Path(model.trainer.best).read_bytes()).hexdigest(),
                      local_quality_validated=False, battle_acceptance=False, promoted=False)
        atomic_write(Path(model.trainer.save_dir) / 'manifest.json', json.dumps(result, indent=2).encode())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
