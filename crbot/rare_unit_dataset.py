"""Add annotated rare-unit frames from training battles; freeze the validation split."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image

from .real_battle_dataset import convert_shapes
from .slice_dataset import REVISION


def class_counts(base, split):
    return Counter(int(line.split()[0]) for path in (base/'labels'/split).glob('*.txt')
                   for line in path.read_text().splitlines() if line.strip())


def enrich_real(base, source, output, *, target_boxes=80, max_extra=400, frame_gap=15):
    base, source, output = (Path(p).resolve() for p in (base, source, output))
    if output.exists() or output.is_relative_to(base) or min(target_boxes, max_extra, frame_gap) < 1:
        raise ValueError('Use a new separate output and positive limits')
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != REVISION:
        raise ValueError('Unpinned source revision')
    manifest = json.loads((base/'manifest.json').read_text())
    if manifest.get('kind') != 'real_battle_bootstrap' or manifest.get('revision') != REVISION:
        raise ValueError('Pinned labeled real-battle base required')
    records = manifest['records']
    train_groups = {r['group'] for r in records if r['split'] == 'train'}
    val_groups = {r['group'] for r in records if r['split'] == 'val'}
    if train_groups & val_groups:
        raise ValueError('Battle group crosses splits')
    before = class_counts(base, 'train')
    validation_counts = class_counts(base, 'val')
    counts = before.copy()
    seen = {r['sha256'] for r in records}
    existing = {r['source'] for r in records}
    arena = source/'images/part2'
    candidates = []
    for group in sorted(train_groups):
        folder = (arena/group).resolve()
        if not folder.is_relative_to(arena.resolve()):
            raise ValueError('Battle source escapes root')
        for path in sorted(folder.glob('*.json')):
            if path.relative_to(arena).as_posix() in existing or not path.with_suffix('.jpg').is_file():
                continue
            payload = json.loads(path.read_text(encoding='utf-8'))
            rows, _ = convert_shapes(payload, manifest['classes'])
            units = Counter(int(r.split()[0]) for r in rows)
            if any(before[i] < target_boxes for i in units):
                candidates.append((path, group, payload, rows, units))
    added = []
    chosen_frames = {}
    while candidates and len(added) < max_extra:
        def gain(item):
            return sum(min(n, max(0, target_boxes-counts[i])) / max(1, counts[i])
                       for i, n in item[4].items())
        index = max(range(len(candidates)), key=lambda i: gain(candidates[i]))
        path, group, payload, rows, units = candidates.pop(index)
        if not gain((path, group, payload, rows, units)):
            break
        frame = int(path.stem) if path.stem.isdigit() else None
        if frame is not None and any(abs(frame-f) < frame_gap for f in chosen_frames.get(group, [])):
            continue
        image = path.with_suffix('.jpg')
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in seen:
            continue
        with Image.open(image) as pixels:
            if pixels.size != (payload['imageWidth'], payload['imageHeight']):
                raise ValueError('Annotation/image dimensions differ')
        seen.add(digest)
        if frame is not None:
            chosen_frames.setdefault(group, []).append(frame)
        counts.update(units)
        added.append((path, image, group, rows, digest))
    # Finish all source validation before creating the replacement dataset.
    shutil.copytree(base, output, ignore=shutil.ignore_patterns('*.cache'))
    for path, image, group, rows, digest in added:
        identifier = hashlib.sha256(path.relative_to(arena).as_posix().encode()).hexdigest()[:20]
        shutil.copy2(image, output/'images/train'/f'{identifier}.jpg')
        (output/'labels/train'/f'{identifier}.txt').write_text('\n'.join(rows)+'\n')
        records.append(dict(group=group, split='train', sha256=digest,
                            source=path.relative_to(arena).as_posix(), objects=len(rows)))
    manifest['records'] = records
    manifest['counts']['train'] += len(added)
    manifest['enrichment'] = dict(kind='annotated_training_battles_only', added_images=len(added),
        target_boxes=target_boxes, max_extra=max_extra, frame_gap=frame_gap,
        validation_frozen=True, base=str(base),
        classes=[dict(name=name, before=before[i], after=counts[i],
                      validation=validation_counts[i]) for i, name in enumerate(manifest['classes'])])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output/'data.yaml').write_text(f'path: {json.dumps(output.as_posix())}\ntrain: images/train\nval: images/val\nnames: {json.dumps(manifest["classes"])}\n')
    return manifest['enrichment']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('training/exports/katacr-real-units-v3'))
    parser.add_argument('--source', type=Path, default=Path('training/external/clash-royale-detection-dataset'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-boxes', type=int, default=80)
    parser.add_argument('--max-extra', type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(enrich_real(args.base, args.source, args.output,
                               target_boxes=args.target_boxes, max_extra=args.max_extra), indent=2))


if __name__ == '__main__':
    main()
