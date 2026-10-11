"""Convert pinned Labelme battles; entire battle groups stay in one split."""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from pathlib import Path

from PIL import Image

from .slice_dataset import SOURCE, REVISION


def label_unit(label: str, classes: list[str]) -> str | None:
    # Match longest first: evolution labels must retain their own identity.
    for name in sorted(classes, key=len, reverse=True):
        if re.fullmatch(re.escape(name) + r'(?:[01])?(?:_[a-z_-]+)?', label):
            return name
    return None


def convert_shapes(payload: dict, classes: list[str]) -> tuple[list[str], int]:
    width, height = float(payload['imageWidth']), float(payload['imageHeight'])
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    rows, ignored = [], 0
    for shape in payload.get('shapes', []):
        unit = label_unit(shape['label'], classes)
        if unit is None:
            ignored += 1
            continue
        if shape.get('shape_type') != 'rectangle' or len(shape['points']) != 2:
            raise ValueError('Unsupported annotation geometry')
        (ax, ay), (bx, by) = shape['points']
        x1, x2 = sorted((max(0, min(width, ax)), max(0, min(width, bx))))
        y1, y2 = sorted((max(0, min(height, ay)), max(0, min(height, by))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Empty annotation box')
        rows.append(f'{classes.index(unit)} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}')
    return rows, ignored


def export_real(source: Path, slices: Path, output: Path, *, per_battle=8, seed=20261011, catalog=None, focus_units=()) -> dict:
    import subprocess
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Unpinned source revision')
    if output.exists() or per_battle < 1:
        raise ValueError('Use a new output directory and positive per-battle limit')
    classes = json.loads((slices / 'manifest.json').read_text())['classes']
    if catalog is not None:
        from .pretrained_battlefield import unit_card_id
        classes = [name for name in classes if unit_card_id(name.replace('-', '_'), catalog)]
        if not classes:
            raise ValueError('No supported battlefield classes')
    arena = source / 'images/part2'
    groups = {}
    for path in sorted(arena.rglob('*.json')):
        group = path.parent.relative_to(arena).as_posix()
        if group == 'background':
            continue
        if path.with_suffix('.jpg').is_file():
            groups.setdefault(group, []).append(path)
    if len(groups) < 2:
        raise ValueError('At least two independent battles required')
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    validation = set(order[:max(1, round(len(order) * .2))])
    records, hashes, counts, ignored = [], {}, {'train': 0, 'val': 0}, 0
    for group in sorted(groups):
        split = 'val' if group in validation else 'train'
        paths = groups[group]
        # Uniform temporal sampling instead of selecting adjacent first frames.
        selected = [paths[min(len(paths)-1, int((i+.5)*len(paths)/min(per_battle,len(paths))))]
                    for i in range(min(per_battle, len(paths)))]
        # Rare units can occupy only a short interval and be missed by uniform
        # sampling. Add their labeled frames within the same battle split.
        focused = []
        if focus_units:
            for path in paths:
                data = json.loads(path.read_text(encoding='utf-8'))
                if any(label_unit(s['label'], classes) in focus_units for s in data.get('shapes', [])):
                    focused.append(path)
            if len(focused) > 64:
                focused = [focused[int((i+.5)*len(focused)/64)] for i in range(64)]
        selected = sorted(set(selected) | set(focused))
        for path in selected:
            payload = json.loads(path.read_text(encoding='utf-8'))
            rows, skipped = convert_shapes(payload, classes)
            ignored += skipped
            if catalog is not None and not rows:
                continue
            image = path.with_suffix('.jpg')
            with Image.open(image) as pixels:
                if pixels.size != (payload['imageWidth'], payload['imageHeight']):
                    raise ValueError(f'Annotation/image dimensions differ: {path}')
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes:
                if hashes[digest] != split:
                    raise ValueError('Identical image crosses battle split')
                continue
            hashes[digest] = split
            identifier = hashlib.sha256(path.relative_to(arena).as_posix().encode()).hexdigest()[:20]
            for folder in ('images', 'labels'):
                (output / folder / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, output / 'images' / split / (identifier+'.jpg'))
            (output / 'labels' / split / (identifier+'.txt')).write_text('\n'.join(rows)+'\n', encoding='utf-8')
            counts[split] += 1
            records.append(dict(group=group, split=split, sha256=digest,
                                source=path.relative_to(arena).as_posix(), objects=len(rows)))
    if not all(counts.values()):
        raise ValueError('Empty split')
    (output / 'data.yaml').write_text(f'path: {json.dumps(output.resolve().as_posix())}\ntrain: images/train\nval: images/val\nnames: {json.dumps(classes)}\n')
    result = dict(source=SOURCE, revision=REVISION, kind='real_battle_bootstrap', classes=classes,
                  counts=counts, battle_groups=len(groups), seed=seed, per_battle=per_battle,
                  ignored_annotations=ignored, records=records, battle_acceptance=False, promoted=False)
    result['units_only'] = catalog is not None
    result['focus_units'] = list(focus_units)
    (output / 'manifest.json').write_text(json.dumps(result, indent=2))
    shutil.copy2(source / 'LICENSE', output / 'LICENSE')
    return result
