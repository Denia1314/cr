"""Choose a compatible saved run instead of resetting to a fixed base model."""
import hashlib
import json
import csv
from pathlib import Path


def dataset_fingerprint(source):
    source = Path(source)
    digest = hashlib.sha256((source/'manifest.json').read_bytes())
    digest.update((source/'data.yaml').read_bytes())
    for path in sorted((source/'labels').rglob('*.txt')):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    for path in sorted(p for p in (source/'images').rglob('*') if p.is_file()):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def inspect_checkpoint(path):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = checkpoint.get('ema') or checkpoint.get('model')
    names = getattr(model, 'names', {})
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    args = checkpoint.get('train_args', {})
    return dict(epoch=int(checkpoint.get('epoch', -1)), epochs=int(args.get('epochs', 0)),
                optimizer=checkpoint.get('optimizer') is not None,
                classes=list(names), data=args.get('data'))


def latest_completed_model(path):
    """Only a finished, checksummed model of the configured class family."""
    path = Path(path)
    anchor = json.loads((path.parent.parent/'manifest.json').read_text())
    options = []
    for manifest in path.parent.parent.parent.glob('candidate*/manifest.json'):
        metadata = json.loads(manifest.read_text())
        best = manifest.parent/'weights/best.pt'
        if (metadata.get('classes') == anchor.get('classes') and
            metadata.get('revision') == anchor.get('revision') and
            metadata.get('kind') == 'external_slice_candidate' and best.is_file()):
            options.append((best, metadata))
    for best, metadata in sorted(options,key=lambda item:item[0].stat().st_mtime_ns,reverse=True):
        if hashlib.sha256(best.read_bytes()).hexdigest() != metadata.get('model_sha256'):
            raise ValueError('Latest completed model checksum failed')
        return best
    return path


def choose_start(source, output, base_model, *, fresh=False, inspect=inspect_checkpoint):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = json.loads((source/'manifest.json').read_text())
    fingerprint = dataset_fingerprint(source)
    initial = dict(mode='new', model=str(base_model), dataset_fingerprint=fingerprint)
    if fresh:
        return initial
    checkpoints = sorted(output.glob('candidate*/weights/last.pt'),
                         key=lambda p:p.stat().st_mtime_ns, reverse=True)
    for checkpoint in checkpoints:
        info = inspect(checkpoint)
        if not info.get('data'):
            continue
        if info['classes'] != manifest['classes']:
            raise ValueError('Latest checkpoint classes differ; export separately or explicitly use --fresh')
        dataset_changed = Path(info['data']).resolve() != source/'data.yaml'
        state_path = checkpoint.parent.parent/'training-state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get('dataset_fingerprint') != fingerprint:
                dataset_changed = True
        resumable = not dataset_changed and info['optimizer'] and 0 <= info['epoch'] < info['epochs']-1
        completed_epochs = max(0,info['epoch']+1)
        results = checkpoint.parent.parent/'results.csv'
        if results.is_file():
            with results.open(encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                completed_epochs = int(float(next(v for k,v in rows[-1].items() if k.strip()=='epoch')))
        chosen = checkpoint
        if not resumable:
            best = checkpoint.parent/'best.pt'
            if best.is_file():
                chosen = best
            metadata = checkpoint.parent.parent/'manifest.json'
            if metadata.is_file() and chosen.name == 'best.pt':
                expected = json.loads(metadata.read_text()).get('model_sha256')
                if expected != hashlib.sha256(chosen.read_bytes()).hexdigest():
                    raise ValueError('Completed model checksum failed; refusing to reset silently')
        return dict(mode='resume' if resumable else 'finetune', model=str(chosen),
                    run=str(checkpoint.parent.parent), completed_epochs=completed_epochs,
                    target_epochs=info['epochs'], dataset_fingerprint=fingerprint,
                    dataset_changed=dataset_changed)
    return initial
