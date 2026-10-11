"""Explicit experimental YOLO adapter; unit identity and badge team evidence."""
import json
import hashlib
from pathlib import Path

from .gpu import vision_device
from .pretrained_battlefield import unit_card_id
from .slice_dataset import REVISION
from .unit_badges import associate_badges, find_level_badges


class SliceBattlefield:
    def __init__(self, root, catalog, config):
        from ultralytics import YOLO
        path = Path(config['slice_model_path'])
        if not path.is_absolute():
            path = root / path
        self.metadata = json.loads((path.parent.parent / 'manifest.json').read_text())
        if self.metadata.get('kind') != 'external_slice_candidate' or self.metadata.get('revision') != REVISION:
            raise ValueError('Invalid external slice candidate manifest')
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata.get('model_sha256'):
            raise ValueError('External slice candidate checksum differs from manifest')
        self.model = YOLO(str(path))
        names = [self.model.names[i] for i in range(len(self.model.names))]
        if names != self.metadata['classes']:
            raise ValueError('YOLO classes differ from slice manifest')
        self.catalog, self.config = catalog, config
        self.device = vision_device(config.get('device', 'auto'))
        self.calls = 0
        self.fallback = None
        self.fallback_error = ''
        if config.get('slice_fallback_enabled', True) and config.get('pretrained_enabled', True):
            try:
                from .pretrained_battlefield import PretrainedBattlefield
                self.fallback = PretrainedBattlefield(root, catalog, config)
            except Exception as exc:
                self.fallback_error = str(exc)

    def detect(self, image):
        # Upstream part2 labels cover the arena, excluding the hand panel.
        arena_model = self.metadata.get('training_dataset_kind') == 'real_battle_bootstrap'
        bottom = round(image.height * .8) if arena_model else image.height
        source = image.crop((0, 0, image.width, bottom)) if arena_model else image
        prediction = self.model.predict(source, conf=float(self.config.get('runtime_confidence', .7)),
                                        imgsz=int(self.config.get('image_size', 640)),
                                        device=self.device, verbose=False)[0]
        self.last_raw_boxes = len(prediction.boxes.cls.cpu().tolist()) if prediction.boxes is not None else 0
        candidates = []
        if prediction.boxes is not None:
            for index, confidence, box in zip(prediction.boxes.cls.cpu().tolist(),
                                             prediction.boxes.conf.cpu().tolist(),
                                             prediction.boxes.xyxyn.cpu().tolist()):
                name = self.model.names[int(index)]
                card_id = unit_card_id(name.replace('-', '_'), self.catalog)
                box = [box[0], box[1]*bottom/image.height, box[2], box[3]*bottom/image.height]
                if card_id and .20 <= (box[1]+box[3])/2 <= .80:
                    candidates.append((card_id, name, confidence, box))
        badges = associate_badges([c[3] for c in candidates], find_level_badges(image))
        result = []
        for i, (card_id, name, confidence, box) in enumerate(candidates):
            badge = badges.get(i)
            if badge is None:
                continue
            result.append(dict(card_id=card_id, unit_name=name, bbox=box, side=badge['side'],
                               confidence=min(confidence, badge['confidence']),
                               side_evidence=badge['evidence'], level=badge['level'],
                               level_badge_bbox=badge['bbox'], identity_source='external_slice_candidate',
                               deployment_confirmed=False))
        self.calls += 1
        self.last_primary_units = len(result)
        self.last_fallback_used = False
        if not result and getattr(self, 'fallback', None) is not None:
            try:
                result = self.fallback.detect(image)
                self.last_fallback_used = True
                self.fallback_error = ''
            except Exception as exc:
                self.fallback_error = str(exc)
        self.last_accepted_units = len(result)
        return result

    def status(self):
        return dict(source='external_slice_candidate', version='katacr-' + self.metadata['model_sha256'][:12],
                    source_revision=self.metadata['revision'],
                    dataset_kind=self.metadata.get('training_dataset_kind', 'synthetic_bootstrap'),
                    device=self.device, inference_calls=self.calls, local_quality_validated=False,
                    raw_boxes=getattr(self, 'last_raw_boxes', 0),
                    primary_units=getattr(self, 'last_primary_units', 0),
                    accepted_units=getattr(self, 'last_accepted_units', 0),
                    fallback_loaded=getattr(self, 'fallback', None) is not None,
                    fallback_used=getattr(self, 'last_fallback_used', False),
                    fallback_error=getattr(self, 'fallback_error', ''),
                    battle_acceptance=False, promoted=False)
