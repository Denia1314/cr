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

    def detect(self, image):
        prediction = self.model.predict(image, conf=float(self.config.get('runtime_confidence', .7)),
                                        imgsz=int(self.config.get('image_size', 640)),
                                        device=self.device, verbose=False)[0]
        candidates = []
        if prediction.boxes is not None:
            for index, confidence, box in zip(prediction.boxes.cls.cpu().tolist(),
                                             prediction.boxes.conf.cpu().tolist(),
                                             prediction.boxes.xyxyn.cpu().tolist()):
                name = self.model.names[int(index)]
                card_id = unit_card_id(name.replace('-', '_'), self.catalog)
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
        return result

    def status(self):
        return dict(source='external_slice_candidate', version=self.metadata['revision'][:12],
                    device=self.device, inference_calls=self.calls, local_quality_validated=False,
                    battle_acceptance=False, promoted=False)
