"""Explicit experimental YOLO adapter; unit identity and badge team evidence."""
import json
import hashlib
import time
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
        if config.get('slice_auto_latest_completed', False):
            from .yolo_continuation import latest_completed_model
            path = latest_completed_model(path)
        self.model_path = path
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
        started = time.perf_counter()
        # Upstream part2 labels cover the arena, excluding the hand panel.
        arena_model = self.metadata.get('training_dataset_kind') == 'real_battle_bootstrap'
        bottom = round(image.height * .8) if arena_model else image.height
        source = image.crop((0, 0, image.width, bottom)) if arena_model else image
        threshold=float(self.config.get('runtime_confidence', .7))
        candidate_threshold=max(.1,min(threshold,float(self.config.get('unconfirmed_unit_confidence',.25))))
        prediction = self.model.predict(source, conf=candidate_threshold,
                                        imgsz=int(self.config.get('image_size', 640)),
                                        device=self.device, verbose=False)[0]
        primary_finished = time.perf_counter()
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
        level_badges = find_level_badges(image)
        badges_finished = time.perf_counter()
        strong=[i for i,c in enumerate(candidates) if c[2]>=threshold]
        badges={strong[i]:badge for i,badge in associate_badges([candidates[i][3] for i in strong],level_badges).items()}
        result = []
        self.unconfirmed_units=[]
        self.last_rejections={'low_confidence':0,'team_badge_missing':0}
        for i, (card_id, name, confidence, box) in enumerate(candidates):
            badge = badges.get(i)
            reason='low_confidence' if confidence<threshold else 'team_badge_missing' if badge is None else ''
            if reason:
                if confidence<threshold:self.last_rejections['low_confidence']+=1
                if confidence<threshold:badge=associate_badges([box],level_badges).get(0)
                if badge is None:self.last_rejections['team_badge_missing']+=1
                if len(self.unconfirmed_units)<64:
                    self.unconfirmed_units.append(dict(bbox=box,confidence=float(confidence),
                        candidate_card_id=card_id,candidate_unit_name=name,side=badge['side'] if badge else None,
                        observation_status='unconfirmed',reason=reason,team_badge_missing=badge is None,
                        identity_source='external_slice_candidate'))
                continue
            result.append(dict(card_id=card_id, unit_name=name, bbox=box, side=badge['side'],
                               confidence=min(confidence, badge['confidence']),
                               side_evidence=badge['evidence'], level=badge['level'],
                               level_badge_bbox=badge['bbox'], identity_source='external_slice_candidate',
                               deployment_confirmed=False))
        self.calls += 1
        self.last_primary_units = len(result)
        self.last_fallback_used = False
        # Both models require the same current-frame team badge evidence. With
        # none present the fallback cannot accept a unit; still run the primary
        # YOLO every frame, and reuse badge extraction when fallback is useful.
        if not result and level_badges and getattr(self, 'fallback', None) is not None:
            try:
                result = self.fallback.detect(image, level_badges=level_badges)
                self.last_fallback_used = True
                self.fallback_error = ''
            except Exception as exc:
                self.fallback_error = str(exc)
        self.last_accepted_units = len(result)
        from .unconfirmed_units import box_overlap
        self.unconfirmed_units=[c for c in self.unconfirmed_units if not any(len(d.get('bbox',()))==4 and box_overlap(c['bbox'],d['bbox'])>.3 for d in result)]
        self.last_primary_ms = (primary_finished-started)*1000
        self.last_badges_ms = (badges_finished-primary_finished)*1000
        self.last_fallback_ms = getattr(self.fallback, 'last_ms', 0.) if self.last_fallback_used else 0.
        return result

    def status(self):
        return dict(source='external_slice_candidate', version='katacr-' + self.metadata['model_sha256'][:12],
                    model_path=str(self.model_path),
                    source_revision=self.metadata['revision'],
                    dataset_kind=self.metadata.get('training_dataset_kind', 'synthetic_bootstrap'),
                    device=self.device, inference_calls=self.calls, local_quality_validated=False,
                    raw_boxes=getattr(self, 'last_raw_boxes', 0),
                    primary_units=getattr(self, 'last_primary_units', 0),
                    accepted_units=getattr(self, 'last_accepted_units', 0),
                    unconfirmed_units=len(getattr(self,'unconfirmed_units',[])),
                    rejected=getattr(self,'last_rejections',{}),
                    fallback_loaded=getattr(self, 'fallback', None) is not None,
                    fallback_used=getattr(self, 'last_fallback_used', False),
                    fallback_error=getattr(self, 'fallback_error', ''),
                    primary_ms=round(getattr(self, 'last_primary_ms', 0.),3),
                    badges_ms=round(getattr(self, 'last_badges_ms', 0.),3),
                    fallback_ms=round(getattr(self, 'last_fallback_ms', 0.),3),
                    battle_acceptance=False, promoted=False)
