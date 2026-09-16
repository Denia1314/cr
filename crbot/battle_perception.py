from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from threading import RLock

import numpy as np
from PIL import Image

from .cards import CardCatalog, CardDefinition
from .gpu import HandDescriptorMatcher

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only on incomplete installations
    cv2 = None  # type: ignore[assignment]


@dataclass(frozen=True)
class HandCardMatch:
    slot_index: int
    card_id: str | None
    confidence: float
    good_matches: int
    second_good_matches: int
    keypoints: int
    empty: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LaneThreat:
    lane: str
    score: float
    unit_count: int
    proximity: float
    threat: str
    centers: tuple[tuple[float, float], ...]
    enemy_cards: tuple[str, ...] = ()
    approach_rate: float = 0.0
    unit_layers: tuple[str, ...] = ()
    layer_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["centers"] = [list(center) for center in self.centers]
        value["enemy_cards"] = list(self.enemy_cards)
        value["unit_layers"] = list(self.unit_layers)
        return value


class UniversalHandRecognizer:
    """Match hand art against a catalog without assuming the player's deck."""

    def __init__(self, catalog: CardCatalog, vision_config: dict[str, Any]):
        self.catalog = catalog
        self._recognition_lock = RLock()
        self.vision = vision_config
        self.available = cv2 is not None
        self.templates: dict[str, tuple[CardDefinition, Any]] = {}
        self.gpu_matcher = None
        if not self.available:
            return
        self.orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.15,
            nlevels=8,
            edgeThreshold=10,
            patchSize=21,
            fastThreshold=8,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        for card in catalog.cards:
            for variant_index, path in enumerate(catalog.local_icon_paths(card)):
                if not path.is_file():
                    continue
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = cv2.resize(image, (180, 220), interpolation=cv2.INTER_AREA)
                _keypoints, descriptors = self.orb.detectAndCompute(image, None)
                if descriptors is not None and len(descriptors) >= 20:
                    template_id = f"{card.card_id}:{variant_index}"
                    self.templates[template_id] = (card, descriptors)

        if self.vision.get('card_match_prewarm',False) and self.templates:
            self.gpu_matcher=HandDescriptorMatcher([ref for _,ref in self.templates.values()],self.vision.get('card_match_device'))
            self.gpu_matcher.warming_up=True
            self.gpu_matcher.counts(np.zeros((2,32),dtype=np.uint8),float(self.vision.get('card_match_ratio',.78)))
            self.gpu_matcher.warmup_calls=self.gpu_matcher.calls
            self.gpu_matcher.calls=0
            self.gpu_matcher.slots=0
            self.gpu_matcher.batches=0
            self.gpu_matcher.warming_up=False

    def compute_status(self) -> dict[str, Any]:
        return {
            "device": self.gpu_matcher.device if self.gpu_matcher is not None else "not_executed",
            "gpu_match_calls": self.gpu_matcher.calls if self.gpu_matcher is not None else 0,
            "unchanged_slot_cache_hits": getattr(self,"slot_cache_hits",0),
            "gpu_matched_slots": getattr(self.gpu_matcher, "slots", 0),
            "gpu_template_batches": getattr(self.gpu_matcher, "batches", 0),
            "gpu_last_match_ms": round(getattr(self.gpu_matcher, "last_ms", 0.0), 3),
            "gpu_last_batch_templates": getattr(self.gpu_matcher, "last_batch_templates", 0),
            "warmup_calls": getattr(self.gpu_matcher,"warmup_calls",0),
        }

    def _slot_image(self, image: Image.Image, center: list[float]) -> np.ndarray:
        half_width = float(self.vision.get("card_roi_half_width", 0.09))
        top = float(self.vision.get("card_roi_top", 0.825))
        bottom = float(self.vision.get("card_roi_bottom", 0.955))
        x = float(center[0])
        left_px = max(0, round((x - half_width) * image.width))
        right_px = min(image.width, round((x + half_width) * image.width))
        top_px = max(0, round(top * image.height))
        bottom_px = min(image.height, round(bottom * image.height))
        crop = np.asarray(image.crop((left_px,top_px,right_px,bottom_px)).convert("L"))
        return cv2.resize(crop, (180, 220), interpolation=cv2.INTER_AREA)

    def recognize(self, image: Image.Image) -> list[HandCardMatch]:
        # Background preprocessing and action confirmation share ORB/CUDA state.
        with getattr(self, "_recognition_lock", RLock()):
            return self._recognize(image)

    def _recognize(self, image: Image.Image) -> list[HandCardMatch]:
        centers = self.vision["card_slot_centers"]
        if not self.available or not self.templates:
            return [
                HandCardMatch(index, None, 0.0, 0, 0, 0, False)
                for index, _center in enumerate(centers)
            ]

        ratio = float(self.vision.get("card_match_ratio", 0.78))
        minimum_good = int(self.vision.get("card_match_min_good", 35))
        minimum_margin = float(self.vision.get("card_match_min_margin", 1.45))
        empty_keypoints = int(self.vision.get("card_empty_max_keypoints", 150))
        matches = [None] * len(centers)
        pending = []
        cache=getattr(self,"_slot_cache",{})
        keys=[]
        for slot_index, center in enumerate(centers):
            observed = self._slot_image(image, center)
            key=(observed.tobytes(),ratio,minimum_good,minimum_margin,empty_keypoints,id(self.templates),len(self.templates))
            keys.append(key)
            if slot_index in cache and cache[slot_index][0] == key:
                matches[slot_index] = cache[slot_index][1]
                self.slot_cache_hits=getattr(self,"slot_cache_hits",0)+1
                continue
            keypoints, descriptors = self.orb.detectAndCompute(observed, None)
            keypoint_count = len(keypoints)
            if descriptors is None or keypoint_count <= empty_keypoints:
                matches[slot_index] = HandCardMatch(slot_index, None, 1.0, 0, 0, keypoint_count, True)
                continue

            pending.append((slot_index, keypoint_count, descriptors))

        if pending:
            if self.gpu_matcher is None:
                self.gpu_matcher = HandDescriptorMatcher(
                    [reference for _card, reference in self.templates.values()],
                    self.vision.get("card_match_device"),
                )
            batch_counts = self.gpu_matcher.counts_many([item[2] for item in pending], ratio)
        for pending_index, (slot_index, keypoint_count, descriptors) in enumerate(pending):
            gpu_counts = None if batch_counts is None else batch_counts[pending_index]
            best_by_card: dict[str, int] = {}
            for index, (_template_id, (card, reference)) in enumerate(self.templates.items()):
                if gpu_counts is None:
                    pairs = self.matcher.knnMatch(descriptors, reference, k=2)
                    good = sum(1 for first, second in pairs if first.distance < ratio * second.distance)
                else:
                    good = gpu_counts[index]
                best_by_card[card.card_id] = max(best_by_card.get(card.card_id, 0), good)
            ranked = [(good, card_id) for card_id, good in best_by_card.items()]
            ranked.sort(reverse=True)
            best_good, best_id = ranked[0]
            second_good = ranked[1][0] if len(ranked) > 1 else 0
            margin = best_good / max(1, second_good)
            accepted = best_good >= minimum_good and margin >= minimum_margin
            strength = min(1.0, best_good / max(1.0, minimum_good * 3.0))
            separation = min(1.0, max(0.0, (margin - 1.0) / 2.0))
            confidence = round(0.65 * strength + 0.35 * separation, 4)
            matches[slot_index] = HandCardMatch(
                slot_index=slot_index,
                card_id=best_id if accepted else None,
                confidence=confidence if accepted else min(0.49, confidence),
                good_matches=best_good,
                second_good_matches=second_good,
                keypoints=keypoint_count,
                empty=False,
            )
        self._slot_cache={i:(key,matches[i]) for i,key in enumerate(keys)}
        return matches


def _level_badge_candidates(image: Image.Image) -> list[tuple[float, float, int]]:
    """Find red enemy level badges across the playable arena."""
    if cv2 is None:
        return []

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    # Evaluate color only inside the same playable bounds as the original
    # full-frame mask. Retain full-size zero padding for identical morphology.
    top, bottom = round(.20 * height), round(.76 * height)
    left, right = round(.13 * width), round(.87 * width)
    arena = rgb[top:bottom, left:right].astype(np.int16)
    red, green, blue = arena[..., 0], arena[..., 1], arena[..., 2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = (
        (red > 150)
        & (green < 110)
        & (red > green * 1.5)
        & (red > blue * 1.08)
        & ((red - green) > 60)
    ).astype(np.uint8) * 255
    mask[round(0.68 * height) :, round(0.75 * width) :] = 0
    # Princess-tower skins contain red plates and white highlights that look
    # like a level badge.  Troops leaving these areas become visible well
    # before the bridge, so exclude the static tower footprints themselves.
    tower_top, tower_bottom = round(0.17 * height), round(0.31 * height)
    mask[tower_top:tower_bottom, round(0.13 * width) : round(0.39 * width)] = 0
    mask[tower_top:tower_bottom, round(0.61 * width) : round(0.87 * width)] = 0
    mask[round(0.10 * height) : round(0.23 * height), round(0.39 * width) : round(0.61 * width)] = 0
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
    )
    _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    detected: list[tuple[float, float, int]] = []
    for x, y, component_width, component_height, area in stats[1:]:
        if not (
            18 <= component_width <= 110
            and 10 <= component_height <= 45
            and area >= 80
        ):
            continue
        x1, y1 = max(0, x - 5), max(0, y - 5)
        x2 = min(width, x + component_width + 5)
        y2 = min(height, y + component_height + 5)
        patch = rgb[y1:y2, x1:x2]
        white_ratio = float(np.mean(np.all(patch > 180, axis=2)))
        if white_ratio < 0.04:
            continue
        normalized_x = float((x + component_width / 2.0) / width)
        normalized_y = float((y + component_height / 2.0) / height)
        if normalized_y > 0.68 and 0.35 < normalized_x < 0.65:
            continue
        detected.append((normalized_x, normalized_y, int(component_width)))

    return detected


def detect_lane_threats(
    image: Image.Image,
    previous: Image.Image | None = None,
    ignore_points: tuple[tuple[float, float], ...] = (),
    frame_dt_s: float | None = None,
    *, candidates=None,
) -> dict[str, LaneThreat]:
    """Detect enemy lanes early and estimate whether units are approaching.

    Opponent level badges are visible before troops reach the bridge.  Matching
    their positions with the preceding frame gives a small forward-motion
    signal which raises the priority of an approaching push.  When
    ``frame_dt_s`` is supplied, ``approach_rate`` is normalized to screen
    fraction per second rather than depending on the screenshot cadence.
    """
    if cv2 is None:
        empty = LaneThreat("left", 0.0, 0, 0.0, "none", ())
        return {
            "left": empty,
            "right": LaneThreat("right", 0.0, 0, 0.0, "none", ()),
        }

    detected = list(candidates) if candidates is not None else _level_badge_candidates(image)
    # A frame-rate independent speed is more useful than the raw pixel delta.
    # ``None`` preserves the historical one-frame interpretation for callers
    # that do not have timestamps.
    dt_s = max(0.05, float(frame_dt_s)) if frame_dt_s is not None else 1.0
    if ignore_points:
        detected = [
            value
            for value in detected
            if not any(
                abs(value[0] - point[0]) <= 0.075
                and abs(value[1] - point[1]) <= 0.065
                for point in ignore_points
            )
        ]
    previous_detected = (
        _level_badge_candidates(previous)
        if previous is not None and previous.size == image.size
        else []
    )
    if ignore_points:
        previous_detected = [
            value
            for value in previous_detected
            if not any(
                abs(value[0] - point[0]) <= 0.075
                and abs(value[1] - point[1]) <= 0.065
                for point in ignore_points
            )
        ]
    results: dict[str, LaneThreat] = {}
    for lane in ("left", "right"):
        lane_units = [
            value
            for value in detected
            if (value[0] < 0.5 if lane == "left" else value[0] >= 0.5)
        ]
        proximity = max((value[1] for value in lane_units), default=0.0)
        unit_count = len(lane_units)
        previous_lane_units = [
            value
            for value in previous_detected
            if (value[0] < 0.5 if lane == "left" else value[0] >= 0.5)
        ]
        approach_rates: list[float] = []
        # Greedy nearest-neighbour matching can assign one delayed badge to
        # several current units.  Build all admissible pairs and consume each
        # previous observation at most once.
        pair_candidates = sorted(
            (
                abs(previous_x - x) + abs(previous_y - y),
                current_index,
                previous_index,
                y - previous_y,
            )
            for current_index, (x, y, _width) in enumerate(lane_units)
            for previous_index, (previous_x, previous_y, _previous_width) in enumerate(previous_lane_units)
            if abs(previous_x - x) <= 0.14
        )
        matched_current: set[int] = set()
        matched_previous: set[int] = set()
        for _distance, current_index, previous_index, delta_y in pair_candidates:
            if current_index in matched_current or previous_index in matched_previous:
                continue
            matched_current.add(current_index)
            matched_previous.add(previous_index)
            approach_rates.append(max(0.0, delta_y) / dt_s)
        approach_rate = max(approach_rates, default=0.0)
        score = min(
            1.0,
            unit_count * 0.20
            + max(0.0, (proximity - 0.20) / 0.52) * 0.62
            + min(0.18, approach_rate * 6.0),
        )
        if unit_count == 0:
            threat = "none"
        elif unit_count >= 3:
            threat = "swarm"
        elif proximity >= 0.62:
            threat = "heavy"
        else:
            threat = "single"
        results[lane] = LaneThreat(
            lane=lane,
            score=round(score, 4),
            unit_count=unit_count,
            proximity=round(proximity, 4),
            threat=threat,
            centers=tuple((round(x, 4), round(y, 4)) for x, y, _width in lane_units),
            approach_rate=round(approach_rate, 4),
        )
    return results
