from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import resolve_project_path


def normalized_box(roi: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x1 = max(0, min(width - 1, round(float(roi[0]) * width)))
    y1 = max(0, min(height - 1, round(float(roi[1]) * height)))
    x2 = max(x1 + 1, min(width, round(float(roi[2]) * width)))
    y2 = max(y1 + 1, min(height, round(float(roi[3]) * height)))
    return x1, y1, x2, y2


def crop_normalized(image: Image.Image, roi: list[float]) -> Image.Image:
    return image.crop(normalized_box(roi, image.size))


def _small_rgb(image: Image.Image, size: tuple[int, int] = (96, 64)) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def patch_similarity(first: Image.Image, second: Image.Image) -> float:
    a = _small_rgb(first)
    b = _small_rgb(second)
    color_error = float(np.mean(np.abs(a - b)) / 255.0)
    gray_a = np.mean(a, axis=2)
    gray_b = np.mean(b, axis=2)
    edge_a = np.concatenate((np.diff(gray_a, axis=0).ravel(), np.diff(gray_a, axis=1).ravel()))
    edge_b = np.concatenate((np.diff(gray_b, axis=0).ravel(), np.diff(gray_b, axis=1).ravel()))
    edge_error = float(np.mean(np.abs(edge_a - edge_b)) / 255.0)
    score = 1.0 - (0.78 * color_error + 0.22 * min(1.0, edge_error))
    return max(0.0, min(1.0, score))


@dataclass(frozen=True)
class ProbeMatch:
    name: str
    kind: str
    priority: int
    score: float
    click: list[float] | None
    requires_offline_gate: bool
    requires_battle_seen: bool


@dataclass(frozen=True)
class BattleResult:
    outcome: str
    player_crowns: int | None
    opponent_crowns: int | None
    confidence: float
    player_slot_scores: tuple[float, float, float]
    opponent_slot_scores: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "player_crowns": self.player_crowns,
            "opponent_crowns": self.opponent_crowns,
            "confidence": round(float(self.confidence), 6),
            "player_slot_scores": [round(value, 6) for value in self.player_slot_scores],
            "opponent_slot_scores": [
                round(value, 6) for value in self.opponent_slot_scores
            ],
        }


class WorkflowRecognizer:
    def __init__(self, config: dict[str, Any], config_path: Path):
        self.config = config
        self.config_path = config_path
        self.threshold = float(config["vision"].get("min_probe_score", 0.78))
        self.templates: dict[str, Image.Image] = {}
        for target in config.get("workflow", []):
            roi = target.get("roi")
            template_path = resolve_project_path(config_path, target["template"])
            if roi and template_path.is_file():
                with Image.open(template_path) as image:
                    self.templates[target["name"]] = image.convert("RGB").copy()

    def calibrated_names(self) -> list[str]:
        return sorted(self.templates)

    def match_all(self, image: Image.Image) -> list[ProbeMatch]:
        matches: list[ProbeMatch] = []
        for target in self.config.get("workflow", []):
            name = target["name"]
            template = self.templates.get(name)
            roi = target.get("roi")
            if template is None or not roi:
                continue
            patch = crop_normalized(image, roi)
            score = patch_similarity(patch, template)
            if score >= float(target.get("threshold", self.threshold)):
                matches.append(
                    ProbeMatch(
                        name=name,
                        kind=target["kind"],
                        priority=int(target.get("priority", 0)),
                        score=score,
                        click=target.get("click"),
                        requires_offline_gate=bool(target.get("requires_offline_gate", False)),
                        requires_battle_seen=bool(target.get("requires_battle_seen", False)),
                    )
                )
        return sorted(matches, key=lambda item: (item.priority, item.score), reverse=True)


def estimate_elixir(image: Image.Image, roi: list[float]) -> tuple[float | None, float]:
    patch = np.asarray(crop_normalized(image, roi).convert("RGB"), dtype=np.float32)
    if patch.size == 0:
        return None, 0.0
    red, green, blue = patch[..., 0], patch[..., 1], patch[..., 2]
    purple = (
        (red > 80)
        & (blue > 95)
        & (blue > green * 1.08)
        & (red > green * 1.02)
    )
    column_coverage = np.mean(purple, axis=0)
    filled_columns = column_coverage > 0.14
    coverage = float(np.mean(filled_columns))
    confidence = float(min(1.0, np.mean(purple) * 7.0))
    if confidence < 0.08:
        return None, confidence
    return max(0.0, min(10.0, coverage * 10.0)), confidence


def battle_ui_score(image: Image.Image, roi: list[float]) -> float:
    """Detect the purple elixir strip/droplet at the bottom of a battle screen.

    This intentionally looks only at the stable Android-rendered game UI. It
    does not depend on arena art, player names, card art, or screen resolution.
    """
    patch = np.asarray(crop_normalized(image, roi).convert("RGB"), dtype=np.float32)
    if patch.size == 0:
        return 0.0
    red, green, blue = patch[..., 0], patch[..., 1], patch[..., 2]
    purple = (
        (red > 80)
        & (blue > 95)
        & (blue > green * 1.08)
        & (red > green * 1.02)
    )
    purple_columns = float(np.mean(np.any(purple, axis=0)))
    purple_pixels = float(np.mean(purple))
    column_component = min(1.0, purple_columns / 0.12)
    pixel_component = min(1.0, purple_pixels / 0.018)
    return max(0.0, min(1.0, 0.55 * column_component + 0.45 * pixel_component))


def _find_direct_open_chest_screen(
    rgb: np.ndarray, hsv: np.ndarray
) -> tuple[list[float] | None, float]:
    """Recognize the four-star chest screen whose footer says ``点击打开``.

    This variant has a multicolor gradient and no question-mark choices, so it
    cannot share the single-color/background checks used by the selection
    screen.  All three stable structures are required: four star outlines in
    their diamond arrangement, one large centered purple chest, and a compact
    line of white footer text on an otherwise uncluttered screen.
    """
    height, width = hsv.shape[:2]

    # Stars stay sharply outlined while the animated gradient and glow remain
    # smooth. Canny contours are substantially more stable here than an HSV
    # yellow mask, because the background itself passes through yellow.
    star_x1, star_y1, star_x2, star_y2 = normalized_box(
        [0.20, 0.12, 0.80, 0.42], (width, height)
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    star_edges = cv2.Canny(
        gray[star_y1:star_y2, star_x1:star_x2], 60, 140
    )
    contours, _hierarchy = cv2.findContours(
        star_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    star_centers: list[tuple[float, float]] = []
    for contour in contours:
        x, y, component_width, component_height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        vertices = len(cv2.approxPolyDP(contour, 0.03 * perimeter, True))
        center = (
            (star_x1 + x + component_width / 2.0) / width,
            (star_y1 + y + component_height / 2.0) / height,
        )
        if not (
            0.28 <= center[0] <= 0.72
            and 0.16 <= center[1] <= 0.39
            and 0.045 <= component_width / width <= 0.13
            and 0.025 <= component_height / height <= 0.075
            and 0.00055 <= area / float(width * height) <= 0.008
            and 8 <= vertices <= 14
        ):
            continue
        if all(
            abs(center[0] - old[0]) > 0.025
            or abs(center[1] - old[1]) > 0.018
            for old in star_centers
        ):
            star_centers.append(center)
    if len(star_centers) != 4:
        return None, 0.0

    center_stars = [point for point in star_centers if abs(point[0] - 0.5) <= 0.08]
    left_stars = [point for point in star_centers if point[0] < 0.46]
    right_stars = [point for point in star_centers if point[0] > 0.54]
    star_ys = [point[1] for point in star_centers]
    if (
        len(center_stars) != 2
        or len(left_stars) != 1
        or len(right_stars) != 1
        or max(star_ys) - min(star_ys) < 0.045
        or abs(left_stars[0][1] - right_stars[0][1]) > 0.035
    ):
        return None, 0.0

    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    purple = (
        (hue >= 125)
        & (hue <= 175)
        & (saturation >= 45)
        & (value >= 45)
    )
    chest_x1, chest_y1, chest_x2, chest_y2 = normalized_box(
        [0.15, 0.35, 0.85, 0.72], (width, height)
    )
    chest_mask = (
        purple[chest_y1:chest_y2, chest_x1:chest_x2].astype(np.uint8) * 255
    )
    close_size = max(5, round(min(width, height) * 0.012)) | 1
    chest_mask = cv2.morphologyEx(
        chest_mask,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), dtype=np.uint8),
    )
    _count, _labels, chest_stats, _centroids = cv2.connectedComponentsWithStats(
        chest_mask
    )
    if len(chest_stats) <= 1:
        return None, 0.0
    chest = max(chest_stats[1:], key=lambda item: int(item[4]))
    x, y, component_width, component_height, area = chest
    chest_center = (
        (chest_x1 + x + component_width / 2.0) / width,
        (chest_y1 + y + component_height / 2.0) / height,
    )
    if not (
        0.38 <= chest_center[0] <= 0.62
        and 0.43 <= chest_center[1] <= 0.62
        and component_width / width >= 0.28
        and component_height / height >= 0.10
        and area / float(width * height) >= 0.015
    ):
        return None, 0.0

    # Require a centered row of multiple white glyph components where
    # “点击打开” is rendered. We do not accept a generic large white panel.
    text_x1, text_y1, text_x2, text_y2 = normalized_box(
        [0.30, 0.86, 0.70, 0.97], (width, height)
    )
    text_hsv = hsv[text_y1:text_y2, text_x1:text_x2]
    text_mask = (
        (text_hsv[..., 1] <= 90) & (text_hsv[..., 2] >= 185)
    ).astype(np.uint8)
    _count, _labels, text_stats, _centroids = cv2.connectedComponentsWithStats(
        text_mask
    )
    glyph_boxes: list[tuple[float, float, float, float]] = []
    for x, y, component_width, component_height, area in text_stats[1:]:
        component_center_x = (text_x1 + x + component_width / 2.0) / width
        component_center_y = (text_y1 + y + component_height / 2.0) / height
        if (
            area >= max(5, round(width * height * 0.00001))
            and 0.003 <= component_width / width <= 0.08
            and 0.004 <= component_height / height <= 0.05
            and 0.88 <= component_center_y <= 0.95
        ):
            glyph_boxes.append(
                (
                    component_center_x,
                    component_center_y,
                    component_width / width,
                    component_height / height,
                )
            )
    if len(glyph_boxes) < 4:
        return None, 0.0
    glyph_xs = [box[0] for box in glyph_boxes]
    if not (
        0.08 <= max(glyph_xs) - min(glyph_xs) <= 0.35
        and 0.40 <= sum(glyph_xs) / len(glyph_xs) <= 0.60
    ):
        return None, 0.0

    # The direct-open page is intentionally almost empty outside the three
    # structures above. This prevents battle/result pages with text and purple
    # art from satisfying the detector by accident.
    edges = cv2.Canny(gray, 60, 140)

    def edge_ratio(roi: list[float]) -> float:
        x1, y1, x2, y2 = normalized_box(roi, (width, height))
        return float(np.mean(edges[y1:y2, x1:x2] > 0))

    if (
        edge_ratio([0.02, 0.10, 0.14, 0.82]) > 0.02
        or edge_ratio([0.86, 0.10, 0.98, 0.82]) > 0.02
    ):
        return None, 0.0

    click = [float(round(chest_center[0], 6)), float(round(chest_center[1], 6))]
    confidence = min(
        1.0,
        0.82
        + min(0.08, area / float(width * height))
        + min(0.05, (max(star_ys) - min(star_ys)) * 0.8)
        + min(0.05, len(glyph_boxes) * 0.006),
    )
    return click, float(confidence)


def find_chest_open_screen(image: Image.Image) -> tuple[list[float] | None, float]:
    """Recognize the uncluttered star/chest/question-mark screen layout.

    This is deliberately a structural detector rather than a pixel-perfect
    screenshot match.  Size, color shade, glow and animation may vary, but the
    frame must either be a clean blue/purple/orange selection field with stars,
    chest and question-mark choices, or the separate four-star multicolor page
    with a centered chest and ``点击打开`` footer. Lobby reward slots are never
    inspected.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.size == 0:
        return None, 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    height, width = hsv.shape[:2]

    direct_point, direct_confidence = _find_direct_open_chest_screen(rgb, hsv)
    if direct_point is not None:
        return direct_point, direct_confidence

    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (
        (hue >= 88)
        & (hue <= 135)
        & (saturation >= 45)
        & (value >= 35)
    )
    purple_background = (
        (hue >= 125)
        & (hue <= 175)
        & (saturation >= 45)
        & (value >= 35)
    )
    orange_background = (
        (hue >= 2)
        & (hue <= 32)
        & (saturation >= 45)
        & (value >= 55)
    )
    warm = (
        ((hue <= 45) | (hue >= 175))
        & (saturation >= 45)
        & (value >= 70)
    )
    star_highlight = (
        (hue >= 10)
        & (hue <= 50)
        & (saturation >= 25)
        & (saturation <= 200)
        & (value >= 210)
    )

    def ratio(mask: np.ndarray, roi: list[float]) -> float:
        x1, y1, x2, y2 = normalized_box(roi, (width, height))
        return float(np.mean(mask[y1:y2, x1:x2]))

    # The screen has no regular game UI or reward-list clutter. Choose the
    # dominant supported theme first, then require both outer columns to be
    # almost entirely that same background family.
    theme_candidates: list[tuple[np.ndarray, float, float]] = []
    for candidate in (blue, purple_background, orange_background):
        left = ratio(candidate, [0.02, 0.10, 0.18, 0.82])
        right = ratio(candidate, [0.82, 0.10, 0.98, 0.82])
        theme_candidates.append((candidate, left, right))
    background, background_left, background_right = max(
        theme_candidates,
        key=lambda item: item[1] + item[2],
    )
    background_ratio = ratio(background, [0.02, 0.08, 0.98, 0.98])
    if (
        background_ratio < 0.60
        or background_left < 0.65
        or background_right < 0.65
    ):
        return None, 0.0

    colorful = (saturation >= 40) & (value >= 45) & ~background
    chest_roi = [0.18, 0.34, 0.82, 0.72]
    chest_color_ratio = ratio(colorful, chest_roi)
    chest_warm_ratio = ratio(warm, chest_roi)
    if chest_color_ratio < 0.10 or chest_warm_ratio < 0.045:
        return None, 0.0

    # Require a centered star-sized yellow component above the chest.
    star_x1, star_y1, star_x2, star_y2 = normalized_box(
        [0.25, 0.12, 0.75, 0.42], (width, height)
    )
    star_mask = star_highlight[star_y1:star_y2, star_x1:star_x2].astype(
        np.uint8
    )
    _count, _labels, star_stats, _centroids = cv2.connectedComponentsWithStats(
        star_mask
    )
    star_centers: list[tuple[float, float]] = []
    for x, y, component_width, component_height, area in star_stats[1:]:
        center_x = (star_x1 + x + component_width / 2.0) / width
        center_y = (star_y1 + y + component_height / 2.0) / height
        area_ratio = area / float(width * height)
        if (
            0.30 <= center_x <= 0.70
            and 0.16 <= center_y <= 0.42
            and 0.00008 <= area_ratio <= 0.018
            and 0.018 <= component_width / width <= 0.22
            and 0.01 <= component_height / height <= 0.15
        ):
            star_centers.append((center_x, center_y))
    if not 1 <= len(star_centers) <= 3:
        return None, 0.0

    # Detect the circular choice outlines geometrically. Color segmentation is
    # unreliable on the orange theme because its floor shares the coin hue.
    coin_x1, coin_y1, coin_x2, coin_y2 = normalized_box(
        [0.08, 0.78, 0.92, 0.995], (width, height)
    )
    coin_gray = cv2.cvtColor(
        rgb[coin_y1:coin_y2, coin_x1:coin_x2], cv2.COLOR_RGB2GRAY
    )
    coin_gray = cv2.GaussianBlur(coin_gray, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        coin_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, round(width * 0.075)),
        param1=100,
        param2=30,
        minRadius=max(5, round(width * 0.025)),
        maxRadius=max(8, round(width * 0.085)),
    )
    coin_centers: list[tuple[float, float]] = []
    if circles is not None:
        for x, y, _radius in circles[0]:
            center_x = (coin_x1 + float(x)) / width
            center_y = (coin_y1 + float(y)) / height
            if 0.18 <= center_x <= 0.82 and 0.84 <= center_y <= 0.98:
                coin_centers.append((center_x, center_y))
    coin_centers.sort()
    if not 3 <= len(coin_centers) <= 5:
        return None, 0.0
    coin_xs = [center[0] for center in coin_centers]
    coin_ys = [center[1] for center in coin_centers]
    gaps = np.diff(coin_xs)
    if (
        max(coin_ys) - min(coin_ys) > 0.06
        or coin_xs[-1] - coin_xs[0] < 0.22
        or coin_xs[-1] - coin_xs[0] > 0.65
        or np.any(gaps < 0.06)
        or np.any(gaps > 0.25)
    ):
        return None, 0.0

    # Click the geometric center of the connected chest body, not the star or
    # a fixed lobby coordinate.
    chest_x1, chest_y1, chest_x2, chest_y2 = normalized_box(
        chest_roi, (width, height)
    )
    chest_mask = (
        colorful[chest_y1:chest_y2, chest_x1:chest_x2].astype(np.uint8) * 255
    )
    chest_close_size = max(5, round(min(width, height) * 0.018)) | 1
    chest_mask = cv2.morphologyEx(
        chest_mask,
        cv2.MORPH_CLOSE,
        np.ones((chest_close_size, chest_close_size), dtype=np.uint8),
    )
    _count, _labels, chest_stats, _centroids = cv2.connectedComponentsWithStats(
        chest_mask
    )
    if len(chest_stats) <= 1:
        return None, 0.0
    chest_component = max(chest_stats[1:], key=lambda item: int(item[4]))
    x, y, component_width, component_height, area = chest_component
    if (
        area / float(width * height) < 0.008
        or component_width / width < 0.18
        or component_height / height < 0.065
    ):
        return None, 0.0
    click = [
        float(round((chest_x1 + x + component_width / 2.0) / width, 6)),
        float(round((chest_y1 + y + component_height / 2.0) / height, 6)),
    ]
    confidence = min(
        1.0,
        0.65
        + min(0.12, max(0.0, background_ratio - 0.60) * 0.5)
        + min(0.08, max(0.0, chest_color_ratio - 0.10) * 0.5)
        + min(0.08, max(0.0, chest_warm_ratio - 0.045) * 0.5)
        + 0.07,
    )
    return click, confidence


_CONFIRM_TEXT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "templates" / "result_continue.png"
)
_confirm_text_template: np.ndarray | None = None


def _load_confirm_text_template() -> np.ndarray | None:
    global _confirm_text_template
    if _confirm_text_template is not None:
        return _confirm_text_template
    if not _CONFIRM_TEXT_TEMPLATE_PATH.is_file():
        return None
    with Image.open(_CONFIRM_TEXT_TEMPLATE_PATH) as template_image:
        _confirm_text_template = np.asarray(
            template_image.convert("L"), dtype=np.uint8
        ).copy()
    return _confirm_text_template


def find_result_confirm_button(image: Image.Image) -> tuple[list[float] | None, float]:
    """Locate the literal ``确定`` text, never a blue button by color alone."""
    template = _load_confirm_text_template()
    if template is None or template.size == 0:
        return None, 0.0

    # Every known result/reward confirmation is in the lower part of the game
    # view.  Template matching the two glyphs prevents lobby navigation buttons
    # and blue bottom bars from being treated as confirmation controls.
    roi = [0.12, 0.70, 0.88, 1.0]
    x1, y1, x2, y2 = normalized_box(roi, image.size)
    search = np.asarray(image.crop((x1, y1, x2, y2)).convert("L"), dtype=np.uint8)
    base_scale = image.width / 1080.0
    best_score = -1.0
    best_location = (0, 0)
    best_size = (0, 0)
    for multiplier in np.linspace(0.78, 1.22, 12):
        scale = base_scale * float(multiplier)
        width = max(12, round(template.shape[1] * scale))
        height = max(8, round(template.shape[0] * scale))
        if width > search.shape[1] or height > search.shape[0]:
            continue
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        scaled = cv2.resize(template, (width, height), interpolation=interpolation)
        scores = cv2.matchTemplate(search, scaled, cv2.TM_CCOEFF_NORMED)
        _minimum, maximum, _min_location, location = cv2.minMaxLoc(scores)
        if maximum > best_score:
            best_score = float(maximum)
            best_location = location
            best_size = (width, height)

    threshold = 0.78
    if best_score < threshold:
        return None, 0.0

    center_x = x1 + best_location[0] + best_size[0] / 2.0
    center_y = y1 + best_location[1] + best_size[1] / 2.0
    normalized = [center_x / image.width, center_y / image.height]
    if not (0.25 <= normalized[0] <= 0.75):
        return None, 0.0
    return [round(normalized[0], 6), round(normalized[1], 6)], best_score


def detect_battle_result(image: Image.Image) -> BattleResult:
    """Read the fixed red/blue crown slots on an offline result screen.

    This deliberately avoids OCR and returns ``unknown`` whenever a crown slot
    is mid-animation or the result confirmation button is not visible.
    """

    confirm_point, confirm_confidence = find_result_confirm_button(image)
    empty = (0.0, 0.0, 0.0)
    if confirm_point is None:
        return BattleResult("unknown", None, None, 0.0, empty, empty)

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    height, width = rgb.shape[:2]
    x_ranges = ((0.14, 0.36), (0.39, 0.61), (0.64, 0.86))
    row_ranges = ((0.125, 0.245), (0.445, 0.565))
    rows: list[tuple[float, float, float]] = []
    occupied_rows: list[list[bool]] = []
    ambiguous = False
    slot_confidences: list[float] = []
    for top, bottom in row_ranges:
        scores: list[float] = []
        occupied: list[bool] = []
        for left, right in x_ranges:
            patch = hsv[
                int(top * height) : int(bottom * height),
                int(left * width) : int(right * width),
            ]
            if patch.size == 0:
                ratio = 0.14
            else:
                gold = (
                    (patch[..., 0] >= 10)
                    & (patch[..., 0] <= 38)
                    & (patch[..., 1] >= 100)
                    & (patch[..., 2] >= 130)
                )
                ratio = float(np.mean(gold))
            scores.append(ratio)
            if ratio >= 0.22:
                occupied.append(True)
                slot_confidences.append(min(1.0, (ratio - 0.14) / 0.08))
            elif ratio <= 0.06:
                occupied.append(False)
                slot_confidences.append(min(1.0, (0.14 - ratio) / 0.08))
            else:
                occupied.append(False)
                ambiguous = True
                slot_confidences.append(0.0)
        rows.append(tuple(scores))
        occupied_rows.append(occupied)

    opponent_scores, player_scores = rows
    if ambiguous:
        return BattleResult(
            "unknown", None, None, 0.0, player_scores, opponent_scores
        )
    opponent_crowns = sum(occupied_rows[0])
    player_crowns = sum(occupied_rows[1])
    if player_crowns > opponent_crowns:
        outcome = "win"
    elif player_crowns < opponent_crowns:
        outcome = "loss"
    else:
        outcome = "draw"
    confidence = min([confirm_confidence, *slot_confidences])
    return BattleResult(
        outcome,
        player_crowns,
        opponent_crowns,
        confidence,
        player_scores,
        opponent_scores,
    )


def motion_score(current: Image.Image, previous: Image.Image | None, roi: list[float]) -> float:
    if previous is None or previous.size != current.size:
        return 0.0
    a = np.asarray(crop_normalized(current, roi).convert("L").resize((128, 96)), dtype=np.int16)
    b = np.asarray(crop_normalized(previous, roi).convert("L").resize((128, 96)), dtype=np.int16)
    difference = np.abs(a - b)
    return float(np.mean(difference > 18))
