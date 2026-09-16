"""ONNX adapter for the Pbatch YOLOv10 unit and team models.

Preprocessing and class aliases follow the pinned upstream model contract.
See docs/battlefield-recognition.md and THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import ast
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .battlefield_assets import RELATIVE_ROOT, assets_ready, manifest
from .unit_badges import find_level_badges, associate_badges

# Unit identities are representatives, not proof that a particular card was played.
# Unsupported split/companion entities must not acquire the parent's statistics.
ALIASES = {
    "archer": "archers", "barbarian": "barbarians", "bat": "bats",
    "elite_barbarian": "elite_barbarians", "elixir_golem_large": "elixir_golem",
    "goblin": "goblins", "guard": "guards", "hungry_dragon": "baby_dragon",
    "minion": "minions", "minipekka": "mini_pekka", "phoenix_large": "phoenix",
    "royal_hog": "royal_hogs", "royal_recruit": "royal_recruits",
    "skeleton": "skeletons", "skeleton_dragon": "skeleton_dragons",
    "spear_goblin": "spear_goblins", "wall_breaker": "wall_breakers", "zappy": "zappies",
}
UNSUPPORTED = frozenset({"brawler", "elixir_golem_medium", "elixir_golem_small",
    "giant_snowball", "golemite", "hog", "lava_pup", "phoenix_egg", "phoenix_small",
    "rascal_boy", "rascal_girl", "royal_guardian"})


def unit_card_id(name, catalog):
    if name in UNSUPPORTED:
        return None
    card_id = ALIASES.get(name, name)
    card = catalog.get(card_id)
    return card_id if card is not None and card.kind in {"troop", "building"} else None


def prepare_image(image, height=480, width=352):
    """Return NCHW FP16 input and the exact rounded crop/letterbox transform."""
    image = image.convert("RGB")
    top, bottom = round(.05 * image.height), round(.80 * image.height)
    crop = image.crop((0, top, image.width, bottom))
    scale = min(width / crop.width, height / crop.height)
    nw, nh = max(1, int(crop.width * scale)), max(1, int(crop.height * scale))
    left, padtop = (width - nw + 1) // 2, (height - nh + 1) // 2
    pixels = np.full((height, width, 3), 114, dtype=np.float16)
    pixels[padtop:padtop + nh, left:left + nw] = np.asarray(crop.resize((nw, nh)), dtype=np.float16)
    tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None] / 255)
    return tensor, (left, padtop, nw, nh, top, bottom, image.width, image.height)


def restore_box(box, transform):
    left, padtop, nw, nh, top, bottom, w, h = transform
    x1, y1, x2, y2 = box
    values = [(x1-left)/nw, ((y1-padtop)*(bottom-top)/nh+top)/h,
              (x2-left)/nw, ((y2-padtop)*(bottom-top)/nh+top)/h]
    return [float(np.clip(v, 0, 1)) for v in values]


def classify_side(image, bbox, scores=None, threshold=.85):
    """A visible level badge owns team identity; legacy model scores cannot override it."""
    badge = associate_badges([bbox], find_level_badges(image)).get(0)
    return (badge['side'], badge['confidence'], badge['evidence']) if badge else None


class PretrainedBattlefield:
    def __init__(self, root: Path, catalog, config):
        if not assets_ready(root):
            raise ValueError("战场预训练权重缺失或校验失败；运行 setup_battlefield.bat")
        import onnxruntime as ort
        self.catalog, self.config = catalog, config
        self.metadata = manifest()
        requested = os.environ.get("CRBOT_COMPUTE_DEVICE", str(config.get("device", "auto"))).lower()
        providers = ["CPUExecutionProvider"]
        if requested != "cpu":
            try:
                import torch  # Preloads the matching CUDA 12/cuDNN 9 DLLs on Windows.
                if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
                    device_id = int(requested.split(":")[-1]) if requested.startswith("cuda:") or requested.isdigit() else 0
                    providers.insert(0, ("CUDAExecutionProvider", {"device_id": device_id,
                        "cudnn_conv_algo_search": "HEURISTIC"}))
            except ImportError:
                pass
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.log_severity_level = 3
        if config.get("pretrained_profile"):
            options.enable_profiling = True
            options.profile_file_prefix = str(config["pretrained_profile"])
        directory = root / RELATIVE_ROOT
        self.session = ort.InferenceSession(str(directory / "units_M_480x352.onnx"), sess_options=options, providers=providers)
        self.providers = self.session.get_providers()
        if requested not in {"auto", "cpu"} and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError("指定 CUDA 但战场识别未能加载 CUDAExecutionProvider")
        self.device = "cuda" if "CUDAExecutionProvider" in self.providers else "cpu"
        inp = self.session.get_inputs()[0]
        if inp.shape != [1, 3, 480, 352] or inp.type != "tensor(float16)":
            raise ValueError("战场模型输入契约不匹配")
        self.input_name = inp.name
        self.names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map["names"])
        if not isinstance(self.names, dict) or set(self.names) != set(range(97)):
            raise ValueError("战场模型类别表不匹配")
        self.calls = self.gpu_calls = self.detected_units = 0
        self.last_ms = 0.
        self.rejected = {}
        # Compile/allocate during startup, never on the first live battle frame.
        tensor, _ = prepare_image(Image.new("RGB", (540, 960)))
        self.session.run(None, {self.input_name: tensor})

    def detect(self, image):
        started = time.perf_counter()
        tensor, transform = prepare_image(image)
        rows = np.asarray(self.session.run(None, {self.input_name: tensor})[0])[0]
        threshold = float(self.config.get("pretrained_confidence", .55))
        result = []; candidates = []
        rejected = {"unmapped": 0, "side_unknown": 0, "invalid": 0}
        for row in rows:
            if len(row) != 6 or not np.isfinite(row).all():
                rejected["invalid"] += 1
                continue
            x1,y1,x2,y2,confidence,index = map(float, row)
            if not threshold <= confidence <= 1:
                continue
            if index != int(index) or int(index) not in self.names:
                rejected["invalid"] += 1
                continue
            unit_name = self.names[int(index)]
            card_id = unit_card_id(unit_name, self.catalog)
            if card_id is None:
                rejected["unmapped"] += 1
                continue
            bbox = restore_box([x1,y1,x2,y2], transform)
            x1,y1,x2,y2 = bbox
            if x2 <= x1 or y2 <= y1 or not .20 <= (y1+y2)/2 <= .80:
                rejected["invalid"] += 1
                continue
            candidates.append((card_id, unit_name, bbox, confidence))
        badges = associate_badges([c[2] for c in candidates], find_level_badges(image))
        for index, (card_id, unit_name, bbox, confidence) in enumerate(candidates):
            badge = badges.get(index)
            if badge is None:
                rejected["side_unknown"] += 1
                continue
            team, side_confidence, evidence = badge['side'], badge['confidence'], badge['evidence']
            result.append(dict(card_id=card_id, unit_name=unit_name, side=team, bbox=bbox,
                confidence=min(confidence, side_confidence), detection_confidence=confidence,
                side_confidence=side_confidence, side_evidence=evidence,
                level=badge['level'], level_badge_bbox=badge['bbox'],
                identity_source="public_pretrained", deployment_confirmed=False))
        self.calls += 1
        self.gpu_calls += self.device == "cuda"
        self.detected_units += len(result)
        self.last_ms = (time.perf_counter()-started)*1000
        self.rejected = rejected
        return result

    def status(self):
        return dict(version=self.metadata["version"], source="public_pretrained", device=self.device,
                    providers=self.providers, side_device="cpu", side_method="level_badge", inference_calls=self.calls,
                    gpu_inference_calls=self.gpu_calls, detected_units=self.detected_units,
                    last_ms=round(self.last_ms,3), rejected=self.rejected,
                    local_quality_validated=False, battle_acceptance=False)
