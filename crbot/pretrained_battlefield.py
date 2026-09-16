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


def classify_side(image, bbox, scores, threshold=.85):
    """Use reliable classifier scores, or strong badge color for ambiguous crops.

    The side model exports two independent sigmoid scores, not softmax logits.
    Position on the board is never used to guess allegiance.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) != 2 or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        return None
    index = int(scores.argmax())
    if scores[index] >= threshold and scores[index] - scores[1-index] >= .35:
        return (1 if index == 0 else -1), float(scores[index]), "side_model"
    w, h = image.size
    x1, y1, x2, y2 = bbox
    box = (round(x1*w), max(0, round(y1*h-.005*h)), round(x2*w), round((y1+.23*(y2-y1))*h))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    pixels = np.asarray(image.crop(box).convert("RGB"), dtype=float)
    r, g, b = pixels.transpose(2, 0, 1)
    red = int(((r>105)&(r>g*1.5)&(r>b*1.1)&(g<120)).sum())
    blue = int(((b>110)&(b>r*1.5)&(b>g*1.05)).sum())
    minimum = max(15, round(w*h * .000035))
    if max(red, blue) >= minimum and max(red, blue) >= 3 * max(1, min(red, blue)):
        return (-1 if red>blue else 1), .85, "top_color_evidence"
    return None


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
        # This tiny 16x16 network is faster on CPU and avoids many GPU transfers.
        self.side_session = ort.InferenceSession(str(directory / "side.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        if inp.shape != [1, 3, 480, 352] or inp.type != "tensor(float16)":
            raise ValueError("战场模型输入契约不匹配")
        self.input_name = inp.name
        self.side_input = self.side_session.get_inputs()[0].name
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
        result = []
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
            crop = image.crop(tuple(round(v*s) for v,s in zip(bbox, [image.width,image.height]*2))).convert("RGB")
            if not crop.width or not crop.height:
                continue
            pixels = np.asarray(crop.resize((16,16), Image.Resampling.BICUBIC), dtype=np.float32)[None] / 255
            scores = self.side_session.run(None, {self.side_input: pixels})[0][0]
            side = classify_side(image, bbox, scores, float(self.config.get("pretrained_side_confidence", .85)))
            if side is None:
                rejected["side_unknown"] += 1
                continue
            team, side_confidence, evidence = side
            result.append(dict(card_id=card_id, unit_name=unit_name, side=team, bbox=bbox,
                confidence=min(confidence, side_confidence), detection_confidence=confidence,
                side_confidence=side_confidence, side_evidence=evidence,
                identity_source="public_pretrained", deployment_confirmed=False))
        self.calls += 1
        self.gpu_calls += self.device == "cuda"
        self.detected_units += len(result)
        self.last_ms = (time.perf_counter()-started)*1000
        self.rejected = rejected
        return result

    def status(self):
        return dict(version=self.metadata["version"], source="public_pretrained", device=self.device,
                    providers=self.providers, side_device="cpu", inference_calls=self.calls,
                    gpu_inference_calls=self.gpu_calls, detected_units=self.detected_units,
                    last_ms=round(self.last_ms,3), rejected=self.rejected,
                    local_quality_validated=False, battle_acceptance=False)
