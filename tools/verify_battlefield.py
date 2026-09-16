"""Read-only inference through the real adapter on stored battle screenshots."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw

from crbot.battle_perception import LaneThreat
from crbot.cards import CardCatalog
from crbot.learned_perception import LearnedBattlefieldDetector


def verify(root, *, runs=3, frames_per_run=8, cpu=False):
    output = root / "reports" / "battlefield-recognition" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    training = {**config.get("training", {}), "pretrained_profile": str(output / "onnx-profile")}
    if cpu:
        training["device"] = "cpu"
    detector = LearnedBattlefieldDetector(root, CardCatalog.load(root / config.get("dataset", {}).get("card_catalog", "data/cards.json")), training)
    if not detector.available:
        raise RuntimeError(detector.error or "battlefield detector unavailable")
    rows, identities, sides = [], Counter(), Counter()
    folders = sorted((root / "runs").glob("*"), reverse=True)
    used = 0
    for folder in folders:
        paths = sorted((folder / "samples").glob("*battle*.jpg"))
        if not paths:
            continue
        chosen = sorted(set(round(i*(len(paths)-1)/max(1, frames_per_run-1)) for i in range(frames_per_run)))
        for index in chosen:
            path = paths[index]
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            fallback = {lane: LaneThreat(lane, 0, 0, 0, "none", ()) for lane in ("left", "right")}
            started = time.perf_counter()
            detector.detect(image, fallback)
            elapsed = (time.perf_counter()-started)*1000
            if not detector.last_detection_succeeded:
                raise RuntimeError(detector.error)
            observations = [dict(d, team="ally") for d in detector.observed_allies] + [dict(d, team="enemy") for d in detector.observed_enemies]
            draw = ImageDraw.Draw(image)
            for d in observations:
                identities[d["card_id"]] += 1
                sides[d["team"]] += 1
                if "bbox" in d:
                    box = [v*s for v,s in zip(d["bbox"], [image.width,image.height]*2)]
                    color = "cyan" if d["team"] == "ally" else "red"
                    draw.rectangle(box, outline=color, width=2)
                    draw.text((box[0],max(0,box[1]-12)), f"{d['card_id']} {d['confidence']:.2f} {d['team']}",fill=color)
            filename = f"{folder.name}_{path.name}"
            image.save(output / filename)
            rows.append(dict(frame=str(path.relative_to(root)), elapsed_ms=round(elapsed,3), observations=observations, preview=filename))
        used += 1
        if used >= runs:
            break
    if not rows:
        raise ValueError("没有可用于验证的本地战斗截图")
    providers = Counter()
    if detector.pretrained is not None:
        profile = Path(detector.pretrained.session.end_profiling())
        for event in json.loads(profile.read_text(encoding="utf-8")):
            provider = event.get("args", {}).get("provider")
            if provider:
                providers[provider] += 1
    summary = dict(status=detector.status(), frames=len(rows), runs=used,
                   identified_observations=sum(identities.values()), cards=dict(identities), teams=dict(sides),
                   median_pipeline_ms=sorted(r["elapsed_ms"] for r in rows)[len(rows)//2],
                   executed_nodes_by_provider=dict(providers),
                   accuracy_measured=False, battle_acceptance=False,
                   limitation="Stored screenshots, no ground-truth benchmark; detection counts are not precision or recall.")
    (output / "results.json").write_text(json.dumps(dict(summary=summary, frames=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(report=str(output / "results.json"), **summary), ensure_ascii=False, indent=2))
    return output, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    verify(Path(__file__).resolve().parents[1], runs=max(1,args.runs), frames_per_run=max(1,args.frames), cpu=args.cpu)
