"""Card metadata / sprite library. Browsing does not load a detector or train it."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .atomic_file import atomic_write
from .cards import CARD_ID_PATTERN, CardCatalog


# These are asset relationships, NOT substitutions for combat unit statistics.
# In particular, companions and spawned units retain their own YOLO classes.
CLASS_CARDS = {
    "archer": ("archers",), "barbarian": ("barbarians",),
    "bat": ("bats",), "elite-barbarian": ("elite_barbarians",),
    "goblin": ("goblins",), "guard": ("guards",), "minion": ("minions",),
    "royal-hog": ("royal_hogs",), "royal-recruit": ("royal_recruits",),
    "skeleton": ("skeletons",), "skeleton-dragon": ("skeleton_dragons",),
    "spear-goblin": ("spear_goblins",), "wall-breaker": ("wall_breakers",),
    "zappy": ("zappies",), "elixir-golem-big": ("elixir_golem",),
    "elixir-golem-mid": ("elixir_golem",), "elixir-golem-small": ("elixir_golem",),
    "phoenix-big": ("phoenix",), "phoenix-small": ("phoenix",),
    "phoenix-egg": ("phoenix",), "golemite": ("golem",),
    "lava-pup": ("lava_hound",), "rascal-boy": ("rascals",),
    "rascal-girl": ("rascals",), "goblin-brawler": ("goblin_cage",),
    "royal-guardian": ("little_prince",), "hog": ("mother_witch",),
    "skeleton-king-bar": ("skeleton_king",), "skeleton-king-skill": ("skeleton_king",),
    "ice-spirit-evolution-symbol": ("ice_spirit",),
}
SHARED_CLASSES = {
    "skeleton_army": ("skeleton",), "graveyard": ("skeleton",),
    "witch": ("skeleton",), "tombstone": ("skeleton",),
    "skeleton_barrel": ("skeleton",), "skeleton_king": ("skeleton",),
    "goblin_gang": ("goblin", "spear-goblin"), "goblin_barrel": ("goblin",),
    "goblin_drill": ("goblin",), "goblin_giant": ("spear-goblin",),
    "goblin_hut": ("spear-goblin",), "minion_horde": ("minion",),
    "three_musketeers": ("musketeer",), "barbarian_hut": ("barbarian",),
    "barbarian_barrel": ("barbarian",), "battle_ram": ("barbarian",),
    "night_witch": ("bat",), "furnace": ("fire-spirit",),
    "royal_delivery": ("royal-recruit",),
}
CLASS_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def encode_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def contained(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("素材路径超出库目录")
    return path


class CardLibrary:
    def __init__(self, root: Path, catalog_path: Path | None = None, slices_path: Path | None = None):
        self.root = root.resolve()
        self.catalog_path = (catalog_path or self.root / "data/cards.json").resolve()
        self.slices_path = (slices_path or self.root / "training/external/katacr-slices").resolve()
        self.local_path = self.root / "training/card-library"
        self.local_manifest = self.local_path / "manifest.json"
        self.reload()

    def reload(self):
        self.catalog = CardCatalog.load(self.catalog_path)
        self.raw_catalog = read_json(self.catalog_path)
        self.raw_cards = {c["id"]: c for c in self.raw_catalog["cards"]}
        self.directory = read_json(self.root / "data/deckbuilder_cards.json", {})
        self.website_cards = {c["id"]: c for c in self.directory.get("cards", [])}
        self.source_manifest = read_json(self.slices_path / "manifest.json", {})
        self.local = read_json(self.local_manifest, {"schema_version": 1, "records": [], "links": {}})
        if self.local.get("schema_version") != 1:
            raise ValueError("本地卡牌库版本不受支持")
        self.records = []
        for origin, directory, rows in (
            ("github", self.slices_path, self.source_manifest.get("records", [])),
            ("github", self.slices_path, self.source_manifest.get("auxiliary", [])),
            ("local", self.local_path, self.local.get("records", [])),
        ):
            for row in rows:
                if not CLASS_PATTERN.fullmatch(row["unit"]):
                    raise ValueError("无效切片类别")
                self.records.append({**row, "origin": origin, "file": contained(directory, row["path"])})
        self.classes = sorted({r["unit"] for r in self.records})
        self.links = {}
        for name in self.classes:
            base = name.removesuffix("-evolution")
            ids = CLASS_CARDS.get(base, (base.replace("-", "_"),))
            linked = {cid for cid in ids if self.catalog.get(cid)}
            # Shared normal sprites do not imply availability of evolved sprites.
            if name == base:
                linked.update(cid for cid, classes in SHARED_CLASSES.items()
                              if base in classes and self.catalog.get(cid))
            if name in self.local.get("links", {}):
                linked = set(self.local["links"][name]) & set(self.catalog.by_id)
            self.links[name] = sorted(linked)
        self.by_card = {cid: [r for r in self.records if cid in self.links[r["unit"]]]
                        for cid in self.catalog.by_id}
        self.unmapped = [r for r in self.records if not self.links[r["unit"]]]

    def search(self, query=""):
        needle = query.strip().casefold()
        return [c for c in self.catalog.cards if not needle or needle in
                " ".join((c.card_id, c.name_zh, c.name_en,
                          *[name for name, ids in self.links.items() if c.card_id in ids])).casefold()]

    @contextlib.contextmanager
    def _write_lock(self):
        self.local_path.mkdir(parents=True, exist_ok=True)
        lock = self.local_path / ".write.lock"
        try:
            handle = lock.open("x")
        except FileExistsError as exc:
            raise ValueError("卡牌库正在保存，请稍后重试") from exc
        try:
            handle.close()
            yield
        finally:
            lock.unlink(missing_ok=True)

    def set_links(self, unit: str, card_ids: list[str]):
        if unit not in self.classes or any(cid not in self.catalog.by_id for cid in card_ids):
            raise ValueError("请选择有效的切片类别与卡牌")
        with self._write_lock():
            local = read_json(self.local_manifest, {"schema_version": 1, "records": [], "links": {}})
            local.setdefault("links", {})[unit] = sorted(set(card_ids))
            atomic_write(self.local_manifest, encode_json(local))
        self.reload()

    @staticmethod
    def image_bytes(path: Path, *, sprite=False):
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if sprite and ("A" not in image.getbands() or not image.getchannel("A").getbbox()):
                raise ValueError(f"切片需要带透明通道且内容非空：{path.name}")
            size = list(image.size)
            if image.format == "PNG":
                return data, size
            image.convert("RGBA").save(buffer := io.BytesIO(), format="PNG")
        return buffer.getvalue(), size

    def add_slices(self, paths: list[Path], unit: str, affiliation: int, card_ids: list[str]):
        if not CLASS_PATTERN.fullmatch(unit) or affiliation not in (0, 1):
            raise ValueError("类别只能使用小写字母、数字、短横线或下划线；来源阵营需为 0 或 1")
        if not paths or not card_ids or any(cid not in self.catalog.by_id for cid in card_ids):
            raise ValueError("请选择图片和对应卡牌")
        # Validate the whole batch before persisting any images.
        incoming = [self.image_bytes(Path(path), sprite=True) for path in paths]
        count = 0
        with self._write_lock():
            self.reload()
            local = self.local
            known = {(r["unit"], r["affiliation"], r["sha256"]) for r in self.records}
            for data, size in incoming:
                digest = hashlib.sha256(data).hexdigest()
                key = (unit, affiliation, digest)
                if key in known:
                    continue
                relative = f"sprites/{unit}/{unit}_{affiliation}_{digest}.png"
                atomic_write(contained(self.local_path, relative), data)
                local["records"].append(dict(path=relative, unit=unit, affiliation=affiliation,
                                            sha256=digest, size=size))
                known.add(key)
                count += 1
            # Adding to an existing shared class preserves all its relationships.
            local.setdefault("links", {})[unit] = sorted(set(self.links.get(unit, [])) | set(card_ids))
            atomic_write(self.local_manifest, encode_json(local))
        self.reload()
        return count

    def add_card(self, *, card_id: str, name_zh: str, name_en: str, elixir: int,
                 kind: str, rarity: str, icon: Path):
        if not CARD_ID_PATTERN.fullmatch(card_id):
            raise ValueError("卡牌 ID 只能使用小写字母、数字和下划线")
        if not (name_zh.strip() or name_en.strip()) or not 0 <= elixir <= 10:
            raise ValueError("请填写卡牌名称和 0–10 的圣水费用")
        if kind not in {"troop", "building", "spell"}:
            raise ValueError("卡牌类型无效")
        data, _ = self.image_bytes(icon)
        digest = hashlib.sha256(data).hexdigest()
        with self._write_lock():
            before = self.catalog_path.read_bytes()
            payload = json.loads(before)
            if any(c["id"] == card_id for c in payload["cards"]):
                raise ValueError("此卡牌 ID 已存在")
            target = self.local_path / "icons" / f"{card_id}-{digest}.png"
            atomic_write(target, data)
            import os
            payload["cards"].append(dict(id=card_id, name_zh=name_zh.strip(), name_en=name_en.strip(),
                elixir=elixir, kind=kind, rarity=rarity, official_id=None, max_level=None,
                icon_path=Path(os.path.relpath(target, self.catalog_path.parent)).as_posix(),
                icon_variants=[], icon_url="", targets=[], roles=[kind], counters=[], synergies=[],
                notes="本地添加；战斗属性与识别支持待补充", library_source="local"))
            # Keep the original source fields and a content-addressed recovery copy.
            atomic_write(self.local_path / "backups" / (hashlib.sha256(before).hexdigest() + ".json"), before)
            if self.catalog_path.read_bytes() != before:
                raise ValueError("卡牌数据已被其他操作更新，请刷新后重试")
            atomic_write(self.catalog_path, encode_json(payload))
        self.reload()

    def export_snapshot(self, output: Path):
        """New self-contained manifest consumable by slice_dataset export/export-real."""
        output = output.resolve()
        if output.exists():
            raise ValueError("请选择尚不存在的输出目录")
        if output.is_relative_to(self.slices_path) or output.is_relative_to(self.local_path):
            raise ValueError("输出不能放在原始素材库内部")
        with self._write_lock():
            self.reload()
            # Complete validation first. Do not create an apparently usable partial manifest.
            for record in self.records:
                if hashlib.sha256(record["file"].read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError(f"切片校验失败：{record['file'].name}")
            output.mkdir(parents=True, exist_ok=False)
            rows, auxiliary, seen = [], [], set()
            for record in self.records:
                key = (record["unit"], record.get("affiliation"), record["sha256"])
                if key in seen:
                    continue
                seen.add(key)
                relative = f"sprites/{record['unit']}/{record['affiliation']}_{record['sha256']}.png"
                atomic_write(contained(output, relative), record["file"].read_bytes())
                row = {k: v for k, v in record.items() if k not in {"file", "path"}}
                row["path"] = relative
                row["card_ids"] = self.links[record["unit"]]
                (auxiliary if record.get("affiliation") is None else rows).append(row)
            result = dict(schema_version=1, kind="external_slices", source=self.source_manifest.get("source", "local"),
                revision=self.source_manifest.get("revision", "local"), created_at=datetime.now(timezone.utc).isoformat(),
                library_customized=bool(self.local["records"] or self.local.get("links")),
                classes=sorted({r["unit"] for r in rows}), records=rows, auxiliary=auxiliary,
                count=len(rows), auxiliary_count=len(auxiliary), total_images=len(rows)+len(auxiliary),
                card_class_links=self.links, promoted=False, battle_acceptance=False)
            if (license_path := self.slices_path / "LICENSE").exists():
                atomic_write(output / "LICENSE", license_path.read_bytes())
            atomic_write(output / "manifest.json", encode_json(result))
        return result
