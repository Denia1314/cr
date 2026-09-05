from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


OFFICIAL_CARDS_URL = "https://api.clashroyale.com/v1/cards"
COMMUNITY_CARDS_URL = "https://royaleapi.github.io/cr-api-data/json/cards.json"
COMMUNITY_ICON_URL = (
    "https://raw.githubusercontent.com/RoyaleAPI/cr-api-assets/master/cards/{key}.png"
)
COMMUNITY_ASSET_TREE_URL = (
    "https://api.github.com/repos/RoyaleAPI/cr-api-assets/git/trees/master?recursive=1"
)
COMMUNITY_VARIANT_ICON_URL = (
    "https://raw.githubusercontent.com/RoyaleAPI/cr-api-assets/master/cards-150/{name}"
)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "unknown_card"


def fetch_official_cards(token: str, *, url: str = OFFICIAL_CARDS_URL) -> dict[str, Any]:
    if not token.strip():
        raise ValueError("缺少 Clash Royale API token")
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {token.strip()}",
            "Accept": "application/json",
            "User-Agent": "Royal-Lab-Dataset/1",
        },
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed official URL by default
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("官方卡牌接口返回格式无效")
    return payload


def merge_official_cards(
    existing: dict[str, Any],
    official_payload: dict[str, Any],
) -> dict[str, Any]:
    items = official_payload.get("items")
    if not isinstance(items, list):
        raise ValueError("官方卡牌 JSON 缺少 items 数组")

    existing_cards = [value for value in existing.get("cards", []) if isinstance(value, dict)]
    by_official_id = {
        int(value["official_id"]): value
        for value in existing_cards
        if value.get("official_id") is not None
    }
    by_name = {
        str(value.get("name_en", "")).casefold(): value
        for value in existing_cards
        if value.get("name_en")
    }

    merged_cards: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for raw in items:
        if not isinstance(raw, dict) or raw.get("id") is None or not raw.get("name"):
            continue
        official_id = int(raw["id"])
        name_en = str(raw["name"])
        previous = dict(
            by_official_id.get(official_id)
            or by_name.get(name_en.casefold())
            or {}
        )
        base_id = str(previous.get("id") or _slug(name_en))
        card_id = base_id
        if card_id in used_ids:
            card_id = f"{base_id}_{official_id}"
        used_ids.add(card_id)

        icon_urls = raw.get("iconUrls") if isinstance(raw.get("iconUrls"), dict) else {}
        previous.update(
            {
                "id": card_id,
                "official_id": official_id,
                "name_en": name_en,
                "name_zh": str(previous.get("name_zh", "")),
                "elixir": raw.get("elixirCost"),
                "rarity": str(raw.get("rarity", "")),
                "max_level": raw.get("maxLevel"),
                "icon_url": str(icon_urls.get("medium", "")),
                "icon_path": str(previous.get("icon_path", "")),
                "icon_variants": list(previous.get("icon_variants", [])),
                "kind": str(previous.get("kind", "unknown")),
                "targets": list(previous.get("targets", [])),
                "roles": list(previous.get("roles", [])),
                "counters": list(previous.get("counters", [])),
                "synergies": list(previous.get("synergies", [])),
                "notes": str(previous.get("notes", "")),
            }
        )
        merged_cards.append(previous)

    merged_cards.sort(key=lambda value: (str(value.get("name_en", "")), int(value["official_id"])))
    return {
        "schema_version": 1,
        "source": "official_api_with_manual_strategy_enrichment",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "field_help": existing.get("field_help", {}),
        "cards": merged_cards,
    }


def _derived_roles(card: dict[str, Any]) -> list[str]:
    description = str(card.get("description", "")).casefold()
    kind = str(card.get("type", "unknown")).casefold()
    key = str(card.get("key", "")).casefold()
    elixir = card.get("elixir")
    roles: set[str] = set()
    if kind == "spell":
        roles.add("spell")
    elif kind == "building":
        roles.add("building")
    elif kind == "troop":
        roles.add("troop")
    if isinstance(elixir, int) and elixir <= 2:
        roles.add("cheap")
    if any(
        phrase in description
        for phrase in (
            "area damage",
            "all enemies",
            "everything in its path",
            "anything in its path",
            "explodes",
            "splash",
        )
    ):
        roles.add("splash")
    if "flying" in description or "from the sky" in description:
        roles.add("air")
    if any(word in description for word in ("ranged", "shoots", "throws", "long range")):
        roles.add("ranged")
    if kind == "troop" and any(
        word in description for word in ("durable", "tough", "heavily armored")
    ):
        roles.add("tank")
    if "only attacks buildings" in description or "targets buildings" in description:
        roles.update(("building_target", "win_condition"))
    if any(
        phrase in description
        for phrase in ("periodically spawns", "summons", "spawns every", "when destroyed, spawns")
    ):
        roles.add("spawner")
    if any(
        phrase in description
        for phrase in ("massive damage", "big damage", "increases in damage", "huge damage")
    ):
        roles.add("tank_killer")
    if any(word in description for word in ("three ", "four ", "six ", "army", "horde")):
        roles.add("swarm")
    if key in {
        "pekka",
        "mini-pekka",
        "inferno-dragon",
        "inferno-tower",
        "hunter",
        "prince",
        "elite-barbarians",
        "mighty-miner",
        "sparky",
        "lumberjack",
    }:
        roles.add("tank_killer")
    if key in {
        "balloon",
        "battle-ram",
        "elixir-golem",
        "goblin-barrel",
        "goblin-drill",
        "graveyard",
        "miner",
        "mortar",
        "ram-rider",
        "royal-hogs",
        "skeleton-barrel",
        "x-bow",
    }:
        roles.add("win_condition")
    if key in {
        "electro-giant",
        "giant",
        "giant-skeleton",
        "goblin-giant",
        "golem",
        "lava-hound",
        "mega-knight",
        "pekka",
        "royal-giant",
    }:
        roles.add("tank")
    if roles.intersection({"ranged", "splash", "spawner"}) and kind == "troop" and not roles.intersection(
        {"tank", "win_condition"}
    ):
        roles.add("support")
    return sorted(roles)


def _derived_targets(card: dict[str, Any]) -> list[str]:
    description = str(card.get("description", "")).casefold()
    if "only attacks buildings" in description or "targets buildings" in description:
        return ["buildings"]
    if "ground and air" in description:
        return ["ground", "air"]
    return []


def merge_community_cards(
    existing: dict[str, Any],
    community_payload: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(community_payload, list):
        raise ValueError("社区卡牌 JSON 必须是数组")
    existing_cards = [value for value in existing.get("cards", []) if isinstance(value, dict)]
    by_official_id = {
        int(value["official_id"]): value
        for value in existing_cards
        if value.get("official_id") is not None
    }
    by_id = {str(value.get("id", "")): value for value in existing_cards}
    merged_cards: list[dict[str, Any]] = []
    for raw in community_payload:
        if raw.get("id") is None or not raw.get("key") or not raw.get("name"):
            continue
        official_id = int(raw["id"])
        card_id = str(raw["key"]).replace("-", "_")
        previous = dict(by_official_id.get(official_id) or by_id.get(card_id) or {})
        previous.update(
            {
                "id": card_id,
                "official_id": official_id,
                "name_en": str(raw["name"]),
                "name_zh": str(previous.get("name_zh", "")),
                "elixir": raw.get("elixir"),
                "rarity": str(raw.get("rarity", "")).casefold(),
                "max_level": previous.get("max_level"),
                "icon_url": COMMUNITY_ICON_URL.format(key=raw["key"]),
                "icon_path": f"card_icons/{raw['key']}.png",
                "icon_variants": list(previous.get("icon_variants", [])),
                "kind": str(raw.get("type", "unknown")).casefold(),
                "targets": list(
                    previous.get("targets", [])
                    if previous.get("strategy_verified")
                    else _derived_targets(raw)
                ),
                "roles": list(
                    previous.get("roles", [])
                    if previous.get("strategy_verified")
                    else _derived_roles(raw)
                ),
                "counters": list(previous.get("counters", [])),
                "synergies": list(previous.get("synergies", [])),
                "notes": str(previous.get("notes", "")),
                "arena": raw.get("arena"),
                "description": str(raw.get("description", "")),
                "sc_key": str(raw.get("sc_key", "")),
                "strategy_verified": bool(previous.get("strategy_verified", False)),
            }
        )
        merged_cards.append(previous)
    merged_cards.sort(key=lambda value: int(value["official_id"]))
    field_help = dict(existing.get("field_help", {}))
    field_help.update(
        {
            "icon_path": "本地基础卡图，相对于卡牌目录 JSON",
            "icon_variants": "本地进化与英雄卡图列表，相对于卡牌目录 JSON",
            "strategy_verified": "战术字段是否已经人工核对；未核对时同步会重新派生",
        }
    )
    return {
        "schema_version": 1,
        "source": "royaleapi_static_with_derived_roles",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "field_help": field_help,
        "cards": merged_cards,
    }


def bootstrap_community_catalog(
    catalog_path: Path,
    *,
    cards_url: str = COMMUNITY_CARDS_URL,
    download_icons: bool = True,
) -> tuple[int, int]:
    if catalog_path.is_file():
        with catalog_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    else:
        existing = {"schema_version": 1, "cards": []}
    with urlopen(cards_url, timeout=30) as response:  # noqa: S310 - configured dataset URL
        payload = json.load(response)
    merged = merge_community_cards(existing, payload)

    variant_names: set[str] = set()
    try:
        request = Request(COMMUNITY_ASSET_TREE_URL, headers={"User-Agent": "Royal-Lab-Dataset/1"})
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed GitHub API URL
            tree_payload = json.load(response)
        for item in tree_payload.get("tree", []):
            path = str(item.get("path", ""))
            if path.startswith("cards-150/") and path.endswith(".png"):
                variant_names.add(Path(path).name)
    except (OSError, ValueError):
        variant_names = set()

    allowed_suffixes = ("-ev1.png", "-hero.png", "-hero-ev1.png")
    for card in merged["cards"]:
        key = Path(str(card["icon_path"])).stem
        variants = sorted(
            name for name in variant_names if name in {f"{key}{suffix}" for suffix in allowed_suffixes}
        )
        card["icon_variants"] = [f"card_variants/{name}" for name in variants]

    downloaded = 0
    if download_icons:
        icon_dir = catalog_path.parent / "card_icons"
        icon_dir.mkdir(parents=True, exist_ok=True)
        for card in merged["cards"]:
            target = icon_dir / Path(str(card["icon_path"])).name
            if target.is_file() and target.stat().st_size > 1000:
                downloaded += 1
                continue
            request = Request(str(card["icon_url"]), headers={"User-Agent": "Royal-Lab-Dataset/1"})
            try:
                with urlopen(request, timeout=20) as response:
                    content = response.read()
            except OSError:
                continue
            if len(content) <= 1000:
                continue
            temporary_icon = target.with_suffix(target.suffix + ".tmp")
            temporary_icon.write_bytes(content)
            temporary_icon.replace(target)
            downloaded += 1

        variant_dir = catalog_path.parent / "card_variants"
        variant_dir.mkdir(parents=True, exist_ok=True)
        for card in merged["cards"]:
            for variant_path in card.get("icon_variants", []):
                target = catalog_path.parent / str(variant_path)
                if target.is_file() and target.stat().st_size > 1000:
                    downloaded += 1
                    continue
                url = COMMUNITY_VARIANT_ICON_URL.format(name=target.name)
                request = Request(url, headers={"User-Agent": "Royal-Lab-Dataset/1"})
                try:
                    with urlopen(request, timeout=20) as response:
                        content = response.read()
                except OSError:
                    continue
                if len(content) <= 1000:
                    continue
                temporary_icon = target.with_suffix(target.suffix + ".tmp")
                temporary_icon.write_bytes(content)
                temporary_icon.replace(target)
                downloaded += 1

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = catalog_path.with_suffix(catalog_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(catalog_path)
    return len(merged["cards"]), downloaded


def sync_card_catalog(
    catalog_path: Path,
    *,
    token: str | None = None,
    input_path: Path | None = None,
) -> int:
    if catalog_path.is_file():
        with catalog_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    else:
        existing = {"schema_version": 1, "cards": []}

    if input_path is not None:
        with input_path.open("r", encoding="utf-8") as handle:
            official_payload = json.load(handle)
    else:
        official_payload = fetch_official_cards(token or "")

    merged = merge_official_cards(existing, official_payload)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = catalog_path.with_suffix(catalog_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(catalog_path)
    return len(merged["cards"])
