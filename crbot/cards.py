from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CARD_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


@dataclass(frozen=True)
class CardDefinition:
    card_id: str
    official_id: int | None
    name_zh: str
    name_en: str
    elixir: int | None
    rarity: str
    max_level: int | None
    icon_url: str
    icon_path: str
    icon_variants: tuple[str, ...]
    kind: str
    targets: tuple[str, ...]
    roles: tuple[str, ...]
    counters: tuple[str, ...]
    synergies: tuple[str, ...]
    notes: str = ""
    description: str = ""


class CardCatalog:
    """Versioned, deck-independent card knowledge used by labeling and policy."""

    def __init__(
        self,
        cards: list[CardDefinition],
        *,
        source: str = "manual",
        catalog_path: Path | None = None,
    ):
        self.cards = cards
        self.source = source
        self.catalog_path = catalog_path
        self.by_id = {card.card_id: card for card in cards}
        if len(self.by_id) != len(cards):
            raise ValueError("卡牌知识库包含重复 card_id")

    @classmethod
    def load(cls, path: Path) -> "CardCatalog":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("不支持的卡牌知识库版本")
        definitions: list[CardDefinition] = []
        for raw in payload.get("cards", []):
            card_id = str(raw.get("id", "")).strip()
            if not CARD_ID_PATTERN.fullmatch(card_id):
                raise ValueError(f"无效 card_id：{card_id!r}")
            raw_elixir = raw.get("elixir")
            elixir = None if raw_elixir is None else int(raw_elixir)
            if elixir is not None and not 0 <= elixir <= 10:
                raise ValueError(f"{card_id} 的圣水费用无效")
            definitions.append(
                CardDefinition(
                    card_id=card_id,
                    official_id=(
                        None
                        if raw.get("official_id") is None
                        else int(raw["official_id"])
                    ),
                    name_zh=str(raw.get("name_zh", "")),
                    name_en=str(raw.get("name_en", "")),
                    elixir=elixir,
                    rarity=str(raw.get("rarity", "")),
                    max_level=(
                        None if raw.get("max_level") is None else int(raw["max_level"])
                    ),
                    icon_url=str(raw.get("icon_url", "")),
                    icon_path=str(raw.get("icon_path", "")),
                    icon_variants=tuple(
                        str(value) for value in raw.get("icon_variants", [])
                    ),
                    kind=str(raw.get("kind", "unknown")),
                    targets=tuple(str(value) for value in raw.get("targets", [])),
                    roles=tuple(str(value) for value in raw.get("roles", [])),
                    counters=tuple(str(value) for value in raw.get("counters", [])),
                    synergies=tuple(str(value) for value in raw.get("synergies", [])),
                    notes=str(raw.get("notes", "")),
                    description=str(raw.get("description", "")),
                )
            )
        return cls(
            definitions,
            source=str(payload.get("source", "manual")),
            catalog_path=path.resolve(),
        )

    def ids(self) -> list[str]:
        return sorted(self.by_id)

    def get(self, card_id: str) -> CardDefinition | None:
        return self.by_id.get(card_id)

    def local_icon_path(self, card: CardDefinition) -> Path | None:
        if not card.icon_path or self.catalog_path is None:
            return None
        path = Path(card.icon_path)
        if not path.is_absolute():
            path = self.catalog_path.parent / path
        return path.resolve()

    def local_icon_paths(self, card: CardDefinition) -> list[Path]:
        values = ([card.icon_path] if card.icon_path else []) + list(card.icon_variants)
        paths: list[Path] = []
        for value in values:
            path = Path(value)
            if not path.is_absolute() and self.catalog_path is not None:
                path = self.catalog_path.parent / path
            paths.append(path.resolve())
        return paths

    def search(self, query: str) -> list[CardDefinition]:
        needle = query.strip().casefold()
        if not needle:
            return list(self.cards)
        return [
            card
            for card in self.cards
            if needle in card.card_id.casefold()
            or needle in card.name_zh.casefold()
            or needle in card.name_en.casefold()
        ]
