from __future__ import annotations

from dataclasses import dataclass

from .cards import CardDefinition


@dataclass(frozen=True)
class CardTactics:
    formation_role: str
    frontline_score: float
    backline_score: float
    is_win_condition: bool
    is_defensive: bool
    splash_strength: float = 0.0
    heavy_counter_strength: float = 0.0
    offensive_commitment: float = 0.0

    @property
    def is_frontline(self) -> bool:
        return self.formation_role in {"frontline", "hybrid"}

    @property
    def is_backline(self) -> bool:
        return self.formation_role in {"backline", "hybrid"}


FRONTLINE_PHRASES = (
    "melee",
    "durable",
    "tough",
    "heavily armored",
    "shield",
    "charges",
    "charge damage",
    "huge punch",
)

BACKLINE_PHRASES = (
    "ranged",
    "shoots",
    " shot",
    "throws",
    "beam",
    "projectile",
    "long range",
    "fireball",
    "arrows",
    "bow",
    "boomstick",
    "slingshot",
    "fragile",
    "from behind",
    "zap machine",
)

BUILDING_TARGET_PHRASES = (
    "targets buildings",
    "towards enemy buildings",
    "toward enemy buildings",
    "attacks buildings",
    "goes straight for buildings",
    "buildings only",
)

DIRECT_AREA_PHRASES = (
    "deal area damage",
    "deals area damage",
    "huge area damage",
    "splash damage",
    "damages nearby enemies",
    "attacks multiple",
)

DEATH_EFFECT_PHRASES = (
    "death damage",
    "upon death",
    "when destroyed",
    "explodes when destroyed",
    "on death",
)


def card_tactics(card: CardDefinition) -> CardTactics:
    """Derive a deck-independent formation role from catalog knowledge."""
    roles = set(card.roles)
    description = f"{card.name_en} {card.description} {card.notes}".casefold()
    building_target = bool(
        "buildings" in card.targets
        or "building_target" in roles
        or any(phrase in description for phrase in BUILDING_TARGET_PHRASES)
    )
    win_condition = bool("win_condition" in roles or building_target)

    splash_strength = 0.0
    if "splash" in roles:
        splash_strength = 0.7
    if any(phrase in description for phrase in DIRECT_AREA_PHRASES):
        splash_strength = max(splash_strength, 1.0)
    if "stunning up to" in description or "chains to" in description:
        splash_strength = max(splash_strength, 0.65)
    if any(phrase in description for phrase in DEATH_EFFECT_PHRASES):
        splash_strength *= 0.25
    if card.kind == "spell" and "splash" in roles:
        splash_strength = max(splash_strength, 1.0)
    splash_strength = min(1.2, splash_strength)

    heavy_counter_strength = 0.0
    if "tank_killer" in roles:
        heavy_counter_strength = 1.0
    elif card.kind == "building":
        heavy_counter_strength = 0.9
    elif "swarm" in roles:
        heavy_counter_strength = 0.6
    elif "tank" in roles:
        heavy_counter_strength = 0.25

    cost = float(card.elixir or 3)
    offensive_commitment = 0.0
    if win_condition:
        offensive_commitment = 0.35 if cost <= 3.0 else min(1.0, 0.55 + 0.07 * cost)
    elif "tank" in roles and cost >= 6.0:
        offensive_commitment = 0.6

    defensive = bool(
        card.kind == "building"
        or "tank_killer" in roles
        or "swarm" in roles
        or splash_strength >= 0.3
    )

    if card.kind == "spell":
        return CardTactics(
            "spell",
            0.0,
            0.0,
            win_condition,
            defensive,
            round(splash_strength, 3),
            round(heavy_counter_strength, 3),
            round(offensive_commitment, 3),
        )
    if card.kind == "building":
        return CardTactics(
            "building",
            0.0,
            0.0,
            win_condition,
            True,
            round(splash_strength, 3),
            round(heavy_counter_strength, 3),
            round(offensive_commitment, 3),
        )

    frontline = 0.5
    backline = 0.0
    if "tank" in roles:
        frontline += 4.0
    if win_condition:
        frontline += 2.5
    if "tank_killer" in roles:
        frontline += 1.4
    if "swarm" in roles:
        frontline += 1.0
    if (card.elixir or 0) >= 6:
        frontline += 0.6

    if "ranged" in roles:
        backline += 4.0
    if "support" in roles:
        backline += 2.5
    if "spawner" in roles:
        backline += 2.0
    if "splash" in roles:
        backline += 0.8
    if "air" in roles and "tank" not in roles:
        backline += 0.8

    frontline += 1.2 * sum(phrase in description for phrase in FRONTLINE_PHRASES)
    backline += 1.4 * sum(phrase in description for phrase in BACKLINE_PHRASES)

    # Some cards contain their own front and rear ranks (for example a lead
    # fighter with ranged units behind). Treat descriptions stating both
    # halves explicitly as self-contained hybrid formations.
    mixed_formation = bool(
        "takes the lead" in description and "from behind" in description
    )
    if mixed_formation:
        frontline = max(frontline, 2.5)
        backline = max(backline, 2.5)
        formation_role = "hybrid"
    elif abs(frontline - backline) < 0.75:
        formation_role = "hybrid"
    elif frontline > backline:
        formation_role = "frontline"
    else:
        formation_role = "backline"
    return CardTactics(
        formation_role,
        round(frontline, 3),
        round(backline, 3),
        win_condition,
        defensive,
        round(splash_strength, 3),
        round(heavy_counter_strength, 3),
        round(offensive_commitment, 3),
    )
