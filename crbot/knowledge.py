"""Versioned public card mechanics; missing and historical data stay explicit."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UnitSpec:
    name: str
    hp: float
    damage: float
    period: float
    first_hit: float
    speed: float
    reach: float
    sight: float
    radius: float
    air: bool
    targets: tuple[str, ...]
    building_only: bool
    building: bool
    splash: float
    projectile_speed: float
    deploy: float
    lifetime: float
    shield: float
    death_damage: float
    death_radius: float
    death_spawn: str | None
    death_count: int
    spawn: str | None
    spawn_count: int
    spawn_period: float
    spawn_start: float
    unsupported: tuple[str, ...]
    charge_distance: float = 0
    charge_multiplier: float = 1
    charge_damage: float = 0
    ramp_times: tuple[float, float] = (0, 0)
    ramp_damage: tuple[float, float] = (0, 0)
    stun: float = 0
    pushback: float = 0
    resource_period: float = 0
    resource_amount: float = 0
    resource_on_death: float = 0
    dash_damage: float = 0
    dash_min: float = 0
    dash_max: float = 0
    dash_windup: float = 0
    dash_travel: float = 0
    dash_speed: float = 0
    dash_recovery: float = 0
    dash_radius: float = 0
    dash_pushback: float = 0
    dash_immune: bool = False
    minimum_range: float = 0
    ignore_pushback: bool = False
    attack_targets: int = 1
    reflected_damage: float = 0
    reflected_radius: float = 0
    reflected_stun: float = 0
    chain_count: int = 1
    chain_radius: float = 0


class KnowledgeBase:
    def __init__(self, payload: dict[str, Any]):
        if payload.get("schema_version") != 1:
            raise ValueError("不支持的战斗知识库 schema")
        self.payload = payload
        self.version = str(payload["version"])
        self.source = dict(payload["source"])
        self.cards = dict(payload["cards"])
        self.units = dict(payload["units"])
        self.projectiles = dict(payload.get("projectiles", {}))
        self.effects = dict(payload.get("effects", {}))
        self.buffs = dict(payload.get("buffs", {}))
        self.relative_levels = dict(payload["relative_levels"])
        self._cache: dict[tuple[str, int], UnitSpec | None] = {}
        for cid, card in self.cards.items():
            if cid != card["id"]:
                raise ValueError("知识库 ID 不一致")
            cost = card.get("elixir")
            if cost is not None and (not isinstance(cost, (int, float)) or not 0 <= cost <= 10):
                raise ValueError(f"无效费用 {cid}")
        for name, raw in self.units.items():
            for key in ("hitpoints", "damage", "hit_speed", "range", "speed"):
                value = raw.get(key)
                if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                    raise ValueError(f"无效单位数值 {name}.{key}")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def value(self, raw: dict, key: str, level: int) -> float:
        if raw.get('_detail_levels'):
            row = raw['_detail_levels'].get(str(level))
            if row is None:
                raise ValueError(f"详情数值等级超出来源范围: {raw.get('name')} level={level}")
            field = {'hitpoints': 'hitpoints', 'damage': 'damage', 'shield_hitpoints': 'shieldHitpoints',
                     'death_damage': 'deathDamage'}.get(key)
            if field in row:
                return float(row[field])
            if key == 'damage' and 'areaDamage' in row:
                return float(row['areaDamage'])
        values = raw.get(key + "_per_level")
        offset = int(self.relative_levels.get(str(raw.get("rarity", "Common")), 0))
        index = level - 1 - offset
        if values:
            if not 0 <= index < len(values):
                raise ValueError(f"数值等级超出来源范围: {raw.get('name')} level={level}")
            return float(values[index])
        # A nonzero unscaled stat is not invented at a different level.
        return float(raw.get(key) or 0)

    def unit(self, name: str, level: int = 11) -> UnitSpec | None:
        key = (name, level)
        if key in self._cache:
            return self._cache[key]
        raw = self.units.get(name)
        if not raw or not raw.get("hitpoints_per_level"):
            self._cache[key] = None
            return None
        projectile = self.projectiles.get(raw.get("projectile"), {})
        damage = self.value(raw, "damage", level)
        if not damage and projectile:
            damage = self.value(projectile, "damage", level)
        hp = self.value(raw, "hitpoints", level)
        scale = hp / max(1, float(raw.get("hitpoints") or hp))
        unsupported = []
        # These mechanisms are retained in source, but are not silently simulated as ordinary attacks.
        for field in ("jump_speed", "attached_character", "minimum_range",
                      "buff_on_damage", "starting_buff", "projectile_special", "damage_special",
                      "hero_ability", "death_area_effect_object", "heal_per_second",
                      "reflect_damage", "spawn_area_effect_object"):
            if raw.get(field):
                unsupported.append(field)
        if raw.get("charge_range") and "damage_special" in unsupported:
            unsupported.remove("damage_special")
        dash = bool(raw.get('dash_damage') and raw.get('dash_min_range') and raw.get('dash_max_range')
                    and int(raw.get('dash_count') or 0) <= 1)
        if dash and 'jump_speed' in unsupported:
            unsupported.remove('jump_speed')
        if 'minimum_range' in unsupported:
            unsupported.remove('minimum_range')
        for field in ('multiple_projectiles', 'projectile_special', 'attached_character',
                      'kamikaze', 'hides_when_not_attacking', 'morph_character', 'convert_on_kill',
                      'area_effect_on_hit', 'death_spawn_projectile', 'spawn_character2'):
            if raw.get(field) and field not in unsupported:
                unsupported.append(field)
        if raw.get('dash_damage') and not dash:
            unsupported.append('dash_chain_or_ability')
        for field in ('projectile_range','spawn_projectile','pingpong_visual_time'):
            if projectile.get(field):unsupported.append('projectile:'+field)
        if raw.get('hide_time_ms'):unsupported.append('invisibility')
        if raw.get('death_spawn_character') and not self.units.get(raw['death_spawn_character'],{}).get('hitpoints_per_level'):
            unsupported.append('death_spawn_missing_definition')
        buff_key = projectile.get("buff_on_damage") or raw.get("buff_on_damage")
        buff = self.buffs.get(buff_key, {})
        stun_time = float(projectile.get("buff_on_damage_time") or raw.get("buff_on_damage_time") or 0) / 1000
        spec = UnitSpec(
            name=name, hp=hp, damage=damage,
            period=max(.1, float(raw.get("hit_speed") or 1000) / 1000),
            first_hit=float(raw.get("load_time") or 0) / 1000,
            speed=float(raw.get("speed") or 0) / 60,  # approximate spatial calibration, reported in audit
            reach=float(raw.get("range") or 0) / 1000,
            sight=float(raw.get("sight_range") or 5500) / 1000,
            radius=float(raw.get("collision_radius") or 300) / 1000,
            air=bool(raw.get("flying_height")),
            targets=tuple(x for x, field in (("ground", "attacks_ground"), ("air", "attacks_air")) if raw.get(field)),
            building_only=bool(raw.get("target_only_buildings") or raw.get("target_only_towers")),
            building=bool(raw.get("_building")),
            splash=float(raw.get("area_damage_radius") or projectile.get("radius") or 0) / 1000,
            projectile_speed=float(projectile.get("speed") or 0) / 60,
            deploy=float(raw.get("deploy_time") or 0) / 1000,
            lifetime=float(raw.get("life_time") or 0) / 1000,
            shield=self.value(raw, 'shield_hitpoints', level) if 'shieldHitpoints' in raw.get('_detail_levels', {}).get(str(level), {}) else float(raw.get("shield_hitpoints") or 0) * scale,
            death_damage=self.value(raw, 'death_damage', level) if 'deathDamage' in raw.get('_detail_levels', {}).get(str(level), {}) else float(raw.get("death_damage") or 0) * scale,
            death_radius=float(raw.get("death_damage_radius") or 0) / 1000,
            death_spawn=raw.get("death_spawn_character"), death_count=int(raw.get("death_spawn_count") or 0),
            spawn=raw.get("spawn_character"), spawn_count=int(raw.get("spawn_number") or 0),
            spawn_period=float(raw.get("spawn_pause_time") or 0) / 1000,
            spawn_start=float(raw.get("spawn_start_time") or 0) / 1000,
            unsupported=tuple(unsupported),
            charge_distance=float(raw.get("charge_range") or 0) / 100,
            charge_multiplier=float(raw.get("charge_speed_multiplier") or 100) / 100,
            charge_damage=self.special_value(raw,'damage_special',level),
            ramp_times=(float(raw.get("variable_damage_time1") or 0) / 1000, float(raw.get("variable_damage_time2") or 0) / 1000),
            ramp_damage=(float(raw.get("variable_damage2") or 0) * scale, float(raw.get("variable_damage3") or 0) * scale),
            stun=stun_time if buff.get("hit_speed_multiplier") == -100 else 0,
            pushback=float(projectile.get("pushback") or raw.get("attack_push_back") or 0) / 1000,
            resource_period=float(raw.get('mana_generate_time_ms') or 0)/1000,
            resource_amount=float(raw.get('mana_collect_amount') or 0),
            resource_on_death=float(raw.get('mana_on_death') or 0),
            dash_damage=self.special_value(raw,'dash_damage',level) if dash else 0,
            dash_min=float(raw.get('dash_min_range') or 0)/1000,
            dash_max=float(raw.get('dash_max_range') or 0)/1000,
            dash_windup=float(raw.get('dash_cooldown') or 0)/1000,
            dash_travel=float(raw.get('dash_constant_time') or 0)/1000,
            dash_speed=float(raw.get('jump_speed') or 0)/60,
            dash_recovery=float(raw.get('dash_landing_time') or 0)/1000,
            dash_radius=float(raw.get('dash_radius') or 0)/1000,
            dash_pushback=float(raw.get('dash_push_back') or 0)/1000,
            dash_immune=bool(raw.get('dash_immune_to_damage_time')),
            minimum_range=float(raw.get('minimum_range') or 0)/1000,
            ignore_pushback=bool(raw.get('ignore_pushback')),
            attack_targets=max(1,int(raw.get('multiple_targets') or 1)),
            reflected_damage=self.special_value(raw,'reflected_attack_damage',level),
            reflected_radius=float(raw.get('reflected_attack_radius') or 0)/1000,
            reflected_stun=float(raw.get('reflected_attack_buff_duration') or 0)/1000
                if self.buffs.get(raw.get('reflected_attack_buff'),{}).get('hit_speed_multiplier') == -100 else 0,
            chain_count=max(1,int(projectile.get('chained_hit_count') or 1)),
            chain_radius=float(projectile.get('chained_hit_radius') or 0)/1000,
        )
        self._cache[key] = spec
        return spec

    def special_value(self, raw, key, level):
        """Prefer explicit levels; otherwise retain the source's level scaling estimate."""
        detail_key={'dash_damage':'jumpDamage','damage_special':'chargeDamage'}.get(key)
        row=raw.get('_detail_levels',{}).get(str(level),{})
        if detail_key in row:return float(row[detail_key])
        if raw.get(key+'_per_level'):return self.value(raw,key,level)
        base=self.units.get(raw.get('_base_unit_name'),raw)
        scale=self.value(base,'hitpoints',level)/max(1,float(base.get('hitpoints') or 1))
        return round(float(raw.get(key) or 0)*scale)

    def roster(self, card_id: str, level: int = 11) -> list[tuple[UnitSpec, int]]:
        card = self.cards.get(card_id, {})
        if card_id == 'elixir_collector':
            unit=self.unit('ElixirCollector',level)
            return [(unit,1)] if unit else []
        if card_id == 'berserker' and card.get('detail_stats'):
            detail=card['detail_stats']
            row=next((r for r in detail['levelStats'] if r['level']==level),None)
            if row is None:
                raise ValueError('berserker: missing explicit level')
            base=self.unit('Knight',level)
            if base is None:
                return []
            return [(replace(base,name='detail:berserker',hp=row['hitpoints'],damage=row['damage'],
                period=detail['hitSpeed'],first_hit=detail['firstHitSpeed'],speed=1.5,reach=self.unit('Skeleton',level).reach,
                deploy=detail['deployTime'],unsupported=('detail_base_attack_unvalidated_balance',)),1)]
        result = []
        for entry in card.get("summons", []):
            unit = self.unit(entry["unit"], level)
            if unit is None:
                return []
            result.append((unit, int(entry["count"])))
        return result

    def spell(self, card_id: str, level: int = 11) -> dict | None:
        from .card_effects import spell_effect
        effect = spell_effect(self, card_id, level)
        if effect is not None:
            return effect
        if "spell_pattern" in self.cards.get(card_id, {}).get("unsupported", []):
            return None  # do not model a line/limited-target/multiwave spell as a circle
        raw = self.cards.get(card_id, {}).get("spell")
        if not raw or not raw.get("damage_per_level"):
            return None
        result = {"damage": self.value(raw, "damage", level),
                "radius": float(raw.get("radius") or 0) / 1000,
                "tower_multiplier": max(0, 1 + float(raw.get("crown_tower_damage_percent") or 0) / 100),
                "air": bool(raw.get("aoe_to_air", raw.get("hits_air", True))),
                "delay": float(raw.get("_delay_s", .5)),
                "duration": float(raw.get("life_duration") or 0) / 1000 if raw.get("hit_speed") else 0,
                "period": max(.1, float(raw.get("hit_speed") or 1000) / 1000),
                "pushback": float(raw.get("pushback") or 0) / 1000,
                "stun": float(raw.get("buff_time") or 0) / 1000 if (raw.get("buff_data") or {}).get("hit_speed_multiplier") == -100 else 0}
        detail = raw.get('_detail_levels', {}).get(str(level), {})
        if 'crownTowerDamage' in detail and result['damage'] > 0:
            result['tower_multiplier'] = detail['crownTowerDamage'] / result['damage']
        return result

    def audit(self, level: int = 11) -> dict:
        from .card_effects import PATTERNS
        from .unit_mechanics import mechanism_status
        supported, missing, partial, numeric = [], [], [], []
        mechanism_counts={}
        for cid, card in self.cards.items():
            try:
                roster = self.roster(cid, level)
                spell = self.spell(cid, level)
                if roster or (card.get("spell") or {}).get("damage_per_level") or any(
                        len(row) > 1 for row in card.get('detail_stats', {}).get('levelStats', [])):
                    numeric.append(cid)
                if roster or spell:
                    supported.append(cid)
                    coverage=mechanism_status(self,cid,level)
                    for mechanism in coverage['implemented']:
                        mechanism_counts[mechanism]=mechanism_counts.get(mechanism,0)+1
                    if coverage['pending']:
                        partial.append(cid)
                else:
                    missing.append(cid)
            except ValueError:
                missing.append(cid)
        return {"knowledge_version": self.version, "source": self.source,
                "effect_record_cards": len(self.cards), "executable_spell_patterns": dict(PATTERNS),
                "executable_unit_mechanism_counts":mechanism_counts,
                "directory_source": self.payload.get('directory_source'),
                "detail_cards": sum('detail_stats' in c for c in self.cards.values()),
                "detail_level_rows": sum(len(c.get('detail_stats', {}).get('levelStats', [])) for c in self.cards.values()),
                "catalog_cards": len(self.cards), "numeric_cards": len(numeric), "simulatable_cards": len(supported),
                "missing_numeric_cards": [cid for cid in self.cards if cid not in numeric],
                "unsupported_simulation_cards": missing, "partial_mechanism_cards": partial,
                "variant_count": sum(len(c.get("variants", [])) for c in self.cards.values()),
                "validated_current_balance": False, "battle_acceptance": False,
                "limitations": ["公开详情数值与历史机制混合来源，非当前平衡核验", "空间/速度近似，血量与等级为观测或配置假设",
                                "特殊机制与变体未核验时降低可信度，不视为完整模拟"]}


def build_knowledge(catalog: dict, raw: dict, rarities: list, source: dict) -> dict:
    """Deterministic import; latest catalog identity/cost never replaced by historic stats."""
    units = {v["name"]: dict(v, _building=(kind == "building"))
             for kind in ("characters", "building") for v in raw.get(kind, []) if v.get("hitpoints")}
    projectiles = {v["name"]: v for v in raw.get("projectile", [])}
    entries = {v.get("key", "").replace("-", "_"): v
               for kind in ("troop", "building", "spell") for v in raw.get(kind, []) if v.get("key")}
    # Explicit source-name mapping for projectile-only cards absent from troop/spell tables.
    projectile_cards = {"fireball": "FireballSpell", "arrows": "ArrowsSpell", "rocket": "RocketSpell",
                        "giant_snowball": "SnowballSpell", "goblin_barrel": "GoblinBarrelSpell",
                        "the_log": "LogProjectileRolling", "barbarian_barrel": "BarbLogProjectileRolling",
                        "royal_delivery": "RoyalDeliveryProjectile"}
    cards = {}
    for c in catalog["cards"]:
        cid = c["id"]
        row = entries.get(cid, {}) or units.get(c.get("sc_key"), {})
        summons = []
        for key, count in (("summon_character", "summon_number"), ("summon_character_second", "summon_character_second_count")):
            if row.get(key) in units:
                summons.append({"unit": row[key], "count": max(1, int(row.get(count) or 1))})
        if not summons and row.get("name") in units and c.get("kind") != "spell":
            summons.append({"unit": row["name"], "count": 1})
        spell = None
        if c.get("kind") == "spell":
            candidate = projectiles.get(projectile_cards.get(cid), {}) or row
            if not candidate.get("damage_per_level") and row.get("projectile"):
                candidate = {**projectiles.get(row["projectile"], {}), **{k: row[k] for k in ("radius",) if row.get(k)}}
            spell = candidate or None
        cards[cid] = {"id": cid, "official_id": c.get("official_id"),
                      "name": c.get("name_zh") or c.get("name_en"), "kind": c.get("kind"),
                      "elixir": c.get("elixir"), "roles": c.get("roles", []),
                      "summons": summons, "spell": spell,
                      "variants": [{"asset": x, "mechanics_status": "unknown"} for x in c.get("icon_variants", [])],
                      "unsupported": [k for k in ("hero_ability", "passive_ability", "mirror", "area_effect_object") if row.get(k)],
                      "source_fields": row}
        if spell:
            cards[cid]["unsupported"].extend(k for k in ("projectile_range", "spawn_character", "target_buff") if spell.get(k))
        if cid in {"arrows", "lightning", "the_log", "barbarian_barrel", "royal_delivery"}:
            cards[cid]["unsupported"].append("spell_pattern")
    digest = hashlib.sha256(json.dumps([cards, units, raw, rarities], sort_keys=True).encode()).hexdigest()
    return {"schema_version": 1, "version": "public-" + digest[:16], "source": source,
            "relative_levels": {v["name"]: v["relative_level"] for v in rarities},
            "cards": cards, "units": units, "projectiles": projectiles,
            "effects": {v["name"]: v for v in raw.get("spell", [])},
            "buffs": {v["name"]: v for v in raw.get("character_buff", [])}}
