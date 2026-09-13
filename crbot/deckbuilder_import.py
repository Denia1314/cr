"""Import factual directory metadata, never execute downloaded JavaScript."""
from __future__ import annotations

import copy
import hashlib
import json
import re

SOURCE_URL = "https://www.clashroyaledeckbuilder.net/zh/clash-royale-cards"
ALIASES = {"p_e_k_k_a": "pekka", "mini_p_e_k_k_a": "mini_pekka"}


def parse_details(script: str) -> dict:
    """Decode the static detail object with a literal-only tokenizer, not eval."""
    start = script.index('let m={') + len('let m=')
    tokens, depth, pos = [], 0, start
    lexer = re.compile(r'\s*("(?:[^"\\]|\\.)*"|[a-zA-Z_$][\w$]*|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[{}\[\],:])')
    while pos < len(script):
        match = lexer.match(script, pos)
        if not match:
            raise ValueError('Non-literal detail database')
        token = match.group(1)
        tokens.append(token)
        pos = match.end()
        if token in {'{', '['}:
            depth += 1
        elif token in {'}', ']'}:
            depth -= 1
            if depth == 0:
                break
    converted = []
    for i, token in enumerate(tokens):
        if re.fullmatch(r'[a-zA-Z_$][\w$]*', token):
            if i + 1 >= len(tokens) or tokens[i + 1] != ':':
                raise ValueError('Non-literal detail value')
            token = json.dumps(token)
        elif token.startswith('.'):
            token = '0' + token
        converted.append(token)
    raw = json.loads(''.join(converted))
    result = {}
    for key, detail in raw.items():
        if key != detail.get('cardId'):
            raise ValueError('Detail identity mismatch')
        levels = detail.get('levelStats', [])
        if len({r['level'] for r in levels}) != len(levels):
            raise ValueError('Duplicate detail level')
        for row in levels:
            if not 1 <= row['level'] <= 16 or any(not isinstance(v, (int, float)) or v < 0 for v in row.values()):
                raise ValueError('Invalid level statistics')
        result[ALIASES.get(key, key)] = detail
    return result


def parse_database(script: str) -> list[dict]:
    # Only literal arguments of the site's card constructor are accepted.
    string = r'"(?:[^"\\]|\\.)*"'
    token = rf'(?:{string}|\[(?:{string}(?:,{string})*)?\]|\d+|!0|!1|void 0)'
    pattern = rf'\bs\(({token}(?:,{token}){{10}})\)'
    rows = []
    for match in re.finditer(pattern, script):
        values = []
        for item in re.findall(token, match.group(1)):
            values.append({"!0": True, "!1": False, "void 0": None}.get(item)
                          if item in {"!0", "!1", "void 0"} else json.loads(item))
        cid, name, cost, rarity, kind, arena, icon, tags, _, evolution, champion = values
        if not re.fullmatch(r"[a-z][a-z0-9_]*", cid):
            raise ValueError("Invalid source card ID")
        if rarity not in {"common", "rare", "epic", "legendary", "champion"} or kind not in {"troop", "spell", "building"}:
            raise ValueError("Invalid source category")
        if not 0 <= cost <= 10 or not icon.startswith("https://api-assets.clashroyale.com/cards/"):
            raise ValueError("Invalid source cost or icon URL")
        rows.append(dict(source_id=cid, id=ALIASES.get(cid, cid), name_en=name,
                         elixir=cost, rarity=rarity, kind=kind, arena=arena,
                         icon_url=icon, tags=tags, evolution=bool(evolution), champion=bool(champion)))
    if len(rows) != 121 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f"Expected 121 unique cards, got {len(rows)}; inspect changed source before importing")
    return rows


def merge_directory(catalog: dict, knowledge: dict, snapshot: dict) -> tuple[dict, dict]:
    catalog, knowledge = copy.deepcopy(catalog), copy.deepcopy(knowledge)
    by_id = {c['id']: c for c in catalog['cards']}
    rows = snapshot['cards']
    if len(rows) != 121 or len({r['id'] for r in rows}) != 121:
        raise ValueError("Incomplete or duplicate directory")
    for card in by_id.values():
        card['deckbuilder_current'] = False
    for row in rows:
        cid = row['id']
        if cid not in by_id:
            by_id[cid] = dict(id=cid, official_id=None, name_en=row['name_en'], name_zh='',
                             icon_url=row['icon_url'], icon_path='', icon_variants=[],
                             kind=row['kind'], targets=[], roles=[], counters=[], synergies=[],
                             strategy_verified=False, notes='Public detail statistics imported; combat mechanics unverified.')
            catalog['cards'].append(by_id[cid])
        card = by_id[cid]
        metadata = {k: v for k, v in row.items() if k != 'battle_stats'}
        # Keep stable identities, known mechanics, local icons and curated strategies.
        card.update(elixir=None if cid == 'mirror' else row['elixir'], rarity=row['rarity'],
                    deckbuilder_current=True, directory_metadata=metadata)
        if cid not in knowledge['cards']:
            knowledge['cards'][cid] = dict(id=cid, official_id=card.get('official_id'),
                name=card['name_en'], kind=card['kind'], roles=[], summons=[], spell=None,
                variants=[], unsupported=['unmapped_combat_mechanics'], source_fields={})
        entry = knowledge['cards'][cid]
        if 'missing_combat_parameters' in entry.get('unsupported', []):
            entry['unsupported'] = ['unmapped_combat_mechanics' if v == 'missing_combat_parameters' else v for v in entry['unsupported']]
            card['notes'] = 'Public detail statistics imported; combat mechanics unverified.'
        entry['elixir'] = card['elixir']
        entry['directory_metadata'] = metadata
        entry['metadata_source'] = snapshot['source']
        detail = row.get('battle_stats')
        if detail is not None:
            entry['detail_stats'] = detail
            # A mixed roster has separate entities: one table cannot describe all of them.
            summons = entry.get('summons', [])
            if len(summons) == 1 and card['kind'] != 'spell' and detail.get('levelStats'):
                original = knowledge['units'][summons[0]['unit']]
                base_name = original.get('_base_unit_name', summons[0]['unit'])
                original = knowledge['units'][base_name]
                levels = detail['levelStats']
                if all('hitpoints' in r for r in levels):
                    unit = copy.deepcopy(original)
                    name = 'deckbuilder:' + cid
                    unit['name'] = name
                    unit['_base_unit_name'] = base_name
                    unit['_detail_levels'] = {str(r['level']): r for r in levels}
                    unit['_detail_source'] = snapshot['source']
                    for source, target, scale in [('hitSpeed', 'hit_speed', 1000),
                            ('firstHitSpeed', 'load_time', 1000), ('deployTime', 'deploy_time', 1000),
                            ('splashRadius', 'area_damage_radius', 1000)]:
                        if source in detail:
                            unit[target] = detail[source] * scale
                    reach = re.search(r'\(([\d.]+)\)|^([\d.]+)$', str(detail.get('range', '')))
                    if reach:
                        unit['range'] = float(reach.group(1) or reach.group(2)) * 1000
                    if detail.get('speed') in {'Slow', 'Medium', 'Fast', 'Very Fast'}:
                        unit['speed'] = {'Slow': 45, 'Medium': 60, 'Fast': 90, 'Very Fast': 120}[detail['speed']]
                    if 'targets' in detail:
                        unit['attacks_air'] = 'Air' in detail['targets']
                        unit['attacks_ground'] = 'Ground' in detail['targets'] or 'Buildings' in detail['targets']
                        unit['target_only_buildings'] = detail['targets'] == 'Buildings'
                    knowledge['units'][name] = unit
                    entry['summons'] = [dict(unit=name, count=detail.get('count', summons[0]['count']))]
            if entry.get('spell') and detail.get('levelStats') and card['kind'] == 'spell':
                entry['spell']['_detail_levels'] = {str(r['level']): r for r in detail['levelStats']}
                if 'radius' in detail:
                    entry['spell']['radius'] = detail['radius'] * 1000
    catalog['directory_source'] = snapshot['source']
    knowledge['directory_source'] = snapshot['source']
    knowledge['version'] = 'directory-' + hashlib.sha256(json.dumps(
        [knowledge['cards'], knowledge['units'], knowledge['source']], sort_keys=True).encode()).hexdigest()[:16]
    return catalog, knowledge
