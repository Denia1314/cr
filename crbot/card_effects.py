"""Executable spell patterns with per-field public-data provenance."""
from __future__ import annotations

import math

PATTERNS = {
    'arrows': 'waves', 'lightning': 'highest_hp', 'poison': 'periodic',
    'the_log': 'line', 'barbarian_barrel': 'line_spawn', 'royal_delivery': 'area_spawn',
    'goblin_barrel': 'spawn', 'graveyard': 'serial_spawn', 'rage': 'rage',
    'clone': 'clone', 'tornado': 'pull', 'earthquake': 'earthquake',
}


def spell_effect(kb, cid, level):
    pattern = PATTERNS.get(cid)
    card = kb.cards.get(cid, {})
    detail = card.get('detail_stats', {})
    if not pattern or not detail:
        return None
    row = next((r for r in detail['levelStats'] if r['level'] == level), None)
    if row is None:
        raise ValueError(f'{cid}: missing explicit level {level}')
    raw = card.get('spell') or {}
    effect = kb.effects.get({'the_log':'Log', 'earthquake':'Earthquake'}.get(cid, cid.title()), {})
    buff = effect.get('buff_data', {})
    damage = row.get('areaDamage', row.get('damage', 0))
    duration, period, count = float(detail.get('duration', 0)), 1., 1
    if pattern == 'periodic':
        damage, count = row['damagePerSecond'], max(1, round(duration))
    elif pattern == 'waves':
        count, period, damage = 3, .1, damage / 3
    elif pattern == 'highest_hp':
        count, damage = int(detail.get('count', 3)), row['damage']
    elif pattern == 'earthquake':
        damage, count = row['damagePerSecond'], max(1, round(duration))
    elif pattern == 'pull':
        count, period, damage = 4, duration / 4, row.get('damage', 0) / 4
    elif pattern in {'area_spawn','line_spawn'}:
        damage = row['damage']  # recruit attack is areaDamage in this source, not the impact
    tower_total = float(row.get('crownTowerDamage', 0))
    if pattern in {'periodic', 'waves', 'pull', 'earthquake'}:
        tower_total /= count
    return dict(pattern=pattern, damage=float(damage), radius=float(detail.get('radius', effect.get('radius', 0)/1000)),
        tower_multiplier=tower_total / damage if damage else 0, air=cid not in {'the_log','barbarian_barrel','earthquake'},
        delay=float(detail.get('deployTime', .5)), duration=duration, period=period, pulses=count,
        pushback=float(raw.get('pushback', 0))/1000, stun=float(detail.get('stunDuration', 0)),
        length=float(detail.get('range', 0)) if pattern.startswith('line') else 0,
        width=float(detail.get('width', 0)), speed=float(raw.get('speed', 200))/60,
        friendly=pattern in {'clone','rage'}, boost=float(str(detail.get('boost','+35%')).strip('+%'))/100,
        slow=float(buff.get('speed_multiplier', 0))/100 if pattern in {'periodic','earthquake'} else 0,
        building_multiplier=1+float(buff.get('building_damage_percent',0))/100,
        spawn={'goblin_barrel':'Goblin', 'graveyard':'Skeleton', 'barbarian_barrel':'Barbarian', 'royal_delivery':'DeliveryRecruit'}.get(cid),
        spawn_count=int(detail.get('count', 1)), spawn_period=float(detail.get('spawnSpeed', .5)))


def queue_effect(sim, state, spell, side, x, y):
    pattern = spell['pattern']
    if pattern == 'rage':
        for i in range(min(80,max(1,math.ceil(spell['duration']/.25)))):
            state.effects.append(dict(spell,at=state.time+spell['delay']+i*.25,side=side,x=x,y=y,
                                      duration=.25,damage=spell['damage'] if i==0 else 0))
        return
    for i in range(spell['pulses'] if pattern not in {'highest_hp','serial_spawn'} else 1):
        state.effects.append(dict(spell, at=state.time+spell['delay']+i*spell['period'], side=side, x=x, y=y))


def resolve_effects(sim, state):
    events, state.effects = state.effects, []
    for event in events:
        if event['at'] > state.time:
            state.effects.append(event)
            continue
        side, x, y, pattern = event['side'], event['x'], event['y'], event['pattern']
        if pattern == 'spawn_entity':
            spec=sim.kb.unit(event['spawn'],sim.level)
            if spec is not None and len(state.entities)<96:
                sim.add(state,spec,side,x,y,deployed=True,value=event.get('summon_value',0))
            continue
        targets = [e for e in state.entities if e.hp > 0 and
                   (e.side == side if event['friendly'] else e.side != side) and
                   (event['air'] or not e.spec.air)]
        if pattern.startswith('line'):
            # Schedule each victim by travel distance; a line never becomes circular splash.
            for enemy in targets:
                if pattern in {'spawn','serial_spawn'}:
                    continue
                forward = (y-enemy.y)*side
                if -enemy.spec.radius <= forward <= event['length']+enemy.spec.radius and abs(enemy.x-x) <= event['width']/2+enemy.spec.radius:
                    state.impacts.append((state.time+max(0,forward)/max(.1,event['speed']),side,enemy.uid,
                        enemy.x,enemy.y,event['damage']*(event['tower_multiplier'] if enemy.tower else 1),0,False,1.,0.,event['pushback']))
        else:
            targets = [e for e in targets if math.hypot(e.x-x,e.y-y) <= event['radius']+e.spec.radius]
            if pattern == 'highest_hp':
                targets = sorted(targets,key=lambda e:(-e.hp,e.uid))[:event['pulses']]
            for enemy in targets:
                if pattern == 'clone':
                    if not enemy.tower and not enemy.spec.building and not enemy.cloned and len(state.entities)<96:
                        child=sim.add(state,enemy.spec,side,enemy.x+.3,enemy.y,hp_fraction=1/enemy.spec.hp)
                        child.cloned=True
                        child.shield=1 if enemy.shield else 0
                    continue
                if pattern == 'rage' and enemy.side == side:
                    enemy.haste_until=max(enemy.haste_until,state.time+event['duration'])
                    enemy.haste=1+event['boost']
                    continue
                multiplier = event['tower_multiplier'] if enemy.tower else (event['building_multiplier'] if enemy.spec.building else 1)
                sim.hit(state,enemy,event['damage']*multiplier)
                if event['stun']:
                    enemy.stunned_until=max(enemy.stunned_until,state.time+event['stun'])
                    enemy.walked=0
                    enemy.locked_at=state.time+event['stun']
                if not enemy.tower and not enemy.spec.building:
                    if event['slow']:
                        enemy.slow_until=max(enemy.slow_until,state.time+event['period'])
                        enemy.slow=max(.1,1+event['slow'])
                    if pattern == 'pull':
                        # Force calibration remains approximate and is reported by the profile.
                        enemy.x+=(x-enemy.x)*.25
                        enemy.y+=(y-enemy.y)*.25
        if event['spawn']:
            spec=sim.kb.unit(event['spawn'],sim.level)
            if spec is not None:
                count=event['spawn_count']
                if pattern == 'line_spawn':
                    y-=side*event['length']
                for i in range(min(20,count)):
                    delay=i*event['spawn_period'] if pattern=='serial_spawn' else (event['length']/max(.1,event['speed']) if pattern=='line_spawn' else 0)
                    state.effects.append(dict(event,pattern='spawn_entity',at=state.time+delay,
                        x=x+((i%3)-1)*.4,y=y+(i//3)*.3))
        if pattern == 'rage':
            # The same cast damages enemies once and accelerates allies for its duration.
            for enemy in state.entities:
                if enemy.side != side and math.hypot(enemy.x-x,enemy.y-y)<=event['radius']+enemy.spec.radius:
                    sim.hit(state,enemy,event['damage']*(event['tower_multiplier'] if enemy.tower else 1))


def mechanism_profile(kb, cid):
    card=kb.cards[cid]
    effects=[]
    summons=card.get('summons',[]) or ([dict(unit='ElixirCollector',count=1)] if cid=='elixir_collector' else [])
    for summon in summons:
        raw=kb.units.get(summon['unit'],{})
        fields={k:v for k,v in raw.items() if v and any(word in k for word in
            ('spawn','buff','damage','heal','shield','charge','jump','reflect','ability','invis','projectile','mana'))}
        projectile=kb.projectiles.get(raw.get('projectile'),{})
        refs={v for k,v in {**raw,**projectile}.items() if isinstance(v,str) and ('buff' in k or 'area_effect' in k)}
        linked={v:kb.buffs.get(v,kb.effects.get(v)) for v in refs if v in kb.buffs or v in kb.effects}
        effects.append(dict(unit=summon['unit'],count=summon['count'],parameters=fields,
                            projectile=projectile,linked_effects=linked))
    unresolved=[v for v in card.get('unsupported',[]) if not (cid in PATTERNS and v=='spell_pattern')]
    return dict(card_id=cid, pattern=PATTERNS.get(cid,'unit' if summons or cid=='berserker' else 'unmapped'),
        unit_effects=effects, spell_parameters=card.get('spell'), detail_parameters=card.get('detail_stats'),
        card_parameters=card.get('source_fields',{}), unresolved=unresolved,
        source=card.get('metadata_source',kb.source), balance_validated=False,
        approximation_notes=['Movement, wave timing and spawn spread are approximate; active abilities and variants require separate validation.'])
