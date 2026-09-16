"""Database-driven special attacks; events are simulation hypotheses, not vision."""
from __future__ import annotations

import math


def pulse(kb, raw, level):
    if not raw or not raw.get('radius'):
        return None
    buff=raw.get('buff_data') or kb.buffs.get(raw.get('buff') or raw.get('buff_on_damage'),{})
    duration=float(raw.get('buff_time') or raw.get('buff_on_damage_time') or 0)/1000
    return dict(pattern='unit_pulse',damage=kb.value(raw,'damage',level),radius=float(raw['radius'])/1000,
                air=bool(raw.get('hits_air',raw.get('aoe_to_air',False))),
                ground=bool(raw.get('hits_ground',raw.get('aoe_to_ground',True))),
                tower_multiplier=1+float(raw.get('crown_tower_damage_percent') or 0)/100,
                stun=duration if buff.get('hit_speed_multiplier') == -100 else 0,
                slow=max(0.,1+float(buff.get('speed_multiplier') or 0)/100),
                attack_slow=max(0.,1+float(buff.get('hit_speed_multiplier') or 0)/100),
                duration=duration,pushback=float(raw.get('pushback') or 0)/1000)


def deployment_pulses(kb, cid, level):
    card=kb.cards.get(cid,{})
    if not card.get('summons'):return []
    source=card.get('source_fields',{})
    raw=source.get('projectile_data') or kb.projectiles.get(source.get('projectile'))
    event=pulse(kb,raw,level)
    if event is None:return []
    delay=max((u.deploy for u,n in kb.roster(cid,level)),default=0)
    return [dict(event,delay=delay,mechanism='card_deployment_projectile')]


def queue_spawn_pulse(sim,state,entity):
    raw=sim.kb.units.get(entity.spec.name,{})
    event=pulse(sim.kb,sim.kb.effects.get(raw.get('spawn_area_object')),sim.level)
    if event:
        state.effects.append(dict(event,at=entity.active_at,side=entity.side,x=entity.x,y=entity.y,
                                  owner=entity.uid,mechanism='unit_spawn_area'))


def control(state,entity,*,stun=0,pushback=0,x=None,y=None,slow=1,attack_slow=1,duration=0):
    if stun:
        entity.stunned_until=max(entity.stunned_until,state.time+stun)
        entity.walked=0
        entity.locked_at=state.time+stun
        entity.winding_target=None
        if entity.dash_phase=='windup':entity.dash_phase=''
    if duration and (slow<1 or attack_slow<1):
        entity.slow_until=max(entity.slow_until,state.time+duration)
        entity.slow=max(.1,slow)
        entity.attack_slow=max(.1,attack_slow)
    if pushback and not (entity.spec.ignore_pushback or entity.spec.building or entity.tower):
        dx,dy=entity.x-(x if x is not None else entity.x),entity.y-(y if y is not None else entity.y)
        distance=math.hypot(dx,dy)
        if distance:
            entity.x=max(0,min(18,entity.x+dx/distance*pushback))
            entity.y=max(0,min(32,entity.y+dy/distance*pushback))
        entity.walked=0
        entity.winding_target=None
        if entity.dash_phase=='windup':entity.dash_phase=''


def resolve_pulse(sim,state,event):
    owner=event.get('owner')
    if owner is not None and not any(e.uid==owner and e.hp>0 for e in state.entities):return
    for enemy in state.entities:
        if enemy.side==event['side'] or enemy.hp<=0:continue
        if not event['air'] and enemy.spec.air:continue
        if not event.get('ground',True) and not enemy.spec.air:continue
        if math.hypot(enemy.x-event['x'],enemy.y-event['y'])>event['radius']+enemy.spec.radius:continue
        if not sim.hit(state,enemy,event['damage']*(event['tower_multiplier'] if enemy.tower else 1)):continue
        control(state,enemy,stun=event['stun'],pushback=event['pushback'],x=event['x'],y=event['y'],
                slow=event['slow'],attack_slow=event['attack_slow'],duration=event['duration'])


def advance_dash(sim,state,entity,target):
    """Bounded windup/travel/landing state machine, using source distances/timing."""
    spec=entity.spec
    if not spec.dash_damage:return False
    if entity.dash_phase=='windup':
        victim=next((e for e in state.entities if e.uid==entity.dash_target and e.hp>0),None)
        if victim is None or sim.distance(entity,victim)>spec.dash_max:
            entity.dash_phase='';return False
        if state.time<entity.dash_at:return True
        entity.dash_x,entity.dash_y=victim.x,victim.y
        travel=spec.dash_travel or math.hypot(victim.x-entity.x,victim.y-entity.y)/max(.1,spec.dash_speed)
        entity.dash_at=state.time+travel
        entity.dash_phase='flight'
        return True
    if entity.dash_phase=='flight':
        if state.time<entity.dash_at:return True
        entity.x,entity.y=entity.dash_x,entity.dash_y
        entity.dash_phase=''
        entity.walked=0
        entity.ready_at=state.time+spec.dash_recovery+spec.first_hit
        entity.dash_ready_at=entity.ready_at
        entity.winding_target=None
        state.impacts.append((state.time,entity.side,entity.dash_target,entity.x,entity.y,
                              spec.dash_damage,spec.dash_radius,'air' in spec.targets,1.,0.,spec.dash_pushback,entity.uid))
        return True
    if target and state.time>=max(entity.ready_at,entity.dash_ready_at) and spec.dash_min<=sim.distance(entity,target)<=spec.dash_max:
        entity.dash_phase='windup';entity.dash_target=target.uid
        entity.dash_at=state.time+spec.dash_windup
        entity.winding_target=None
        state.uncertainties.add('dash_timing_and_level_scaling_estimate')
        return True
    return False


def mechanism_status(kb,cid,level=11):
    """Compact executable/pending coverage, not a claim of current balance."""
    implemented=set();pending=set()
    if deployment_pulses(kb,cid,level):implemented.add('deployment_damage')
    for spec,count in kb.roster(cid,level):
        raw=kb.units.get(spec.name,{})
        pending.update(spec.unsupported)
        for field,label in [('dash_damage','jump_or_dash'),('charge_damage','charge'),
                            ('reflected_damage','retaliation'),('minimum_range','minimum_range'),
                            ('death_damage','death_damage'),('death_spawn','death_spawn'),
                            ('spawn','periodic_spawn'),('shield','shield'),('stun','stun'),
                            ('resource_period','resource_generation')]:
            if getattr(spec,field):implemented.add(label)
        if spec.death_spawn and not kb.unit(spec.death_spawn,level):implemented.discard('death_spawn')
        if spec.chain_count>1:implemented.add('chain_attack')
        if spec.attack_targets>1:implemented.add('multiple_attack_targets')
        if pulse(kb,kb.effects.get(raw.get('spawn_area_object')),level):implemented.add('spawn_damage_and_control')
        elif raw.get('spawn_area_object'):pending.add('spawn_area_object')
    from .card_effects import PATTERNS
    if cid in PATTERNS:implemented.add('spell_'+PATTERNS[cid])
    pending.update(v for v in kb.cards[cid].get('unsupported',[]) if not (cid in PATTERNS and v=='spell_pattern'))
    if kb.cards[cid].get('variants'):pending.add('evolution_or_hero_variant')
    if not kb.roster(cid,level) and not kb.spell(cid,level):pending.add('unmapped_combat_mechanics')
    return dict(implemented=sorted(implemented),pending=sorted(pending),validated_current_balance=False)


def placement_special_bonus(point,roster,projected,events):
    """Coarse shortlist hint only; full temporal combat still selects the action."""
    score=0.
    for enemy,position in projected:
        weight=1+enemy.value
        distance=math.dist(point,position)
        for event in events:
            if (event['air'] or not enemy.spec.air) and (event.get('ground',True) or enemy.spec.air) and distance<=event['radius']+enemy.spec.radius:
                score+=weight*(min(enemy.hp,event['damage'])/max(1,enemy.hp)*2+min(1,event['stun']))
        for spec,count in roster:
            gap=max(0,distance-spec.radius-enemy.spec.radius)
            if spec.dash_damage and spec.dash_min<=gap<=spec.dash_max and ('air' if enemy.spec.air else 'ground') in spec.targets:
                score+=weight*min(enemy.hp,spec.dash_damage)/max(1,enemy.hp)/(1+spec.dash_windup+spec.dash_travel)
    return score
