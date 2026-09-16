"""Inspectable world packets and next-observation movement calibration.

Packets are immutable by convention after publication; consumers never mutate
planner state. Unknown identities retain unknown statistics.
"""
import math
from dataclasses import asdict
from .grid_navigation import route


def target_for(sim, state, entity):
    if entity.tower_kind == 'king' and not entity.active:
        return None
    enemies=[t for t in state.entities if t.side!=entity.side and t.hp>0
             and not (t.spec.dash_immune and t.dash_phase=='flight')
             and sim.distance(entity,t)>=entity.spec.minimum_range
             and ('air' if t.spec.air else 'ground') in entity.spec.targets
             and (not entity.spec.building_only or t.spec.building or t.tower)]
    if entity.tower:
        enemies=[t for t in enemies if sim.distance(entity,t)<=entity.spec.reach]
    sight=max(entity.spec.sight,entity.spec.reach) if entity.tower else entity.spec.sight
    locked=next((t for t in enemies if t.uid==entity.target and sim.distance(entity,t)<=sight),None)
    nearby=[t for t in enemies if sim.distance(entity,t)<=sight]
    # A building seeker marching toward a tower can still be intercepted by a
    # nearer defensive building. Preserve a tower attack already in range.
    if (locked is not None and locked.tower and not entity.tower and entity.spec.building_only
            and sim.distance(entity,locked)>entity.spec.reach and entity.winding_target is None):
        interceptors=[t for t in nearby if t.spec.building and not t.tower
                      and sim.distance(entity,t)<sim.distance(entity,locked)]
        if interceptors:
            return min(interceptors,key=lambda t:sim.distance(entity,t))
    return locked or min(nearby or [t for t in enemies if t.tower],key=lambda t:sim.distance(entity,t),default=None)


def along(points, distance):
    if not points:return None
    x,y=points[0]
    for tx,ty in points[1:]:
        length=math.hypot(tx-x,ty-y)
        if length>=distance and length:
            return [x+(tx-x)*distance/length,y+(ty-y)*distance/length]
        distance-=length;x,y=tx,ty
    return [x,y]


class GridWorldAudit:
    def __init__(self):
        self.previous=None

    def update(self, world, state, sim, plan):
        tracks={t.track_id:t for t in world.tracks}
        rows=[]; observed_ids=set()
        for entity in state.entities:
            t=tracks.get(entity.track_id)
            key=f'track:{t.track_id}' if t else f'tower:{entity.side}:{entity.tower_kind}:{entity.x:.2f}:{entity.y:.2f}' if entity.tower else f'predicted:{entity.uid}'
            if t:observed_ids.add(t.track_id)
            target=target_for(sim,state,entity)
            points=route(state,entity,target,sim.geometry) if target and entity.spec.speed>0 and sim.distance(entity,target)>entity.spec.reach else []
            source='observed' if t and world.at-t.last_seen<.05 else 'occluded' if t else 'tower_anchor' if entity.tower else 'deployment_hypothesis'
            tower_fraction=None
            if entity.tower:
                if entity.tower_kind=='king':
                    index=4 if entity.side==1 else 5
                else:
                    index=min(range(len(sim.geometry.tower_points)),key=lambda i: math.hypot(
                        sim.xy(*sim.geometry.tower_points[i])[0]-entity.x,sim.xy(*sim.geometry.tower_points[i])[1]-entity.y))
                tower_fraction=world.tower_health[index] if len(world.tower_health)>index else None
            rows.append(dict(id=key,track_id=entity.track_id,uid=entity.uid,side=entity.side,
                x=entity.x,y=entity.y,cell=[int(entity.x),int(entity.y)],card_id=t.card_id if t else entity.spec.name,
                hp=(None if tower_fraction is None else entity.hp) if entity.tower else entity.hp,
                hp_fraction=tower_fraction if entity.tower else entity.hp/entity.spec.hp,
                simulated_hp=entity.hp,simulated_hp_fraction=entity.hp/entity.spec.hp,
                hp_estimated=(t.hp_fraction is None if t else tower_fraction is None),
                identity_estimated=bool(t and (t.hypotheses or t.card_id.startswith('unknown:'))),
                level=t.level if t else None,side_evidence=t.side_evidence if t else None,
                source=source,confidence=t.confidence if t else None,age_s=world.at-t.last_seen if t else None,
                hypotheses=list(t.hypotheses) if t else [],attributes=asdict(entity.spec),tower=entity.tower,
                tower_kind=entity.tower_kind or ('princess' if entity.tower else ''),active=entity.active,
                tower_index=index if entity.tower else None,
                target_uid=target.uid if target else None,path=points,position_model='navigation_without_future_combat'))
        for t in world.tracks:
            if t.track_id in observed_ids:continue
            x,y=sim.xy(t.x,t.y)
            rows.append(dict(id=f'track:{t.track_id}',track_id=t.track_id,uid=None,side=t.side,x=x,y=y,
                cell=[int(x),int(y)],card_id=t.card_id,hp=None,hp_fraction=t.hp_fraction,hp_estimated=t.hp_fraction is None,
                identity_estimated=t.card_id.startswith('unknown:'),source='unmodeled_observation' if world.at-t.last_seen<.05 else 'occluded',confidence=t.confidence,age_s=world.at-t.last_seen,
                level=t.level,side_evidence=t.side_evidence,
                hypotheses=list(t.hypotheses),attributes=None,tower=False,active=None,target_uid=None,path=[]))
        # Destroyed anchors remain inspectable, but never re-enter simulation.
        for index,point in enumerate(sim.geometry.tower_points):
            if index>=len(world.tower_health) or world.tower_health[index]!=0:continue
            x,y=sim.xy(*point)
            rows.append(dict(id=f'destroyed-tower:{index}',side=1 if index<2 else -1,
                x=x,y=y,cell=[int(x),int(y)],card_id='princess_tower',tower=True,tower_kind='princess',
                tower_index=index,hp=0,hp_fraction=0.,hp_estimated=False,destroyed=True,active=False,
                source='observed_destroyed',target_uid=None,path=[]))
        errors=[]
        previous=self.previous
        if previous and 0 < world.at-previous['at'] <= 2:
            dt=world.at-previous['at'];old={r['id']:r for r in previous['entities']}
            for r in rows:
                p=old.get(r['id'])
                if r['source']!='observed' or not p or p['source']!='observed' or p['card_id']!=r['card_id'] or not p['attributes']:continue
                expected=along(p['path'],p['attributes']['speed']*dt) or [p['x'],p['y']]
                error=math.hypot(r['x']-expected[0],r['y']-expected[1])
                errors.append(dict(id=r['id'],position_error_tiles=round(error,3),expected=expected,actual=[r['x'],r['y']],
                    hp_change=None if p['hp_estimated'] or r['hp_estimated'] else round(r['hp_fraction']-p['hp_fraction'],3)))
        packet=dict(schema='grid_world_v1',revision=world.revision,at=world.at,elapsed=world.elapsed,
                    width=18,height=32,river_rows=[15,16],bridges=list(sim.geometry.bridges),geometry=asdict(sim.geometry),
                    elixir=world.elixir,enemy_elixir=list(world.enemy_elixir),entities=rows,tower_health=list(world.tower_health),
                    action=asdict(plan.action),decision=plan.reason,status=plan.status,
                    calibration=dict(samples=len(errors),errors=errors,mean_position_error_tiles=round(sum(e['position_error_tiles'] for e in errors)/len(errors),3) if errors else None,
                        scope='movement_forecast_and_observed_hp_change; not outcome_or_combat_accuracy'),
                    pending_effects=[dict(e) for e in state.effects],pending_projectiles=len(state.impacts))
        self.previous=packet
        return packet
