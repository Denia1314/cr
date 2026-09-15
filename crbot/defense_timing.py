"""Conservative arrival timing; all values are predictions, not observations."""
import math


def approach_speed(entity, target):
    dx,dy=target.x-entity.x,target.y-entity.y
    distance=max(.01,math.hypot(dx,dy))
    observed=(entity.observed_vx*dx+entity.observed_vy*dy)/distance
    nominal=entity.spec.speed
    # Never let noisy/standing observations postpone an otherwise fast threat.
    return max(nominal,min(nominal*1.5,max(0.,observed)))


def forecast(sim, state, hand, pipeline_s=.35):
    from .grid_navigation import route
    towers=[t for t in state.entities if t.side==1 and t.tower and t.hp>0 and t.tower_kind!='king']
    result=[]
    for enemy in (e for e in state.entities if e.side==-1 and not e.tower and e.hp>0):
        tower=min(towers,key=lambda t:sim.distance(enemy,t),default=None)
        if tower is None:
            continue
        path=route(state,enemy,tower,sim.geometry) if getattr(sim,"grid_navigation",False) else [(enemy.x,enemy.y),(tower.x,tower.y)]
        distance=sum(math.dist(a,b) for a,b in zip(path,path[1:])) if path else sim.distance(enemy,tower)
        gap=max(0.,distance-enemy.spec.reach-tower.spec.radius-enemy.spec.radius)
        speed=approach_speed(enemy,tower)
        eta=gap/max(.1,speed)
        leads=[]
        for _,cid in hand:
            card=sim.kb.cards.get(cid,{})
            if card.get('elixir') is None or card['elixir']>state.elixir[1]:continue
            spell=sim.kb.spell(cid,sim.level)
            if spell and not spell.get('friendly') and (not enemy.spec.air or spell.get('air')):
                leads.append(float(spell['delay']))
            for spec,count in sim.kb.roster(cid,sim.level):
                hits=(not spec.building_only or enemy.spec.building) and ('air' if enemy.spec.air else 'ground') in spec.targets
                pulls=(not enemy.spec.building_only or spec.building) and ('air' if spec.air else 'ground') in enemy.spec.targets
                if hits or pulls:leads.append(spec.deploy+spec.first_hit)
        lead=max(0.,pipeline_s)+min(leads,default=1.5)
        result.append(dict(unit=enemy.spec.name,track_id=enemy.track_id,x=enemy.x,y=enemy.y,
                           target_x=tower.x,target_y=tower.y,unopposed_tower_eta_s=round(eta,3),
                           defense_lead_s=round(lead,3),intervention_slack_s=round(eta-lead,3),
                           approach_speed_tiles_s=round(speed,3),path=path,
                           observed_velocity=[enemy.observed_vx,enemy.observed_vy],
                           linear_samples=[dict(dt=dt,x=enemy.x+enemy.observed_vx*dt,y=enemy.y+enemy.observed_vy*dt) for dt in (.5,1.,2.)],
                           status='hypothesis'))
    return result


def project_approach(sim,state,enemy,tower,delay):
    from .grid_navigation import route
    path=route(state,enemy,tower,sim.geometry) if getattr(sim,"grid_navigation",False) else [(enemy.x,enemy.y),(tower.x,tower.y)]
    if not path:return enemy.x,enemy.y
    lengths=[math.dist(a,b) for a,b in zip(path,path[1:])]
    remaining=min(approach_speed(enemy,tower)*max(0.,delay),
                  max(0.,sum(lengths)-enemy.spec.reach-tower.spec.radius-enemy.spec.radius))
    for a,b,length in zip(path,path[1:],lengths):
        if remaining<=length:
            f=remaining/max(.001,length)
            return a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f
        remaining-=length
    return path[-1]
