"""Weak visual bodies may locate an existing deployment hypothesis, never a track."""
import math


def box_overlap(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def deployment_positions(world,sim):
    """Mutually unique nearby bodies, recent confirmed single-entity cards only."""
    options={};candidates={}
    for i,event in enumerate(world.confirmed_placements):
        age=world.at-event['at']
        roster=sim.kb.roster(event['card_id'],sim.level)
        if not 0<=age<=3 or sum(n for _,n in roster)!=1:continue
        if any(t.side==1 and t.card_id==event['card_id'] and world.at-t.last_seen<=1.5 for t in world.tracks):continue
        spec=roster[0][0]
        x,y=sim.xy(event['x'],event['y'])
        y=max(16.,y-spec.speed*max(0,age-spec.deploy))
        for j,c in enumerate(world.unconfirmed_units):
            box=c.get('bbox',())
            if (c.get('side') not in (None,1) or c.get('observation_status')!='unconfirmed'
                    or not math.isfinite(c.get('confidence',0)) or c.get('confidence',0)<.25 or len(box)!=4):continue
            if not all(math.isfinite(v) and 0<=v<=1 for v in box):continue
            if not .012<box[2]-box[0]<.25 or not .012<box[3]-box[1]<.25:continue
            nx,ny=(box[0]+box[2])/2,(box[1]+box[3])/2
            if not .2<=ny<=.78:continue
            if any(world.at-t.last_seen<=1.5 and math.hypot(t.x-nx,t.y-ny)<.08 for t in world.tracks):continue
            px,py=sim.xy(nx,ny)
            if math.hypot(px-x,py-y)>(6.5 if spec.dash_damage else 2.5):continue
            options.setdefault(i,[]).append(j)
            candidates.setdefault(j,[]).append(i)
    return {i:sim.xy((world.unconfirmed_units[js[0]]['bbox'][0]+world.unconfirmed_units[js[0]]['bbox'][2])/2,
                    (world.unconfirmed_units[js[0]]['bbox'][1]+world.unconfirmed_units[js[0]]['bbox'][3])/2)
            for i,js in options.items() if len(js)==1 and len(candidates[js[0]])==1}
