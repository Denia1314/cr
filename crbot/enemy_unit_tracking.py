"""Enemy identity anchored by a real >=80% detection, with bounded occlusion."""
import copy
import math


class EnemyUnitTracker:
    def __init__(self, sim):
        self.sim=sim
        self.reset()

    def reset(self, generation=None):
        self.generation=generation
        self.locks={}
        self.sequence=0
        self.cached_at=None
        self.cached=None

    def update(self, observations, candidates, *, now, generation, image=None):
        if generation != self.generation or self.cached_at is not None and now < self.cached_at:
            self.reset(generation)
        if now == self.cached_at:
            return copy.deepcopy(self.cached)
        observations=[dict(d) for d in observations]
        candidates=[dict(d) for d in candidates]
        self.locks={k:v for k,v in self.locks.items() if now-v['at']<=3}
        existing=set(self.locks)
        bodies=[]
        for kind,rows in (('observation',observations),('candidate',candidates)):
            for index,d in enumerate(rows):
                box=d.get('bbox',())
                confidence=float(d.get('confidence',0))
                if d.get('side') != -1 or len(box)!=4 or not math.isfinite(confidence) or not .25<=confidence<=1:
                    continue
                if not all(math.isfinite(float(v)) and 0<=float(v)<=1 for v in box) or not .012<box[2]-box[0]<.25 or not .012<box[3]-box[1]<.25:
                    continue
                x,y=(box[0]+box[2])/2,(box[1]+box[3])/2
                if not .20<=y<=.80:
                    continue
                bodies.append(dict(kind=kind,index=index,row=d,x=x,y=y))
        model=getattr(self.sim,'tracking_association_model',None)
        if model is not None and image is not None:
            from .tracking_learning import descriptor
            for body in bodies:body['row']['appearance']=descriptor(image,body['row']['bbox'])
        def distance(lock,body):
            age=min(.75,max(0.,now-lock['at']))
            a=self.sim.xy(lock['x']+lock['vx']*age,lock['y']+lock['vy']*age)
            return math.dist(a,self.sim.xy(body['x'],body['y']))
        choices={}
        for key,lock in self.locks.items():
            ranked=sorted((distance(lock,b),i) for i,b in enumerate(bodies))
            roster=self.sim.kb.roster(lock['card_id'],self.sim.level)
            speed=max((spec.speed for spec,count in roster),default=1.5)
            radius=min(5.,1.25+max(0.,now-lock['at'])*speed)
            if model is not None and 'appearance' in lock['row'] and all('appearance' in b['row'] for b in bodies):
                from .tracking_learning import features,probability
                scores=[]
                for i,b in enumerate(bodies):
                    if distance(lock,b)>radius:continue
                    row=dict(b['row'],card_id=b['row'].get('card_id',b['row'].get('candidate_card_id','')))
                    score=probability(model['weights'],features(dict(lock['row'],card_id=lock['card_id']),row,now-lock['at']))
                    scores.append((score,i))
                scores.sort(reverse=True)
                if scores and scores[0][0]>=model['threshold'] and (len(scores)==1 or scores[0][0]-scores[1][0]>=.1):
                    choices[key]=(distance(lock,bodies[scores[0][1]]),scores[0][1])
                continue
            if ranked and ranked[0][0]<=radius and (len(ranked)==1 or ranked[1][0]-ranked[0][0]>=.5):
                choices[key]=ranked[0]
        matched={}
        for key,(dist,index) in choices.items():
            rivals=sorted((distance(lock,bodies[index]),other) for other,lock in self.locks.items())
            if model is not None and 'appearance' in bodies[index]['row'] and all('appearance' in l['row'] for l in self.locks.values()):
                from .tracking_learning import features,probability
                row=dict(bodies[index]['row'],card_id=bodies[index]['row'].get('card_id',bodies[index]['row'].get('candidate_card_id','')))
                rivals=sorted((-probability(model['weights'],features(dict(l['row'],card_id=l['card_id']),row,now-l['at'])),other)
                    for other,l in self.locks.items() if distance(l,bodies[index])<=5)
                if rivals and rivals[0][1]==key and (len(rivals)==1 or rivals[1][0]-rivals[0][0]>=.1):matched[index]=key
                continue
            if rivals[0][1]==key and (len(rivals)==1 or rivals[1][0]-rivals[0][0]>=.5):
                matched[index]=key
        consumed_obs=set();consumed_candidates=set();visible=[]
        for index,body in enumerate(bodies):
            d=body['row'];key=matched.get(index)
            cid=d.get('card_id')
            if key is None:
                # Nearby ambiguous bodies cannot start another identity and
                # double-count the old locked enemy in the planner.
                nearby=any(distance(lock,body)<=5 for k,lock in self.locks.items() if k in existing and k not in matched.values())
                if nearby:
                    if body['kind']=='observation':
                        observations[body['index']]['identity_ambiguous']=True
                    continue
                if body['kind']!='observation' or not cid or cid.startswith('unknown:') or float(d['confidence'])<.8 or not self.sim.kb.roster(cid,self.sim.level):
                    continue
                self.sequence+=1;key=f'enemy:{generation}:{self.sequence}'
                self.locks[key]=dict(card_id=cid,x=body['x'],y=body['y'],at=now,vx=0.,vy=0.,
                                     anchor_confidence=float(d['confidence']),row=dict(d))
            lock=self.locks[key]
            if d.get('hp_fraction')==0 and float(d.get('hp_confidence',0))>=.8:
                visible.append(dict(d,card_id=lock['card_id'],side=-1,x=body['x'],y=body['y'],
                    identity_source='enemy_yolo_tracking',identity_lock_id=key,
                    identity_anchor_confidence=lock['anchor_confidence']))
                (consumed_obs if body['kind']=='observation' else consumed_candidates).add(body['index'])
                del self.locks[key]
                continue
            dt=now-lock['at']
            if .05<=dt<=1.5:
                lock['vx']=max(-.25,min(.25,(body['x']-lock['x'])/dt))
                lock['vy']=max(-.25,min(.25,(body['y']-lock['y'])/dt))
            lock.update(x=body['x'],y=body['y'],at=now,row=dict(d))
            row=dict(d,card_id=lock['card_id'],side=-1,x=body['x'],y=body['y'],
                     identity_lock_id=key,identity_source='enemy_yolo_tracking',
                     identity_anchor_confidence=lock['anchor_confidence'],
                     visual_card_id=d.get('card_id',d.get('candidate_card_id')),
                     visual_confidence=float(d['confidence']),position_state='visible',hypotheses=())
            visible.append(row)
            (consumed_obs if body['kind']=='observation' else consumed_candidates).add(body['index'])
        output=[d for i,d in enumerate(observations) if i not in consumed_obs]+visible
        # Suppress overlapping badge-only hypotheses; these are not extra troops.
        for d in output:
            if d.get('side')==-1 and str(d.get('card_id','')).startswith('unknown:') and any(
                    math.dist(self.sim.xy(d['x'],d['y']),self.sim.xy(v['x'],v['y']))<=1.5 for v in visible):
                d['identity_ambiguous']=True
        remaining=[d for i,d in enumerate(candidates) if i not in consumed_candidates]
        tracks=[]
        for key,lock in self.locks.items():
            age=now-lock['at'];advance=min(.75,age)
            row=dict(lock['row'],card_id=lock['card_id'],side=-1,
                x=lock['x']+lock['vx']*advance,y=lock['y']+lock['vy']*advance,
                identity_lock_id=key,identity_source='enemy_yolo_tracking',
                identity_anchor_confidence=lock['anchor_confidence'],last_visual_at=lock['at'],
                position_state='visible' if age==0 else 'occluded')
            if age>0:
                dx,dy=lock['vx']*advance,lock['vy']*advance
                row['bbox']=[max(0.,min(1.,v+(dx if i%2==0 else dy))) for i,v in enumerate(row['bbox'])]
            tracks.append(row)
        result=(output,remaining,tracks)
        self.cached_at=now;self.cached=copy.deepcopy(result)
        return result
