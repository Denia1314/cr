"""Bind confirmed deployments to unique visual bodies; keep identity across poses.

This is deployment-assisted identity, not a new YOLO classification or proof of
survival. Each consumer owns its tracker; only confirmed events are shared.
"""
import math

from .unconfirmed_units import box_overlap


class OwnUnitTracker:
    def __init__(self, sim):
        self.sim = sim
        self.reset()

    def reset(self):
        self.generation = None
        self.events = {}
        self.locks = {}
        self.probes = {}
        self.history = []
        self.at = None
        self.last_rows=[]
        self.last_pending=[]

    def _distance(self, a, b):
        ax, ay = self.sim.xy(*a)
        bx, by = self.sim.xy(*b)
        return math.hypot(ax-bx, ay-by)

    @staticmethod
    def _unique_nearest(options, distance):
        ranked=sorted(((distance(item),index,item) for index,item in enumerate(options)),key=lambda r:r[0])
        if not ranked or (len(ranked)>1 and ranked[1][0]-ranked[0][0]<.5):
            return None
        return ranked[0][2]

    def update(self, detections, candidates, events, *, now, generation=0):
        if generation != self.generation:
            self.reset()
            self.generation = generation
        if self.at is not None and now == self.at:
            # Reused or older captures cannot add confirmation evidence.
            return [dict(r) for r in self.last_rows], [dict(r) for r in self.last_pending], self.rows(now)
        if self.at is not None and now < self.at:
            return list(detections), [dict(c) for c in candidates if not c.get('tracking_only')], []
        dt = now-self.at if self.at is not None else 0
        self.at = now
        for event in events:
            if event['id'] not in self.events and 0 <= now-event['at'] <= 3.5:
                roster = self.sim.kb.roster(event['card_id'], self.sim.level)
                # Homogeneous swarms have separate slots. Mixed summons need
                # entity-specific attribution and are deliberately not guessed.
                if len(roster) == 1 and 1 <= roster[0][1] <= 20:
                    spec, count = roster[0]
                    self.events[event['id']] = dict(event, count=count, spec=spec, used=0)
        bodies = []
        raw = [dict(d) for d in detections]
        pending = [dict(c) for c in candidates]
        for weak, source in ((False, raw), (True, pending)):
            for index, row in enumerate(source):
                box = row.get('bbox', ())
                score = row.get('confidence', 0)
                if (len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box)
                        or not math.isfinite(score) or score < (.1 if row.get('tracking_only') else .25)
                        or not .012 < box[2]-box[0] < .25 or not .012 < box[3]-box[1] < .25):
                    continue
                if weak and any(box_overlap(box, d.get('bbox', (0,0,0,0))) > .3
                                for d in raw if len(d.get('bbox', ())) == 4):
                    continue
                point = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                if not .15 <= point[1] <= .8:
                    continue
                previous = [h for h in self.history if self._distance(h['point'], point) < 1.5]
                first = min((h['first'] for h in previous), default=now)
                bodies.append(dict(row=row, weak=weak, index=index, point=point, first=first))
        claimed = set()
        matches = {}
        dead = set()
        # Locks expire after a bounded occlusion; their slots are never recycled.
        self.locks = {k:v for k,v in self.locks.items() if now-v['seen'] <= 3}
        proposals = {}
        possible_locked_bodies=set()
        for key, lock in self.locks.items():
            age = now-lock['seen']
            expected = (lock['point'][0]+lock['vx']*min(age,.75),
                        lock['point'][1]+lock['vy']*min(age,.75))
            radius = min(6.5 if lock['jump'] else 3.5, 1.5+age*3)
            options = []
            for j, body in enumerate(bodies):
                row = body['row']
                if row.get('side') not in (None, 1):
                    continue
                if age > 1.5 and (row.get('side') != 1 or row.get('card_id') != lock['card_id']):
                    if self._distance(expected,body['point']) > 2.5+age*lock['speed']:
                        continue
                if self._distance(expected, body['point']) <= radius:
                    options.append(j)
            possible_locked_bodies.update(options)
            nearest=self._unique_nearest(options,lambda j:self._distance(expected,bodies[j]['point']))
            proposals[key] = [] if nearest is None else [nearest]
        for key, options in proposals.items():
            if len(options) != 1:
                continue
            j = options[0]
            if sum(j in others for others in proposals.values()) != 1:
                continue
            lock = self.locks[key]
            body = bodies[j]
            if body['row'].get('hp_fraction') == 0 and body['row'].get('hp_confidence',1) >= .8:
                dead.add(key)
                claimed.add(j)
                matches[j] = key
                continue
            age = now-lock['seen']
            lock['vx'] = max(-.3,min(.3,(body['point'][0]-lock['point'][0])/age)) if age > .02 else 0
            lock['vy'] = max(-.3,min(.3,(body['point'][1]-lock['point'][1])/age)) if age > .02 else 0
            lock.update(point=body['point'], seen=now, bbox=list(body['row']['bbox']))
            matches[j] = key
            claimed.add(j)
        available = {}
        for j, body in enumerate(bodies):
            if j in claimed or body['row'].get('side') not in (None,1) or (body['row'].get('hp_fraction') == 0 and body['row'].get('hp_confidence',1) >= .8):
                continue
            owners = []
            for event_id, event in self.events.items():
                age = now-event['at']
                if not 0 <= age <= 6 or event['used'] >= event['count'] or body['first'] < event.get('issued_at',event['at']):
                    continue
                spec = event['spec']
                ex, ey = self.sim.xy(event['x'],event['y'])
                ey -= spec.speed*max(0,now-event.get('issued_at',event['at'])-spec.deploy)
                bx, by = self.sim.xy(*body['point'])
                if math.hypot(ex-bx,ey-by) <= (6.5 if spec.dash_damage else 2.5):
                    owners.append(event_id)
            if len(owners) == 1:
                available.setdefault(owners[0],[]).append(j)
        next_probes = {}
        waiting_bodies={j for indices in available.values() for j in indices}
        for event_id, indices in available.items():
            event = self.events[event_id]
            # Extra nearby bodies make attribution ambiguous, rather than
            # silently assigning the nearest one to a confirmed card.
            if len(indices) > event['count']-event['used']:
                continue
            for j in indices:
                body = bodies[j]
                def probe_radius(probe):
                    return 6.5 if event['spec'].dash_damage else min(4.,1.5+event['spec'].speed*(now-probe['at']))
                old = [p for p in self.probes.get(event_id,[]) if self._distance(p['point'],body['point']) <= probe_radius(p)]
                nearest=self._unique_nearest(old,lambda p:self._distance(p['point'],body['point']))
                old=[] if nearest is None else [nearest]
                if not old and now-event['at'] > 3.5:
                    continue
                probe = dict(point=body['point'], at=now)
                next_probes.setdefault(event_id,[]).append(probe)
                if len(old) != 1 or not .02 <= now-old[0]['at'] <= 2.5:
                    continue
                nearby=[i for i in indices if self._distance(old[0]['point'],bodies[i]['point']) <= probe_radius(old[0])]
                if self._unique_nearest(nearby,lambda i:self._distance(old[0]['point'],bodies[i]['point'])) != j:
                    continue
                event['used'] += 1
                key = f"{event_id}:{event['used']}"
                self.locks[key] = dict(card_id=event['card_id'], deployment_id=event_id,
                    point=body['point'], bbox=list(body['row']['bbox']), seen=now, vx=0.,vy=0.,
                    jump=bool(event['spec'].dash_damage),speed=event['spec'].speed)
                matches[j] = key
                claimed.add(j)
        self.probes = next_probes
        self.history = [dict(point=b['point'],first=b['first']) for b in bodies]
        for j, key in matches.items():
            body = bodies[j]
            if key is None:
                continue
            lock = self.locks[key]
            row = dict(body['row'],card_id=lock['card_id'],side=1,x=body['point'][0],y=body['point'][1],
                identity_source='confirmed_deployment_tracking', identity_lock_id=key,
                deployment_id=lock['deployment_id'],visual_card_id=body['row'].get('card_id') or body['row'].get('candidate_card_id'),
                deployment_at=self.events[lock['deployment_id']].get('issued_at',self.events[lock['deployment_id']]['at']),
                visual_confidence=body['row']['confidence'],position_state='visible',
                observation_status='deployment_bound',hypotheses=(),variant=body['row'].get('variant','unknown'))
            # Side comes from confirmed action + association, not a fake badge.
            row['side_evidence'] = 'confirmed_deployment_visual_association'
            row.pop('identity_ambiguous',None)
            if body['weak']:
                raw.append(row)
            else:
                raw[body['index']] = row
        for j in (possible_locked_bodies | waiting_bodies)-claimed:
            body=bodies[j]
            if not body['weak']:
                raw[body['index']]['identity_ambiguous']=True
        removed = {bodies[j]['index'] for j in claimed if bodies[j]['weak']}
        pending = [c for i,c in enumerate(pending) if i not in removed and not c.get('tracking_only')]
        for key in dead:
            del self.locks[key]
        self.events = {k:v for k,v in self.events.items() if now-v['at'] <= 6 or any(l['deployment_id']==k for l in self.locks.values())}
        self.last_rows=[dict(r) for r in raw]
        self.last_pending=[dict(r) for r in pending]
        return raw, pending, self.rows(now)

    def rows(self, now):
        rows=[]
        for key,v in self.locks.items():
            age=max(0,now-v['seen']);dx=v['vx']*min(age,.75);dy=v['vy']*min(age,.75)
            box=[max(0,min(1,b+(dx if i%2==0 else dy))) for i,b in enumerate(v['bbox'])]
            rows.append(dict(identity_lock_id=key,card_id=v['card_id'],deployment_id=v['deployment_id'],side=1,
                bbox=box,x=max(0,min(1,v['point'][0]+dx)),y=max(0,min(1,v['point'][1]+dy)),age_s=age,
                identity_source='confirmed_deployment_tracking',position_state='visible' if age==0 else 'occluded'))
        return rows
