"""Batched position/enemy/tower ranking; detailed combat remains in Simulator."""
from __future__ import annotations
import numpy as np
from .gpu import resolve_device


class PlacementBatch:
    roster_scoring = True
    def __init__(self, requested="auto", *, trajectory=False, trajectory_step=.25):
        self.device = resolve_device(requested)
        self.trajectory=trajectory
        self.trajectory_step=max(.125,min(.5,float(trajectory_step)))
        self.trajectory_steps=round(8/self.trajectory_step)+1
        self.calls = 0
        self.positions = 0
        self.error = ""
        self.domain_points = 0
        self.spell_positions = 0
        self.quiet_positions = 0
        if self.device != 'cpu':
            from types import SimpleNamespace
            # Warm every ranking operation before the first battle's time budget starts.
            spec = SimpleNamespace(reach=3.,radius=.5,hp=1000.,shield=0.,damage=100.,period=1.,first_hit=.5,speed=1.,building=True)
            self._compute([[1.,20.]], [[2.,18.,.5,1.,1.,2.,1.,500.,100.,4.,5.,6.,1.,1.,.5,.5]],
                          [[3.,28.,8.,100.]], [[1.]], spec)

    def status(self):
        return dict(device=self.device, calls=self.calls, positions=self.positions, error=self.error,
                    scope="placement_ranking_and_dps_trajectory" if self.trajectory else "placement_ranking",
                    domain_points=self.domain_points, spell_positions=self.spell_positions,
                    quiet_positions=self.quiet_positions,
                    trajectory_steps=self.trajectory_steps if self.trajectory else 0, combat_device="cpu")

    def _arrays(self):
        if self.device == 'cpu':
            return np, lambda v: np.asarray(v, dtype=np.float64)
        import torch
        return torch, lambda v: torch.as_tensor(v, dtype=torch.float64, device=self.device)

    def _run(self, function):
        try:
            return function()
        except (RuntimeError, ImportError, OSError) as exc:
            self.error = str(exc)
            self.device = 'cpu'
            return function()

    def legal_points(self, sim, state, card_id, side, points):
        """Same legal domain as Simulator, with bounded obstacle batches."""
        if not points:
            return []
        def calculate():
            xp, arr = self._arrays()
            result = []
            l, t, r, b = sim.geometry.bounds
            obstacles = [[e.x,e.y,e.spec.radius+.6] for e in state.entities
                         if e.hp > 0 and (e.tower or e.spec.building)]
            spell = sim.kb.cards.get(card_id, {}).get('kind') == 'spell'
            for start in range(0,len(points),4096):
                chunk = points[start:start+4096]
                p = arr(chunk)
                xy = (p-arr([l,t]))/arr([r-l,b-t])*arr([18,32])
                x,y = xy[:,0],xy[:,1]
                valid = (p[:,0]>=l)&(p[:,0]<=r)&(p[:,1]>=t)&(p[:,1]<=b)
                home = y+1e-8>=16.5 if side==1 else y-1e-8<=15.5
                if card_id=='royal_delivery' or not spell:
                    valid &= home
                if not spell:
                    valid &= ~((abs(x-9)<2)&(abs(y-(30 if side==1 else 2))<2))
                    if obstacles:
                        obs=arr(obstacles)
                        distance=((xy[:,None,:]-obs[None,:,:2])**2).sum(axis=-1)**.5
                        valid &= ~(distance<obs[None,:,2]).any(axis=1)
                mask=valid.tolist() if self.device=='cpu' else valid.cpu().tolist()
                result.extend(point for point, keep in zip(chunk,mask) if keep)
            return result
        result=self._run(calculate)
        self.domain_points+=len(points)
        return result

    def spell_score(self, points, enemies, projected, spell):
        def calculate():
            xp, arr = self._arrays()
            if not enemies:
                return [0.] * len(points)
            targets=arr([[*q,e.spec.radius,float(not e.spec.air or spell['air']),
                          (e.spec.damage if spell.get('friendly') else min(e.hp+e.shield,spell['damage']*(spell.get('tower_multiplier',1.) if e.tower else 1.)))*(1+e.value)]
                         for e,q in zip(enemies,projected)])
            output=[]
            for start in range(0,len(points),4096):
                p=arr(points[start:start+4096])
                d=((p[:,None,:]-targets[None,:,:2])**2).sum(axis=-1)**.5
                score=((d<=spell['radius']+targets[None,:,2])*targets[None,:,3]*targets[None,:,4]).sum(axis=-1)
                output.extend(score.tolist() if self.device=='cpu' else score.cpu().tolist())
            return output
        result=self._run(calculate)
        self.spell_positions+=len(points)
        self.calls+=1
        return result

    def quiet_score(self, points, towers, side):
        def calculate():
            xp,arr=self._arrays()
            p=arr(points)
            if towers:
                t=arr([[e.x,e.y-side*2] for e in towers])
                d=((p[:,None,:]-t[None,:,:])**2).sum(axis=-1)**.5
                score=-.2*(d.min(axis=1) if self.device=='cpu' else d.amin(dim=1))
            else:
                score=-.2*abs(p[:,0]-9)
            return score.tolist() if self.device=='cpu' else score.cpu().tolist()
        result=self._run(calculate)
        self.quiet_positions+=len(points)
        self.calls+=1
        return result

    def score(self, points, spec, projected, towers, *, count=1):
        # Precompute immutable card attributes once; all positions share this batch.
        if not projected:
            return None
        from .card_matchup import matchup
        rows = []
        for enemy, q in projected:
            comparison = matchup(spec, enemy, count)
            hit = (not spec.building_only or enemy.spec.building) and ('air' if enemy.spec.air else 'ground') in spec.targets
            pull = (not enemy.spec.building_only or spec.building) and ('air' if spec.air else 'ground') in enemy.spec.targets
            urgency = 1/(1+min((max(0,np.hypot(q[0]-t.x,q[1]-t.y)-enemy.spec.reach-t.spec.radius) for t in towers),default=12)/5)
            rows.append([*q, enemy.spec.radius, float(hit), float(pull),
                         max(.5,spec.speed+(enemy.spec.speed if pull else 0)), (1+enemy.value)*urgency,
                         enemy.hp+enemy.shield,enemy.spec.damage/max(.1,enemy.spec.period),
                         min((max(0,np.hypot(q[0]-t.x,q[1]-t.y)-enemy.spec.reach-t.spec.radius)/max(.1,enemy.spec.speed) for t in towers),default=8.),
                         enemy.spec.sight, min((np.hypot(q[0]-t.x,q[1]-t.y)-t.spec.radius-enemy.spec.radius for t in towers),default=32.),
                         float(enemy.spec.building_only),enemy.spec.speed, comparison['attack_weight'], comparison['stall_weight']])
        towers = [t for t in towers if t.active]
        cover = [[t.x,t.y,t.spec.reach+t.spec.radius,t.spec.damage/max(.1,t.spec.period)] for t in towers]
        eligible = [[float(('air' if e.spec.air else 'ground') in t.spec.targets) for t in towers] for e,_ in projected]
        try:
            result = self._compute(points, rows, cover, eligible, spec)
        except (RuntimeError, ImportError, OSError) as exc:
            self.error = str(exc)
            self.device = 'cpu'
            result = self._compute(points, rows, cover, eligible, spec)
        self.calls += 1
        if self.calls == 1 and self.device.startswith("cuda"):
            print(f"[GPU] Defensive placement ranking: {self.device}, positions={len(points)}")
        self.positions += len(points)
        return result

    def _compute(self, points, rows, cover, eligible, spec):
        if self.device == 'cpu':
            xp = np
            arr = lambda a: np.asarray(a,dtype=np.float64)
            clamp = lambda a: np.maximum(a,0)
        else:
            import torch
            xp = torch
            arr = lambda a: torch.tensor(a,dtype=torch.float64,device=self.device)
            clamp = lambda a: a.clamp_min(0)
        p, e = arr(points), arr(rows)
        distance = ((p[:,None,:]-e[None,:,:2])**2).sum(axis=-1)**.5
        gap=distance-spec.radius-e[None,:,2]
        pull=e[None,:,4]+distance*0
        if spec.building:
            # Building seekers must acquire this building before locking the tower.
            viable=(gap<=e[None,:,10])&(gap<e[None,:,11])
            pull=pull*((1-e[None,:,12])+e[None,:,12]*viable)
        contact = clamp(distance-spec.reach-spec.radius-e[None,:,2])/(.5+clamp(spec.speed+e[None,:,13]*pull-.5))
        fight = e[None,:,:2] + (p[:,None,:]-e[None,:,:2])*.5*pull[:,:,None]
        tower_dps = distance*0
        if cover:
            t = arr(cover)
            d = ((fight[:,:,None,:]-t[None,None,:,:2])**2).sum(axis=-1)**.5
            tower_dps = ((d <= t[None,None,:,2]+e[None,:,None,2])*arr(eligible)[None,:,:]*t[None,None,:,3]).sum(axis=-1)
        value = e[None,:,6]*xp.exp(-contact/2)*((2+2*e[None,:,14])*e[None,:,3]+pull*(1+e[None,:,15])+tower_dps/100)
        value -= e[None,:,6]*abs(distance-min(4.,spec.reach)*.85)*.15*e[None,:,3]
        if spec.reach > 2:
            value -= e[None,:,6]*clamp(min(spec.reach,4)-distance)*.8*e[None,:,3]*pull
        value -= e[None,:,6]*(1-e[None,:,3])*(1-pull)
        if self.trajectory:
            # All positions and observed enemies receive the same temporal DPS screen.
            # Share firepower rather than allowing a single unit/tower to shoot every target.
            ticks=arr([i*self.trajectory_step for i in range(self.trajectory_steps)])[None,None,:]
            own_dps=spec.damage/max(.1,spec.period)*e[None,:,3]/max(1,sum(r[3] for r in rows))
            shared_tower=tower_dps/max(1,len(rows))
            incoming=(e[None,:,8]*pull*xp.exp(-contact/2)).sum(axis=-1)
            lifetime=(spec.hp+spec.shield)/(incoming+.001)
            active=clamp(ticks-contact[:,:,None]-spec.first_hit)
            cap=clamp(lifetime[:,None]-contact-spec.first_hit)[:,:,None]
            active=xp.minimum(active,cap)
            hp=e[None,:,None,7]-own_dps[:,:,None]*active-shared_tower[:,:,None]*ticks
            stall=clamp(lifetime[:,None]-contact)*pull*(contact<e[None,:,9])
            arrival=e[None,:,9]+stall
            tower_damage=((hp>0)*(ticks>=arrival[:,:,None])*e[None,:,None,8]*self.trajectory_step).sum(axis=-1)
            value-=tower_damage/150
        result = value.sum(axis=-1)
        return result.tolist() if self.device == 'cpu' else result.cpu().tolist()
