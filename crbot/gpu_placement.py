"""Batched position/enemy/tower ranking; detailed combat remains in Simulator."""
from __future__ import annotations
import numpy as np
from .gpu import resolve_device


class PlacementBatch:
    def __init__(self, requested="auto", *, trajectory=False):
        self.device = resolve_device(requested)
        self.trajectory=trajectory
        self.calls = 0
        self.positions = 0
        self.error = ""
        if self.device != 'cpu':
            from types import SimpleNamespace
            # Warm every ranking operation before the first battle's time budget starts.
            spec = SimpleNamespace(reach=3.,radius=.5,hp=1000.,shield=0.,damage=100.,period=1.,first_hit=.5)
            self._compute([[1.,20.]], [[2.,18.,.5,1.,1.,2.,1.,500.,100.,4.]],
                          [[3.,28.,8.,100.]], [[1.]], spec)

    def status(self):
        return dict(device=self.device, calls=self.calls, positions=self.positions, error=self.error,
                    scope="placement_ranking_and_dps_trajectory" if self.trajectory else "placement_ranking",
                    trajectory_steps=33 if self.trajectory else 0, combat_device="cpu")

    def score(self, points, spec, projected, towers):
        # Precompute immutable card attributes once; all positions share this batch.
        if not projected:
            return None
        rows = []
        for enemy, q in projected:
            hit = ('air' if enemy.spec.air else 'ground') in spec.targets
            pull = (not enemy.spec.building_only or spec.building) and ('air' if spec.air else 'ground') in enemy.spec.targets
            urgency = 1/(1+min((max(0,np.hypot(q[0]-t.x,q[1]-t.y)-enemy.spec.reach-t.spec.radius) for t in towers),default=12)/5)
            rows.append([*q, enemy.spec.radius, float(hit), float(pull),
                         max(.5,spec.speed+(enemy.spec.speed if pull else 0)), (1+enemy.value)*urgency,
                         enemy.hp+enemy.shield,enemy.spec.damage/max(.1,enemy.spec.period),
                         min((max(0,np.hypot(q[0]-t.x,q[1]-t.y)-enemy.spec.reach-t.spec.radius)/max(.1,enemy.spec.speed) for t in towers),default=8.)])
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
        contact = clamp(distance-spec.reach-spec.radius-e[None,:,2])/e[None,:,5]
        fight = e[None,:,:2] + (p[:,None,:]-e[None,:,:2])*.5*e[None,:,4:5]
        tower_dps = distance*0
        if cover:
            t = arr(cover)
            d = ((fight[:,:,None,:]-t[None,None,:,:2])**2).sum(axis=-1)**.5
            tower_dps = ((d <= t[None,None,:,2]+e[None,:,None,2])*arr(eligible)[None,:,:]*t[None,None,:,3]).sum(axis=-1)
        value = e[None,:,6]*xp.exp(-contact/2)*(2*e[None,:,3]+e[None,:,4]+tower_dps/100)
        value -= e[None,:,6]*abs(distance-min(4.,spec.reach)*.85)*.15*e[None,:,3]
        if spec.reach > 2:
            value -= e[None,:,6]*clamp(min(spec.reach,4)-distance)*.8*e[None,:,3]*e[None,:,4]
        value -= e[None,:,6]*(1-e[None,:,3])*(1-e[None,:,4])
        if self.trajectory:
            # All positions and observed enemies receive the same 33-step DPS screen.
            # Share firepower rather than allowing a single unit/tower to shoot every target.
            ticks=arr([i*.25 for i in range(33)])[None,None,:]
            own_dps=spec.damage/max(.1,spec.period)*e[None,:,3]/max(1,sum(r[3] for r in rows))
            shared_tower=tower_dps/max(1,len(rows))
            incoming=(e[None,:,8]*e[None,:,4]*xp.exp(-contact/2)).sum(axis=-1)
            lifetime=(spec.hp+spec.shield)/(incoming+.001)
            active=clamp(ticks-contact[:,:,None]-spec.first_hit)
            cap=clamp(lifetime[:,None]-contact-spec.first_hit)[:,:,None]
            active=xp.minimum(active,cap)
            hp=e[None,:,None,7]-own_dps[:,:,None]*active-shared_tower[:,:,None]*ticks
            stall=clamp(lifetime[:,None]-contact)*e[None,:,4]*(contact<e[None,:,9])
            arrival=e[None,:,9]+stall
            tower_damage=((hp>0)*(ticks>=arrival[:,:,None])*e[None,:,None,8]*.25).sum(axis=-1)
            value-=tower_damage/150
        result = value.sum(axis=-1)
        return result.tolist() if self.device == 'cpu' else result.cpu().tolist()
