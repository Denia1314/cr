"""Batched position/enemy/tower ranking; detailed combat remains in Simulator."""
from __future__ import annotations
import numpy as np
from .gpu import resolve_device


class PlacementBatch:
    def __init__(self, requested="auto"):
        self.device = resolve_device(requested)
        self.calls = 0
        self.positions = 0
        self.error = ""
        if self.device != 'cpu':
            from types import SimpleNamespace
            # Warm every ranking operation before the first battle's time budget starts.
            spec = SimpleNamespace(reach=3.,radius=.5)
            self._compute([[1.,20.]], [[2.,18.,.5,1.,1.,2.,1.]],
                          [[3.,28.,8.,100.]], [[1.]], spec)

    def status(self):
        return dict(device=self.device, calls=self.calls, positions=self.positions, error=self.error,
                    scope="placement_ranking", combat_device="cpu")

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
                         max(.5,spec.speed+(enemy.spec.speed if pull else 0)), (1+enemy.value)*urgency])
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
        result = value.sum(axis=-1)
        return result.tolist() if self.device == 'cpu' else result.cpu().tolist()
