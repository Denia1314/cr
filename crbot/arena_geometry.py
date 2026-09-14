"""Explicit, invertible arena calibration; defaults preserve legacy coordinates."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ArenaGeometry:
    bounds: tuple = (.05,.18,.95,.82)
    tower_points: tuple = ((.28,.77),(.72,.77),(.28,.23),(.72,.23))
    bridges: tuple = (4.5,13.5)
    source: str = 'legacy_approximation'

    @classmethod
    def from_config(cls, config):
        if not config:return cls()
        bounds=tuple(map(float,config['bounds']))
        towers=tuple(tuple(map(float,p)) for p in config['tower_points'])
        if len(bounds)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in bounds):raise ValueError('invalid arena bounds')
        if bounds[0]>=bounds[2] or bounds[1]>=bounds[3]:raise ValueError('empty arena bounds')
        if len(towers)!=4 or any(len(p)!=2 or not all(math.isfinite(v) and 0<=v<=1 for v in p) for p in towers):raise ValueError('invalid tower anchors')
        bridges=tuple(map(float,config.get('bridges',[3.5,14.5])))
        if len(bridges)!=2 or not all(0<v<18 for v in bridges):raise ValueError('invalid bridges')
        return cls(bounds,towers,bridges,str(config.get('source','manual_calibration')))

    def xy(self,x,y):
        l,t,r,b=self.bounds
        return (x-l)/(r-l)*18,(y-t)/(b-t)*32

    def screen(self,x,y):
        l,t,r,b=self.bounds
        return round(l+x/18*(r-l),12),round(t+y/32*(b-t),12)
