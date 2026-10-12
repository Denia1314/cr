"""Lossless host/device packets for the reference simulator's mutable state."""
from __future__ import annotations
from .battle_simulation import Entity
from .unit_mechanics import pulse

FIELDS = ('uid','side','x','y','hp','shield','ready_at','expires_at','target','value',
 'tower','walked','locked_at','stunned_until','haste_until','haste','slow_until','slow',
 'active_at','winding_target','active','attack_slow','king','spec_hp','damage','period',
 'first_hit','speed','reach','sight','radius','air','ground_target','air_target',
 'building_only','building','splash','projectile_speed','charge_distance','charge_multiplier',
 'charge_damage','ramp_time0','ramp_time1','ramp_damage0','ramp_damage1','stun','pushback',
 'minimum_range','ignore_pushback','death_damage','death_radius','death_tower_multiplier',
 'death_delay','death_air','chain_count','chain_radius','attack_targets',
 'resource_at','spawn_at','cloned','dash_phase','dash_target','dash_at','dash_ready_at','dash_x','dash_y',
 'resource_period','resource_amount','resource_on_death','spawn_spec','spawn_count','spawn_period',
 'spawn_start','death_spec','death_count','deploy','lifetime','dash_damage','dash_min','dash_max',
 'dash_windup','dash_travel','dash_speed','dash_recovery','dash_radius','dash_pushback','dash_immune',
 'reflected_damage','reflected_radius','reflected_stun','pulse_template','prototype','has_spawn')
INDEX={name:i for i,name in enumerate(FIELDS)}
MUTABLE=FIELDS[:22]+FIELDS[57:66]
MUTABLE_INDEX=[INDEX[f] for f in MUTABLE]
PATTERNS=('unit_pulse','spawn_entity','waves','highest_hp','periodic','line','line_spawn',
          'area_spawn','spawn','serial_spawn','rage','clone','pull','earthquake')
EFFECT_FIELDS=('pattern','at','side','x','y','owner','damage','radius','air','ground',
 'tower_multiplier','stun','pushback','slow','attack_slow','duration','period','pulses',
 'friendly','length','width','speed','boost','building_multiplier','spawn_spec',
 'spawn_count','spawn_period','summon_value','source')
EFFECT_INDEX={name:i for i,name in enumerate(EFFECT_FIELDS)}
DEFAULTS=dict(owner=-1,air=True,ground=True,slow=0.,attack_slow=1.,period=1.,pulses=1,
              tower_multiplier=1.,building_multiplier=1.,spawn_spec=-1)
DASH={'':0,'windup':1,'flight':2}


class CombatCodec:
    def __init__(self, sim):
        self.sim=sim
        self.specs=[];self.spec_ids={};self.rows=[];self.pulse_sources=[];self.pulse_rows=[]
        self.identity_ids={}
        self.identity_specs={}

    def ensure(self, spec):
        identity=id(spec)
        if identity in self.identity_ids:return self.identity_ids[identity]
        if spec in self.spec_ids:
            idx=self.spec_ids[spec];self.identity_ids[identity]=idx;self.identity_specs[identity]=spec;return idx
        idx=len(self.specs);self.spec_ids[spec]=idx;self.specs.append(spec);self.rows.append(None)
        self.identity_ids[identity]=idx
        self.identity_specs[identity]=spec
        spawn=self.sim.kb.unit(spec.spawn,self.sim.level) if spec.spawn else None
        death=self.sim.kb.unit(spec.death_spawn,self.sim.level) if spec.death_spawn else None
        spawn_id=self.ensure(spawn) if spawn else -1
        death_id=self.ensure(death) if death else -1
        raw=self.sim.kb.units.get(spec.name,{})
        event=pulse(self.sim.kb,self.sim.kb.effects.get(raw.get('spawn_area_object')),self.sim.level)
        pulse_id=-1
        if event:
            pulse_id=len(self.pulse_rows)
            event=dict(event,pattern='unit_pulse',at=0.,side=0,x=0.,y=0.,owner=-1,mechanism='unit_spawn_area')
            self.pulse_sources.append(event)
            self.pulse_rows.append(self.effect_row(event,pulse_id))
        e=Entity(0,spec,0,0.,0.,spec.hp,spec.shield)
        self.rows[idx]=self.entity_row(e,idx,spawn_id,death_id,pulse_id)
        return idx

    def entity_row(self,e,idx=None,spawn_id=None,death_id=None,pulse_id=None):
        u=e.spec
        if idx is None:
            idx=self.ensure(u)
            row=list(self.rows[idx])
        else:
            values=dict(spec_hp=u.hp,ground_target='ground' in u.targets,air_target='air' in u.targets,
                spawn_spec=spawn_id,death_spec=death_id,pulse_template=pulse_id,prototype=idx,has_spawn=bool(u.spawn),
                ramp_time0=u.ramp_times[0],ramp_time1=u.ramp_times[1],ramp_damage0=u.ramp_damage[0],ramp_damage1=u.ramp_damage[1])
            row=[float(values.get(f,getattr(u,f,0.))) for f in FIELDS]
        for f in MUTABLE:
            v=getattr(e,f)
            if f in ('target','winding_target','dash_target'):v=-1 if v is None else v
            elif f=='dash_phase':v=DASH[v]
            row[INDEX[f]]=float(v)
        row[INDEX['king']]=float(e.tower_kind=='king')
        return row

    def effect_row(self,event,source):
        row=[]
        for f in EFFECT_FIELDS:
            v=event.get(f,DEFAULTS.get(f,0.))
            if f=='pattern':v=PATTERNS.index(event['pattern'])
            elif f=='owner':v=-1 if v is None else v
            elif f=='source':v=source
            elif f=='spawn_spec':
                spec=self.sim.kb.unit(event.get('spawn'),self.sim.level) if event.get('spawn') else None
                v=self.ensure(spec) if spec is not None else -1
            row.append(float(v))
        return row

    def entity(self,row,existing):
        uid=int(row[INDEX['uid']]);spec=self.specs[int(row[INDEX['prototype']])]
        e=existing.get(uid) or Entity(uid,spec,int(row[INDEX['side']]),0.,0.,0.)
        values=dict(zip(MUTABLE,row[MUTABLE_INDEX].tolist()))
        for f in ('uid','side'):values[f]=int(values[f])
        for f in ('target','winding_target','dash_target'):values[f]=None if values[f]<0 else int(values[f])
        for f in ('tower','active','cloned'):values[f]=bool(values[f])
        values['dash_phase']=('', 'windup','flight')[int(values['dash_phase'])]
        e.__dict__.update(values)
        e.tower_kind='king' if row[INDEX['king']] else e.tower_kind
        return e

    def effect(self,row,sources):
        event=dict(sources[int(row[EFFECT_INDEX['source']])])
        event['pattern']=PATTERNS[int(row[0])]
        for f in EFFECT_FIELDS[1:]:
            if f in ('source','spawn_spec'):continue
            v=row[EFFECT_INDEX[f]]
            if f=='owner':
                if v<0:event.pop('owner',None);continue
                v=int(v)
            elif f in ('side','pulses','spawn_count'):v=int(v)
            elif f in ('air','ground','friendly'):v=bool(v)
            else:v=float(v)
            if f in event or f in ('at','side','x','y','owner'):event[f]=v
        return event
