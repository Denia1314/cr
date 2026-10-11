"""CUDA forward simulation; unsupported mechanics retain the reference simulator.

One CUDA thread owns one independent battle branch. Entity turns and projectile
arrival order remain sequential inside a branch; branches execute in parallel.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
import numpy as np

FIELDS = ('uid','side','x','y','hp','shield','ready_at','expires_at','target','value',
          'tower','walked','locked_at','stunned_until','haste_until','haste',
          'slow_until','slow','active_at','winding_target','active','attack_slow',
          'king','spec_hp','damage','period','first_hit','speed','reach','sight',
          'radius','air','ground_target','air_target','building_only','building',
          'splash','projectile_speed','charge_distance','charge_multiplier',
          'charge_damage','ramp_time0','ramp_time1','ramp_damage0','ramp_damage1',
          'stun','pushback','minimum_range','ignore_pushback')
N, P = 96, 512


class GPUCombat:
    def __init__(self, sim, device='auto', batch_size=128):
        self.sim = sim
        self.batch_size = max(8, min(512, int(batch_size)))
        self.error = ''
        self.available = False
        self.completed = self.calls = self.fallbacks = self.deadline_exits = 0
        self.device = 'cpu'
        self.last_ms = {}
        try:
            from .gpu import resolve_device
            self.device = resolve_device(device)
            if self.device == 'cpu':
                return
            import cupy as cp
            self.cp = cp
            cp.cuda.Device(int(self.device.split(':')[-1])).use()
            definitions = '\n'.join(f'#define {f.upper()} {i}' for i,f in enumerate(FIELDS))
            source = (Path(__file__).with_name('gpu_combat.cu')).read_text(encoding='utf-8')
            self.kernel = cp.RawKernel(definitions+'\n'+source, 'advance_battles',
                                       options=('--std=c++17', '--fmad=false'))
            self.kernel.compile()
            self.route_keys=cp.zeros(65536,cp.uint64)
            self.route_values=cp.full(65536,-3,cp.int32)
            # Warm allocation, launch and download outside the decision budget.
            from .battle_simulation import SimState
            self.available = True
            self.advance([SimState()], .25, time.perf_counter()+10)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.available = False

    def supports(self, state):
        if len(state.entities) > N or len(state.impacts) > P or state.effects:
            return False
        for e in state.entities:
            u = e.spec
            if (u.spawn or u.death_spawn or u.death_damage or u.resource_period
                    or u.resource_on_death or u.dash_damage or u.reflected_damage
                    or u.chain_count > 1 or u.attack_targets > 1 or e.cloned or e.dash_phase):
                return False
        return all(len(i) <= 13 for i in state.impacts)

    @staticmethod
    def pack_entity(e):
        u=e.spec
        return [e.uid,e.side,e.x,e.y,e.hp,e.shield,e.ready_at,e.expires_at,
            -1 if e.target is None else e.target,e.value,e.tower,e.walked,e.locked_at,
            e.stunned_until,e.haste_until,e.haste,e.slow_until,e.slow,e.active_at,
            -1 if e.winding_target is None else e.winding_target,e.active,e.attack_slow,
            e.tower_kind=='king',u.hp,u.damage,u.period,u.first_hit,u.speed,u.reach,u.sight,
            u.radius,u.air,'ground' in u.targets,'air' in u.targets,u.building_only,u.building,
            u.splash,u.projectile_speed,u.charge_distance,u.charge_multiplier,u.charge_damage,
            u.ramp_times[0],u.ramp_times[1],u.ramp_damage[0],u.ramp_damage[1],
            u.stun,u.pushback,u.minimum_range,u.ignore_pushback]

    def advance(self, states, duration, deadline):
        if not states or duration <= 0:
            return
        if time.perf_counter() >= deadline:
            raise TimeoutError('GPU combat deadline before packing')
        began=time.perf_counter()
        b=len(states)
        entities=np.zeros((b,N,len(FIELDS)),np.float64)
        impacts=np.zeros((b,P,13),np.float64)
        meta=np.zeros((b,10),np.float64)
        for k,s in enumerate(states):
            if not self.supports(s):
                raise ValueError('unsupported GPU combat state')
            for i,e in enumerate(s.entities):entities[k,i]=self.pack_entity(e)
            for i,event in enumerate(s.impacts):
                row=list(event)+[0.]*(13-len(event))
                row[2]=-1 if row[2] is None else row[2]
                impacts[k,i]=row
            meta[k]=[s.time,len(s.entities),len(s.impacts),s.elixir[1],s.elixir[-1],
                     s.damage[1],s.damage[-1],s.tower_shots[1],s.tower_shots[-1],s.seconds_per_elixir]
        packed=time.perf_counter()
        cp=self.cp
        de,di,dm=cp.asarray(entities),cp.asarray(impacts),cp.asarray(meta)
        codes=cp.zeros(b,cp.int32)
        left=max(0.,deadline-time.perf_counter())
        if not left:raise TimeoutError('GPU combat deadline after packing')
        bridges=self.sim.geometry.bridges
        # A branch is sequential and route-heavy. One branch per block spreads
        # 128 branches across the GPU instead of occupying only four SMs.
        self.kernel((b,),(1,),
            (de,di,dm,codes,self.route_keys,self.route_values,np.int32(b),np.float64(duration),np.float64(self.sim.step),
             np.int32(bool(getattr(self.sim,'grid_navigation',False))),
             np.float64(bridges[0]),np.float64(bridges[1]),np.uint64(left*1e9)))
        codes=codes.get()
        self.calls+=1
        if np.any(codes==1) or time.perf_counter() >= deadline:
            self.deadline_exits+=1
            raise TimeoutError('GPU combat deadline')
        if np.any(codes):
            raise RuntimeError(f'GPU combat capacity failure: {codes.tolist()}')
        entities,impacts,meta=de.get(),di.get(),dm.get()
        downloaded=time.perf_counter()
        # Publish only complete batches, never partially advanced host states.
        for k,s in enumerate(states):
            m=meta[k]
            s.time=float(m[0]);s.elixir={1:float(m[3]),-1:float(m[4])}
            s.damage={1:float(m[5]),-1:float(m[6])}
            s.tower_shots={1:int(m[7]),-1:int(m[8])}
            for i,e in enumerate(s.entities):
                row=entities[k,i]
                for field in FIELDS[:22]:
                    if field in {'uid','side','value','tower'}:continue
                    value=row[FIELDS.index(field)]
                    if field in {'target','winding_target'}:value=None if value<0 else int(value)
                    elif field=='active':value=bool(value)
                    else:value=float(value)
                    setattr(e,field,value)
            s.entities=[e for e in s.entities if e.hp>0]
            s.impacts=[]
            for row in impacts[k,:int(m[2])]:
                s.impacts.append((float(row[0]),int(row[1]),None if row[2]<0 else int(row[2]),
                    *map(float,row[3:7]),bool(row[7]),*map(float,row[8:12]),int(row[12])))
        self.last_ms=dict(pack=round((packed-began)*1000,3),kernel_and_transfer=round((downloaded-packed)*1000,3),
                          unpack=round((time.perf_counter()-downloaded)*1000,3),branches=b)

    def rows(self, planner, world, roots, scenarios, horizon, phase, deadline, initial_states, *, stop_at=None):
        """Batch complete root branches; fallback retains all special mechanics."""
        from .battle_simulation import SimAction
        from .tactical_objective import score_action
        self.sim.tactical_phase=phase
        pipeline=max(0.,min(1.,float(planner.config.get('defense_pipeline_s',0))))
        fast=bool(planner.config.get('fast_defense',False))
        if not all(self.supports(s) for s in initial_states):
            for root in roots:
                self.fallbacks+=1
                yield planner.evaluate_root(world,root,scenarios,horizon,phase,deadline,initial_states)
            return
        bases=[s.clone() for s in initial_states]
        try:
            self.advance(bases,pipeline,deadline)
        except RuntimeError as exc:
            self.error=str(exc);self.available=False
            for root in roots:
                self.fallbacks+=1
                yield planner.evaluate_root(world,root,scenarios,horizon,phase,deadline,initial_states)
            return
        roots_per_batch=max(8,self.batch_size//max(1,len(scenarios)))
        for start in range(0,len(roots),roots_per_batch):
            if planner.clock()>=deadline:raise TimeoutError()
            batch=roots[start:start+roots_per_batch]
            work=[]; owners=[]; fallback=[]
            for j,root in enumerate(batch):
                states=[s.clone() for s in bases]
                # Unsupported roots are never silently simplified.
                if not all(self.sim.apply(s,root,1) and self.supports(s) for s in states):
                    fallback.append((j,root));continue
                work.extend(states);owners.extend((j,i) for i in range(len(states)))
            try:
                responses=[]
                supported=[];supported_owners=[]
                no_response=not any(s[2] for s in scenarios) and not (fast and phase in {'develop','counterpush','prepare'})
                if no_response:
                    self.advance(work,horizon-pipeline,deadline)
                    supported,supported_owners=work,owners
                    responses=[SimAction() for _ in work]
                else:
                    self.advance(work,1.5,deadline)
                for state,(j,i) in zip(work,owners):
                    if no_response:break
                    root=batch[j]
                    response=(planner.attack_response(state) if fast and phase in {'develop','counterpush','prepare'} and root.card_id
                              else planner.prior_response(state) if scenarios[i][2] else SimAction())
                    self.sim.apply(state,response,-1)
                    if self.supports(state):
                        supported.append(state);supported_owners.append((j,i));responses.append(response)
                    elif all(x[0]!=j for x in fallback):fallback.append((j,root))
                if not no_response:self.advance(supported,horizon-1.5-pipeline,deadline)
                groups={}
                for state,(j,i),response in zip(supported,supported_owners,responses):
                    if any(x[0]==j for x in fallback):continue
                    root=batch[j];score,parts=self.sim.evaluate(state)
                    if phase!='legacy':score=score_action(score,parts,world,planner.kb,root,phase,float(planner.config.get('attack_reserve',3)),self.sim.geometry)
                    groups.setdefault(j,[]).append((i,score,dict(enemy_elixir_assumption=scenarios[i][0],enemy_response=response.label,
                        own_followup='WAIT',pipeline_delay_s=pipeline,score=round(score,4),**parts)))
                for j in sorted(groups):
                    branches=sorted(groups[j]);values=[v[1] for v in branches]
                    if len(values)!=len(scenarios):continue
                    risk=float(planner.config.get('downside_weight',.65))
                    self.completed+=1
                    yield dict(action=asdict(batch[j]),label=batch[j].label,
                        score=round((1-risk)*sum(values)/len(values)+risk*min(values),4),branches=[v[2] for v in branches])
            except (RuntimeError,ValueError) as exc:
                self.error=str(exc);self.available=False
                fallback=list(enumerate(batch))
            fallback_roots=[root for _,root in fallback]
            pool=planner.combat_pool
            if fallback_roots and pool is not None and pool.available and all(f.done() for f in pool.pending):
                cpu_rows=pool.rows(world,fallback_roots,scenarios,horizon,phase,deadline,completion_order=True)
            else:
                cpu_rows=(planner.evaluate_root(world,root,scenarios,horizon,phase,deadline,initial_states) for root in fallback_roots)
            for row in cpu_rows:
                if row is not None:self.fallbacks+=1
                yield row
            if stop_at is not None and planner.clock()>=stop_at:
                return
            if not self.available:
                for root in roots[start+len(batch):]:
                    self.fallbacks+=1
                    yield planner.evaluate_root(world,root,scenarios,horizon,phase,deadline,initial_states)
                return

    def status(self):
        return dict(device=self.device,available=self.available,error=self.error,batch_size=self.batch_size,
                    completed_roots=self.completed,kernel_calls=self.calls,cpu_fallback_roots=self.fallbacks,
                    deadline_exits=self.deadline_exits,last_batch_ms=self.last_ms,scope='detailed_combat_supported_mechanics')
