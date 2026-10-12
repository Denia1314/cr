"""CUDA forward simulation; unsupported mechanics retain the reference simulator.

One CUDA thread owns one independent battle branch. Entity turns and projectile
arrival order remain sequential inside a branch; branches execute in parallel.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .gpu_combat_codec import CombatCodec, FIELDS, EFFECT_FIELDS, PATTERNS, INDEX

N, P, Q = 112, 512, 512


class GPUCombat:
    def __init__(self, sim, device='auto', batch_size=128):
        self.sim = sim
        self.codec=CombatCodec(sim)
        self.batch_size = max(8, min(512, int(batch_size)))
        self.error = ''
        self.available = False
        self.completed = self.calls = self.fallbacks = self.deadline_exits = 0
        self.device = 'cpu'
        self.last_ms = {}
        self.completed_combinations=0
        try:
            from .gpu import resolve_device
            self.device = resolve_device(device)
            if self.device == 'cpu':
                return
            import cupy as cp
            self.cp = cp
            cp.cuda.Device(int(self.device.split(':')[-1])).use()
            definitions = '\n'.join(f'#define {f.upper()} {i}' for i,f in enumerate(FIELDS))
            definitions+='\n'+f'#define NF {len(FIELDS)}\n#define NE {N}\n#define NI {P}\n#define NQ {Q}\n#define EF {len(EFFECT_FIELDS)}\n'
            definitions+='\n'.join(f'#define FX_{f.upper()} {i}' for i,f in enumerate(EFFECT_FIELDS))
            definitions+='\n'+'\n'.join(f'#define PAT_{name.upper()} {i}' for i,name in enumerate(PATTERNS))
            source = (Path(__file__).with_name('gpu_combat.cu')).read_text(encoding='utf-8')
            mechanics=(Path(__file__).with_name('gpu_combat_mechanics.cuh')).read_text(encoding='utf-8')
            source=source.replace('// EXTENDED_MECHANICS',mechanics)
            self.kernel = cp.RawKernel(definitions+'\n'+source, 'advance_battles',
                                       options=('--std=c++17', '--fmad=false'))
            self.kernel.compile()
            self.route_keys=cp.zeros(65536,cp.uint64)
            self.route_values=cp.full(65536,-3,cp.int32)
            # Warm allocation, launch and download outside the decision budget.
            from .battle_simulation import SimState
            self.available = True
            self.advance([SimState()], .25, time.perf_counter()+10)
            # Exercise route-heavy batches before a short live frame deadline;
            # an empty single branch does not warm CUDA local-memory capacity.
            knight=self.sim.kb.unit('Knight',self.sim.level)
            tower=self.sim.kb.unit('PrincessTower',self.sim.level)
            if knight is not None and tower is not None:
                warm=SimState()
                self.sim.add(warm,knight,1,12.,23.)
                self.sim.add(warm,tower,-1,3.,4.,tower=True)
                self.advance([warm.clone() for _ in range(self.batch_size)],.5,time.perf_counter()+10)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.available = False

    def supports(self, state):
        return not self.unsupported_reasons(state)

    def unsupported_reasons(self, state):
        reasons=set()
        if len(state.entities) > N or len(state.impacts) > P or len(state.effects)>Q:
            if len(state.entities)>N:reasons.add('entity_capacity')
            if len(state.impacts)>P:reasons.add('projectile_capacity')
            if len(state.effects)>Q:reasons.add('effect_capacity')
        for e in state.entities:
            u = e.spec
            for name,active in (('chain_capacity',u.chain_count>32),('target_capacity',u.attack_targets>32)):
                if active:reasons.add(f'{u.name}:{name}')
        if any(e.get('pattern') not in PATTERNS for e in state.effects):reasons.add('unknown_effect_pattern')
        if any(len(i)>13 for i in state.impacts):reasons.add('projectile_format')
        return sorted(reasons)

    def pack_entity(self,e):
        return self.codec.entity_row(e)

    def advance(self, states, duration, deadline, *, effect_capacity=None):
        if not states or duration <= 0:
            return
        if time.perf_counter() >= deadline:
            raise TimeoutError('GPU combat deadline before packing')
        began=time.perf_counter()
        b=len(states)
        for s in states:
            for e in s.entities:self.codec.ensure(e.spec)
            for event in s.effects:
                spec=self.sim.kb.unit(event.get('spawn'),self.sim.level) if event.get('spawn') else None
                if spec:self.codec.ensure(spec)
        sources=list(self.codec.pulse_sources)
        ids={self.codec.ensure(e.spec) for s in states for e in s.entities}
        growing=any(self.codec.rows[i][INDEX['spawn_spec']]>=0 or self.codec.rows[i][INDEX['death_spec']]>=0 for i in ids)
        growing |= any(event.get('spawn') or event.get('pattern')=='clone' for s in states for event in s.effects)
        entity_capacity=N if growing else min(N,max(8,1<<(max(len(s.entities) for s in states)-1).bit_length()))
        # Empty queues need no 512-event upload. Any possible spawn pulse retains
        # room; exceptionally large queues retry atomically with full capacity.
        active=max(len(s.effects) for s in states)
        has_pulses=bool(self.codec.pulse_sources) and growing
        if effect_capacity is None:effect_capacity=min(Q,max(64 if has_pulses else 1,1<<max(0,(active-1).bit_length())))
        entities=np.zeros((b,entity_capacity,len(FIELDS)),np.float64)
        impacts=np.zeros((b,P,13),np.float64)
        effects=np.zeros((b,effect_capacity,len(EFFECT_FIELDS)),np.float64)
        meta=np.zeros((b,13),np.float64)
        for k,s in enumerate(states):
            if not self.supports(s):
                raise ValueError('unsupported GPU combat state')
            for i,e in enumerate(s.entities):entities[k,i]=self.pack_entity(e)
            for i,event in enumerate(s.impacts):
                row=list(event)+[0.]*(13-len(event))
                row[2]=-1 if row[2] is None else row[2]
                impacts[k,i]=row
            for i,event in enumerate(s.effects):
                source=len(sources);sources.append(event)
                effects[k,i]=self.codec.effect_row(event,source)
            meta[k]=[s.time,len(s.entities),len(s.impacts),s.elixir[1],s.elixir[-1],
                     s.damage[1],s.damage[-1],s.tower_shots[1],s.tower_shots[-1],s.seconds_per_elixir,0.,s.next_uid,len(s.effects)]
        packed=time.perf_counter()
        cp=self.cp
        de,di,dm=cp.asarray(entities),cp.asarray(impacts),cp.asarray(meta)
        df=cp.asarray(effects);scratch=cp.empty_like(df)
        prototypes=cp.asarray(np.asarray(self.codec.rows,dtype=np.float64).reshape((-1,len(FIELDS))))
        pulses=cp.asarray(np.asarray(self.codec.pulse_rows,dtype=np.float64).reshape((-1,len(EFFECT_FIELDS))))
        spawned=cp.zeros((b,len(self.codec.specs)),cp.uint8)
        codes=cp.zeros(b,cp.int32)
        left=max(0.,deadline-time.perf_counter())
        if not left:raise TimeoutError('GPU combat deadline after packing')
        bridges=self.sim.geometry.bridges
        # A branch is sequential and route-heavy. One branch per block spreads
        # 128 branches across the GPU instead of occupying only four SMs.
        self.kernel((b,),(1,),
            (de,di,dm,df,scratch,prototypes,pulses,spawned,np.int32(len(self.codec.specs)),np.int32(entity_capacity),np.int32(effect_capacity),codes,self.route_keys,self.route_values,np.int32(b),np.float64(duration),np.float64(self.sim.step),
             np.int32(bool(getattr(self.sim,'grid_navigation',False))),
             np.float64(bridges[0]),np.float64(bridges[1]),np.uint64(left*1e9)))
        codes=codes.get()
        self.calls+=1
        if np.any(codes==1) or time.perf_counter() >= deadline:
            self.deadline_exits+=1
            raise TimeoutError('GPU combat deadline')
        if np.any(codes):
            if np.all((codes==0)|(codes==4)) and effect_capacity<Q:
                return self.advance(states,duration,deadline,effect_capacity=Q)
            raise RuntimeError(f'GPU combat capacity failure: {codes.tolist()}')
        entities,impacts,meta=de.get(),di.get(),dm.get()
        effects,spawned=df.get(),spawned.get()
        downloaded=time.perf_counter()
        # Publish only complete batches, never partially advanced host states.
        for k,s in enumerate(states):
            m=meta[k]
            s.time=float(m[0]);s.elixir={1:float(m[3]),-1:float(m[4])}
            s.damage={1:float(m[5]),-1:float(m[6])}
            s.tower_shots={1:int(m[7]),-1:int(m[8])}
            for bit,name in enumerate(('chain_travel_time_estimate','dash_timing_and_level_scaling_estimate',
                'retaliation_source_and_level_scaling_estimate','entity_budget')):
                if int(m[10])&(1<<bit):s.uncertainties.add(name)
            for index,flag in enumerate(spawned[k]):
                if flag:s.uncertainties.update(self.codec.specs[index].unsupported)
            existing={e.uid:e for e in s.entities}
            s.entities=[self.codec.entity(row,existing) for row in entities[k,:int(m[1])]]
            s.next_uid=int(m[11])
            s.effects=[self.codec.effect(row,sources) for row in effects[k,:int(m[12])]]
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
        # Commit a useful comparison before spending a short frame's remaining
        # time preparing a large batch. Hypotheses count toward this first wave.
        roots_per_batch=max(8,min(32,self.batch_size)//max(1,len(scenarios)))
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
                    deadline_exits=self.deadline_exits,last_batch_ms=self.last_ms,completed_combinations=self.completed_combinations,
                    scope='full_reference_combat')

    def combination_rows(self,planner,world,roots,scenarios,horizon,phase,deadline):
        completed=set()
        try:
            for row in self._combination_rows(planner,world,roots,scenarios,horizon,phase,deadline):
                completed.add(row['label'])
                yield row
        except (RuntimeError,ValueError) as exc:
            self.error=str(exc);self.available=False
            for root in roots:
                if root.label not in completed:
                    yield planner.evaluate_combination(world,root,scenarios,horizon,phase,deadline)

    def _combination_rows(self,planner,world,roots,scenarios,horizon,phase,deadline):
        """All conditional two-card forward phases execute in CUDA batches."""
        from .battle_simulation import SimAction
        from .placement_search import placement_points
        from .tactical_objective import score_action
        self.sim.tactical_phase=phase
        positions=max(1,min(4,int(planner.config.get('combo_positions_per_card',2))))
        pipeline=max(0.,min(1.,float(planner.config.get('defense_pipeline_s',0))))
        bases=[planner.initial(world,c,h) for c,h,_ in scenarios]
        self.advance(bases,pipeline,deadline)
        for root in roots:
            if planner.clock()>=deadline:raise TimeoutError()
            states=[s.clone() for s in bases]
            if not all(self.sim.apply(s,root,1) for s in states):raise ValueError('invalid tactical root')
            self.advance(states,1.5,deadline)
            trials=[];owners=[];responses=[];counts=[]
            for i,state in enumerate(states):
                response=planner.attack_response(state) if phase in {'develop','counterpush','prepare'} and root.card_id else SimAction()
                self.sim.apply(state,response,-1);responses.append(response)
                followups=[SimAction()]
                for slot,cid in state.hands[1]:
                    cost=planner.kb.cards[cid].get('elixir')
                    if cost is None or cost>state.elixir[1]:continue
                    followups.extend(SimAction(cid,slot,x,y) for x,y in placement_points(self.sim,state,cid,1,limit=positions))
                counts.append(len(followups))
                for follow in followups:
                    if planner.clock()>=deadline:raise TimeoutError()
                    trial=state.clone()
                    if not self.sim.apply(trial,follow,1):continue
                    second=SimAction()
                    if follow.card_id and phase in {'develop','counterpush','prepare'}:
                        second=planner.attack_response(trial);self.sim.apply(trial,second,-1)
                    trials.append(trial);owners.append((i,follow,second))
            for start in range(0,len(trials),self.batch_size):
                self.advance(trials[start:start+self.batch_size],max(0.,horizon-1.5),deadline)
            alternatives={i:[] for i in range(len(states))}
            for trial,(i,follow,second) in zip(trials,owners):
                score,parts=self.sim.evaluate(trial)
                score=score_action(score,parts,world,planner.kb,root,phase,float(planner.config.get('attack_reserve',3)),self.sim.geometry)
                spent=sum(float(planner.kb.cards[c]['elixir']) for c in (root.card_id,follow.card_id) if c)
                if phase!='defend':
                    reserve=float(planner.config.get('attack_reserve',3))
                    root_cost=float(planner.kb.cards[root.card_id]['elixir']) if root.card_id else 0
                    score-=(max(0,reserve-world.elixir+spent)-max(0,reserve-world.elixir+root_cost))*1.5
                alternatives[i].append((score,parts,follow,spent,second))
            outcomes=[];costs=[];branches=[]
            for i,options in alternatives.items():
                if not options:raise ValueError('no complete followup')
                best=max(options,key=lambda v:v[0])
                if phase=='defend':
                    towers=sum(e.side==1 and e.tower for e in states[i].entities)
                    def damage(v):return 100000*(towers-v[1]['own_towers_remaining'])+v[1]['own_tower_damage']+v[1].get('imminent_tower_exposure',0)
                    least=min(map(damage,options));tolerance=float(planner.config.get('defense_damage_tolerance',30))
                    best=min((v for v in options if damage(v)<=least+tolerance),key=lambda v:(v[3],damage(v),-v[0]))
                score,parts,follow,spent,second=best;outcomes.append(score);costs.append(spent)
                branches.append(dict(enemy_elixir_assumption=scenarios[i][0],enemy_response=responses[i].label,
                    own_followup=follow.label,followup_action=asdict(follow),followup_after_s=1.5+pipeline,pipeline_delay_s=pipeline,
                    followup_candidates=counts[i],enemy_followup_response=second.label,score=round(score,4),**parts))
            risk=float(planner.config.get('downside_weight',.65));self.completed_combinations+=1
            yield dict(action=asdict(root),label=root.label,score=round((1-risk)*sum(outcomes)/len(outcomes)+risk*min(outcomes),4),
                branches=branches,planned_cost=max(costs),simulation_nodes=len(trials),scope='conditional_two_card_dynamic_positions')
