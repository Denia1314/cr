"""Local evidence workbench. Synthetic invariants are not battle acceptance."""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import asdict
from .knowledge import KnowledgeBase
from .cards import CardCatalog
from .learning import audit_learning_data
from .battle_world import Track,WorldSnapshot
from .predictive_planner import PredictivePlanner


def coverage(kb):
    result={}
    for cid,card in kb.cards.items():
        try:
            roster=kb.roster(cid,11)
            spell=kb.spell(cid,11)
            unknown=list(card.get('unsupported',[]))
            for unit,_ in roster:unknown.extend(unit.unsupported)
            result[cid]=dict(status='approximate' if roster or spell else 'missing',
                unknown=sorted(set(unknown)),roles=card.get('roles',[]),
                pattern=(spell or {}).get('pattern','direct' if spell else None),
                variants=len(card.get('variants',[])),verified_game_cases=0)
        except (ValueError,KeyError,TypeError) as exc:
            result[cid]=dict(status='missing',error=str(exc),verified_game_cases=0)
    return result


def synthetic_scenarios():
    scenarios=[]
    for cid in ['giant','hog_rider','balloon','musketeer','knight','mini_pekka',
                'baby_dragon','minions','valkyrie','witch','giant_skeleton','pekka']:
        for lane,x in [('left',.24),('right',.76)]:
            scenarios.append(dict(name=f'{cid}_{lane}',enemies=[dict(card_id=cid,x=x,y=.49,hp_fraction=1)],elixir=7))
    scenarios.extend([
        dict(name='low_health_cleanup',enemies=[dict(card_id='knight',x=.24,y=.55,hp_fraction=.01)],elixir=7),
        dict(name='dual_lane',enemies=[dict(card_id='hog_rider',x=x,y=.49,hp_fraction=1) for x in (.24,.76)],elixir=7),
        dict(name='zero_elixir',enemies=[dict(card_id='giant',x=.24,y=.49,hp_fraction=1)],elixir=0),
        dict(name='quiet_development',enemies=[],elixir=10),
        dict(name='reserve_resources',enemies=[],elixir=2),
        dict(name='critical_tower',enemies=[dict(card_id='hog_rider',x=.24,y=.49,hp_fraction=1)],elixir=7,tower_health=[.03,1,1,1])])
    return scenarios


def review_queue(root,limit=120):
    queue=[]
    paths=sorted((root/'runs').glob('*/events.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)[:12]
    for path in paths:
        last_frame=None
        samples=[]
        sample_path=path.parent/'samples.jsonl'
        if sample_path.exists():
            for line in sample_path.read_text(encoding='utf-8').splitlines():
                try:samples.append(json.loads(line))
                except ValueError:continue
        with path.open(encoding='utf-8') as handle:
            for line_number,line in enumerate(handle,1):
                try:e=json.loads(line)
                except ValueError:continue
                if e.get('frame'):last_frame=e['frame']
                if e.get('event')!='battle_prediction':continue
                plan=e.get('plan') or {}
                tags=[]
                if plan.get('perception_mode')=='lane_hypotheses':tags.append('identity_unverified')
                if plan.get('fallback_reason'):tags.append(plan['fallback_reason'])
                if plan.get('compute',{}).get('combat_complete') is False:tags.append('incomplete_combat_search')
                if not tags:continue
                # Limit repeated adjacent evidence; no invented ground truth or failure label.
                if queue and queue[-1]['run']==path.parent.name and e.get('timestamp_unix',0)-queue[-1]['at']<5:continue
                nearest=min(samples,key=lambda s:abs(s.get('timestamp_unix',0)-e.get('timestamp_unix',0)),default=None)
                if nearest and abs(nearest.get('timestamp_unix',0)-e.get('timestamp_unix',0))<=3:
                    last_frame=nearest.get('frame')
                queue.append(dict(run=path.parent.name,line=line_number,at=e.get('timestamp_unix',0),
                                  frame=last_frame,tags=tags,review_status='needs_human_review',
                                  true_card=None,true_position=None,true_health=None,failure_cause=None))
                if len(queue)>=limit:return queue
    return queue


def prepare(root,config):
    out=root/'reports'/'battle_lab';out.mkdir(parents=True,exist_ok=True)
    kb=KnowledgeBase.load(root/config.get('prediction',{}).get('knowledge_path','data/battle_knowledge.json'))
    audit=audit_learning_data(root,CardCatalog.load(root/'data/cards.json'),config.get('training',{})).to_dict()
    payload=dict(schema=1,knowledge_version=kb.version,mechanisms=coverage(kb),training=audit,
                 review_queue=review_queue(root),scenarios=synthetic_scenarios(),battle_acceptance=False)
    (out/'manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Battle evidence workbench','',f"Knowledge: {kb.version}",
        f"Human frames: {audit['human_frames']}; boxes: {audit['human_boxes']}",
        f"Review cases: {len(payload['review_queue'])}; synthetic scenarios: {len(payload['scenarios'])}",
        '', 'No generated case is human ground truth. No mechanism is marked game-validated.',
        'Use the existing `annotate` command to label source battle frames.',
        '', '| Card | Status | Unverified mechanisms |','|---|---|---|']
    for cid,v in payload['mechanisms'].items():lines.append(f"| {cid} | {v['status']} | {', '.join(v.get('unknown',[]))} |")
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(report=str(out/'README.md'),manifest=str(out/'manifest.json'),training=audit,
                review_cases=len(payload['review_queue']),battle_acceptance=False)


def evaluate(root,config):
    kb=KnowledgeBase.load(root/config.get('prediction',{}).get('knowledge_path','data/battle_knowledge.json'))
    planner=PredictivePlanner(kb,config.get('prediction',{}))
    rows=[]
    for case in synthetic_scenarios():
        tracks=tuple(Track(i+1,e['card_id'],-1,e['x'],e['y'],99.,100.,.99,hp_fraction=e['hp_fraction'],variant='base') for i,e in enumerate(case['enemies']))
        world=WorldSnapshot(1,100.,20.,case['elixir'],(0.,7.),((0,'knight'),(1,'musketeer'),(2,'cannon'),(3,'fireball')),
                            tracks,(),(),(),tower_health=tuple(case.get('tower_health',[1]*4)))
        result=planner.plan(world)
        state=planner.initial(world,7,1)
        legal=result.action.card_id is None or (kb.cards[result.action.card_id]['elixir']<=world.elixir and planner.sim.legal_placement(state,result.action.card_id,1,result.action.x,result.action.y))
        rows.append(dict(name=case['name'],status=result.status,invariants_passed=legal,plan=result.to_dict()))
    out=root/'reports'/'battle_lab';out.mkdir(parents=True,exist_ok=True)
    payload=dict(reference='synthetic_invariants_only',cases=rows,battle_acceptance=False)
    (out/'evaluation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return dict(cases=len(rows),invariants_passed=all(r['invariants_passed'] for r in rows),
                actionable=sum(r['status'] in ('ready','wait') for r in rows),report=str(out/'evaluation.json'),battle_acceptance=False)


def render_calibration(reference,config,out):
    import base64
    from PIL import Image
    from .arena_geometry import ArenaGeometry
    image=Image.open(reference)
    w,h=image.size
    geom=ArenaGeometry.from_config(config.get('prediction',{}).get('arena_geometry'))
    mime='image/png' if image.format=='PNG' else 'image/jpeg'
    data=base64.b64encode(Path(reference).read_bytes()).decode('ascii')
    elements=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
              f'<image width="{w}" height="{h}" href="data:{mime};base64,{data}"/>']
    for x in range(19):
        a,b=geom.screen(x,0),geom.screen(x,32)
        elements.append(f'<path d="M {a[0]*w} {a[1]*h} L {b[0]*w} {b[1]*h}" stroke="#00ffcc" stroke-opacity=".35"/>')
    for y in range(33):
        a,b=geom.screen(0,y),geom.screen(18,y)
        elements.append(f'<path d="M {a[0]*w} {a[1]*h} L {b[0]*w} {b[1]*h}" stroke="#00ffcc" stroke-width="{3 if y==16 else 1}" stroke-opacity=".5"/>')
    for x,y in geom.tower_points:elements.append(f'<circle cx="{x*w}" cy="{y*h}" r="9" stroke="lime" fill="none" stroke-width="3"/>')
    for x,y in ArenaGeometry().tower_points:elements.append(f'<circle cx="{x*w}" cy="{y*h}" r="9" stroke="red" fill="none" stroke-dasharray="4 3"/>')
    elements.append('</svg>')
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(''.join(elements),encoding='utf-8')
    return str(out)
