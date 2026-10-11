"""Offline detailed-root throughput; never connects to a game or sends actions."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--budget-ms', type=float, default=160)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--gpu-combat', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--gpu-batch', type=int, default=64)
    args = parser.parse_args()
    from crbot.battle_world import Track, WorldSnapshot
    from crbot.knowledge import KnowledgeBase
    from crbot.predictive_planner import PredictivePlanner
    config = json.loads((ROOT/'config.json').read_text(encoding='utf-8')).get('prediction', {})
    config = {**config, 'budget_ms': args.budget_ms, 'adaptive_defense_budget': False}
    if args.workers is not None:
        config['combat_workers'] = args.workers
    if args.gpu_combat is not None:
        config['gpu_combat'] = args.gpu_combat
    config['gpu_combat_batch_size']=args.gpu_batch
    planner = PredictivePlanner(KnowledgeBase.load(ROOT/'data/battle_knowledge.json'), config)
    planner.warm_pool()
    base = WorldSnapshot(1,100,20,10,(0,7),((0,'knight'),(1,'musketeer'),(2,'cannon'),(3,'fireball')),
        (Track(1,'giant',-1,.28,.5,99,100,.9,3),),('giant',),(),())
    cases = [('known',base), ('uncertain',replace(base,tracks=(
        Track(1,'unknown:left:heavy',-1,.28,.5,99,100,.9,3,hypotheses=('giant','hog_rider','knight')),)))]
    results=[]
    try:
        for name, snapshot in cases:
            rows=[]
            for i in range(args.samples+2):
                result=planner.plan(replace(snapshot,revision=i+1))
                rows.append(dict(elapsed_ms=result.elapsed_ms,completed_roots=len(result.candidates),
                    combat_evaluated=result.compute.get('combat_evaluated',0),
                    spatial_scored=result.compute.get('spatial_scored',0),
                    by_card={e['card_id']:e['evaluated'] for e in result.hand_evaluations},
                    stage_ms=result.compute.get('stage_ms'),status=result.status,
                    gpu_combat=result.compute.get('gpu_combat')))
                # Let obsolete jobs exit rather than charging them to next frame.
                pending=getattr(planner.combat_pool,'pending',[]) if planner.combat_pool else []
                while any(not f.done() for f in pending):time.sleep(.005)
            rows=rows[2:]
            record=dict(scene=name,budget_ms=args.budget_ms,
                median_completed=statistics.median(r['completed_roots'] for r in rows),
                min_completed=min(r['completed_roots'] for r in rows),
                max_completed=max(r['completed_roots'] for r in rows),samples=rows)
            results.append(record)
            print(json.dumps({k:v for k,v in record.items() if k!='samples'}),flush=True)
    finally:
        planner.close()
    report=ROOT/'reports/planner-speed/throughput.json'
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
