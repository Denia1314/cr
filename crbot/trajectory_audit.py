"""Compare predicted track positions with later detections, never ground truth."""
import math
from .arena_geometry import ArenaGeometry


def trajectory_errors(events):
    pending=[]
    errors=[]
    previous_revision=0
    for event in events:
        if event.get('event')!='battle_prediction':continue
        world=event.get('world') or {}
        plan=event.get('plan') or {}
        revision=world.get('revision',0)
        if revision<=previous_revision:pending=[]
        previous_revision=revision
        now=world.get('at')
        if now is None:continue
        observations={t['track_id']:t for t in world.get('observations',[]) if not t['card_id'].startswith('unknown:') and abs(t['last_seen']-now)<.01}
        remaining=[]
        for expected in pending:
            if abs(now-expected['at'])<=.25 and expected['track_id'] in observations:
                t=observations[expected['track_id']]
                geom=expected['geometry']
                x,y=geom.xy(t['x'],t['y'])
                errors.append(math.hypot(x-expected['x'],y-expected['y']))
            elif now<expected['at']+.25:remaining.append(expected)
        pending=remaining
        raw=plan.get('compute',{}).get('arena_geometry')
        if raw is None:continue
        geom=ArenaGeometry.from_config(raw)
        for forecast in plan.get('enemy_forecast',[]):
            if forecast.get('track_id') is None:continue
            for sample in forecast.get('linear_samples',[]):
                pending.append(dict(track_id=forecast['track_id'],at=now+sample['dt'],x=sample['x'],y=sample['y'],geometry=geom))
    return dict(reference='later_detector_observations_not_ground_truth',samples=len(errors),
                mean_error_tiles=sum(errors)/len(errors) if errors else None,
                within_half_tile=sum(e<=.5 for e in errors)/len(errors) if errors else None,
                battle_acceptance=False)
