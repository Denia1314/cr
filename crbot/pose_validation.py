"""Frozen per-class real holdouts and explicit trial admission, independent of augmentation."""
import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from PIL import Image


def overlap(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def match_boxes(truth,predictions,*,iou=.5):
    """Confidence ordered one-to-one matching; duplicates remain false positives."""
    remaining=set(range(len(truth)));matched=0
    for prediction in sorted(predictions,key=lambda p:p['confidence'],reverse=True):
        best=max(remaining,key=lambda i:overlap(truth[i],prediction['bbox']),default=None)
        if best is not None and overlap(truth[best],prediction['bbox'])>=iou:
            remaining.remove(best);matched+=1
    return dict(tp=matched,fp=len(predictions)-matched,fn=len(remaining))


def trial_gate(baseline,candidate):
    """Limited pose trial only; never confers full recognition/battle acceptance."""
    values=[row[k] for row in (baseline,candidate) for k in ('precision','recall','general_map50','general_precision','median_ms')]
    if not all(math.isfinite(v) for v in values):
        return dict(passed=False,reasons=['nonfinite_metrics'],scope='limited_pose_trial',battle_acceptance=False)
    reasons=[]
    if not candidate.get('evaluation_id') or baseline.get('evaluation_id')!=candidate.get('evaluation_id'):
        reasons.append('different_evaluation_data')
    if candidate['positive_frames']<3 or candidate['battle_groups']<2 or candidate['negative_frames']<2:reasons.append('insufficient_independent_holdout')
    if candidate['precision']<.9 or candidate['recall']<2/3:reasons.append('pose_quality')
    if candidate['recall']<=baseline['recall']:reasons.append('no_pose_recall_improvement')
    if candidate['general_map50']<baseline['general_map50']-.02:reasons.append('general_map_regression')
    if candidate['general_precision']<baseline['general_precision']-.02:reasons.append('general_precision_regression')
    if candidate['median_ms']>100:reasons.append('slower_than_10fps_inference')
    return dict(passed=not reasons,reasons=reasons,scope='limited_pose_trial',battle_acceptance=False)


def validate_holdouts(root,annotations,manifest):
    """Reject duplicate frames, checksum drift and train/holdout contamination."""
    from .pose_dataset import validate_reviews
    root=Path(root).resolve()
    # Extend validation with negatives without changing the frozen train data.
    positives=[r for r in annotations if r.get('bbox')]
    validate_reviews(root,[r for r in manifest['pose_sources'] if r['split']=='train']+positives,manifest['classes'])
    train_groups={r['group'] for r in manifest['pose_sources'] if r['split']=='train'}
    train_hashes={r['sha256'] for r in manifest['pose_sources'] if r['split']=='train'}
    seen=set()
    if not annotations or len({r['card'] for r in annotations})!=1:
        raise ValueError('One target class required for pose holdouts')
    for row in annotations:
        path=(root/row['path']).resolve()
        if (row['split']!='holdout' or row['group'] in train_groups or row.get('annotation_source')!='manual_visual_review'
                or row['card'] not in manifest['classes'] or row['sha256'] in train_hashes or row['sha256'] in seen
                or not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']):
            raise ValueError('Holdout isolation or checksum failed')
        seen.add(row['sha256'])


def evaluate(model_path,root,reviews,dataset,output,*,device='auto'):
    from ultralytics import YOLO
    from .gpu import vision_device
    from .yolo_continuation import dataset_fingerprint
    root=Path(root).resolve();dataset=Path(dataset);output=Path(output)
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    annotations=json.loads(Path(reviews).read_text(encoding='utf-8'))
    validate_holdouts(root,annotations,manifest)
    evaluation_id=hashlib.sha256((dataset_fingerprint(dataset)+json.dumps(annotations,sort_keys=True)+':conf=.7:iou=.5').encode()).hexdigest()
    model=YOLO(str(model_path));names=[model.names[i] for i in range(len(model.names))]
    if names!=manifest['classes']:raise ValueError('Validation classes differ from checkpoint')
    device=vision_device(device);output.mkdir(parents=True,exist_ok=True)
    overall=model.val(data=str((dataset/'data.yaml').resolve()),batch=32,imgsz=640,device=device,
        workers=0,conf=.001,plots=False,save=False,project=str(output),name='general',exist_ok=True,verbose=False)
    records=[];totals=dict(tp=0,fp=0,fn=0)
    for row in annotations:
        image=Image.open(root/row['path']).convert('RGB');source=image.crop((0,0,image.width,round(image.height*.8)))
        prediction=model.predict(source,conf=.7,imgsz=640,device=device,verbose=False)[0]
        predictions=[]
        for cls,score,box in zip(prediction.boxes.cls.cpu().tolist(),prediction.boxes.conf.cpu().tolist(),prediction.boxes.xyxyn.cpu().tolist()):
            if names[int(cls)]==row['card']:
                predictions.append(dict(confidence=score,bbox=[box[0],box[1]*source.height/image.height,box[2],box[3]*source.height/image.height]))
        truth=[row['bbox']] if row.get('bbox') else []
        result=match_boxes(truth,predictions)
        for key in totals:totals[key]+=result[key]
        records.append(dict(path=row['path'],group=row['group'],sha256=row['sha256'],truth=truth,predictions=predictions,**result))
        Image.fromarray(prediction.plot()[...,::-1]).save(output/f"{len(records):03}.jpg")
    # Fully labeled upstream scenes without the target are additional hard negatives.
    # Aggregate mAP excludes absent classes, so it cannot expose these target errors.
    target=manifest['classes'].index(annotations[0]['card'])
    negative_paths=[]
    for path in sorted((dataset/'images/val').glob('*.jpg')):
        labels=(dataset/'labels/val'/f'{path.stem}.txt').read_text().splitlines()
        if not any(int(line.split()[0])==target for line in labels if line.strip()):
            negative_paths.append(str(path))
    negative_records=[];negative_fp=0
    if negative_paths:
        for prediction in model.predict(negative_paths,batch=32,conf=.7,imgsz=640,device=device,stream=True,verbose=False):
            scores=[score for cls,score in zip(prediction.boxes.cls.cpu().tolist(),prediction.boxes.conf.cpu().tolist()) if int(cls)==target]
            if scores:
                negative_fp+=len(scores)
                negative_records.append(dict(path=Path(prediction.path).name,confidence=scores,fp=len(scores)))
                Image.fromarray(prediction.plot()[...,::-1]).save(output/f'negative-{Path(prediction.path).name}')
    local_totals=dict(totals);totals['fp']+=negative_fp
    # Warmed repetitions of a holdout frame measure inference, not game FPS.
    times=[]
    for i in range(8):
        began=time.perf_counter();model.predict(source,conf=.7,imgsz=640,device=device,verbose=False)
        if i>=2:times.append((time.perf_counter()-began)*1000)
    result=dict(model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),evaluation_id=evaluation_id,**totals,
        local_totals=local_totals,general_negative_frames=len(negative_paths),general_target_fp=negative_fp,
        general_negative_records=negative_records,
        precision=totals['tp']/max(1,totals['tp']+totals['fp']),recall=totals['tp']/max(1,totals['tp']+totals['fn']),
        positive_frames=sum(bool(r.get('bbox')) for r in annotations),negative_frames=sum(not r.get('bbox') for r in annotations),
        battle_groups=len({r['group'] for r in annotations}),general_map50=float(overall.box.map50),
        general_precision=float(overall.box.mp),median_ms=statistics.median(times),records=records,
        confidence=.7,iou=.5,battle_acceptance=False,local_full_quality_validated=False)
    (output/'quality.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','root','reviews','dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='auto');a=p.parse_args()
    base=evaluate(a.baseline,a.root,a.reviews,a.dataset,a.output/'baseline',device=a.device)
    candidate=evaluate(a.candidate,a.root,a.reviews,a.dataset,a.output/'candidate',device=a.device)
    gate=trial_gate(base,candidate)
    (a.output/'gate.json').write_text(json.dumps(dict(baseline=base,candidate=candidate,gate=gate),indent=2),encoding='utf-8')
    print(json.dumps(dict(baseline={k:v for k,v in base.items() if k!='records'},candidate={k:v for k,v in candidate.items() if k!='records'},gate=gate),indent=2))


if __name__=='__main__':main()
