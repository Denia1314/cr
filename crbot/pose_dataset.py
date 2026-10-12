"""Mix manually reviewed pose crops into labeled training scenes; freeze real holdouts."""
import argparse
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from PIL import Image


def validate_reviews(root,records,classes):
    groups={};hashes={}
    for row in records:
        if row.get('annotation_source')!='manual_visual_review':raise ValueError('Manual visual review required')
        split=row['split']
        if split not in {'train','holdout'}:raise ValueError('Invalid pose split')
        group=row['group']
        if group in groups and groups[group]!=split:raise ValueError('Battle group crosses pose splits')
        groups[group]=split
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Pose source escapes root')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=row['sha256']:raise ValueError('Pose source checksum changed')
        if digest in hashes:
            if hashes[digest]!=split:raise ValueError('Identical pose image crosses splits')
            raise ValueError('Duplicate reviewed pose source')
        hashes[digest]=split
        if row['card'] not in classes:raise ValueError('Pose class differs from model family')
        box=row['bbox']
        if len(box)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in box) or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError('Invalid reviewed pose box')
    if set(groups.values())!={'train','holdout'}:raise ValueError('Independent pose holdout required')


def export_pose(base,root,reviews,output,*,copies=20,seed=20261012):
    base,root,output=map(lambda p:Path(p).resolve(),(base,root,output))
    if output.exists() or output.is_relative_to(base):raise ValueError('Use a new separate pose output')
    if not 1<=copies<=100:raise ValueError('Pose copies must be between 1 and 100')
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='real_battle_bootstrap':raise ValueError('Labeled real battle base required')
    records=json.loads(Path(reviews).read_text(encoding='utf-8'))
    classes=manifest['classes'];validate_reviews(root,records,classes)
    backgrounds=sorted((base/'images/train').glob('*.jpg'))
    if not backgrounds:raise ValueError('Labeled training backgrounds required')
    for folder in ('images','labels'):
        shutil.copytree(base/folder,output/folder,ignore=shutil.ignore_patterns('*.cache'))
    if (base/'LICENSE').is_file():shutil.copy2(base/'LICENSE',output/'LICENSE')
    # Never use validation scenes as augmentation backgrounds.
    rng=random.Random(seed);added=[]
    for row in records:
        if row['split']!='train':continue
        with Image.open(root/row['path']) as source:
            source=source.convert('RGB');sw,sh=source.size
            box=row['bbox'];crop=source.crop(tuple(round(v*(sw if i%2==0 else sh)) for i,v in enumerate(box)))
            if crop.width<2 or crop.height<2:raise ValueError('Reviewed crop is empty')
            for n in range(copies):
                placed=False
                for attempt in range(100):
                    background=rng.choice(backgrounds)
                    image=Image.open(background).convert('RGB');w,h=image.size
                    lines=(base/'labels/train'/f'{background.stem}.txt').read_text().splitlines()
                    rectangles=[]
                    for line in lines:
                        _,x,y,bw,bh=map(float,line.split())
                        rectangles.append(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h))
                    scale=w/sw*rng.uniform(.8,1.2)
                    pw,ph=max(2,round(crop.width*scale)),max(2,round(crop.height*scale))
                    left=round(rng.uniform(.12,.82)*w-pw/2);top=round(rng.uniform(.3,.66)*h-ph/2)
                    if left<0 or top<0 or left+pw>w or top+ph>h:continue
                    if any(min(left+pw+3,r[2])>max(left-3,r[0]) and min(top+ph+3,r[3])>max(top-3,r[1]) for r in rectangles):continue
                    image.paste(crop.resize((pw,ph),Image.Resampling.BILINEAR),(left,top))
                    identity=hashlib.sha256(f"{row['sha256']}:{n}:{seed}".encode()).hexdigest()[:20]
                    name=f'pose_{identity}'
                    image.save(output/'images/train'/f'{name}.jpg',quality=95)
                    line=f"{classes.index(row['card'])} {(left+pw/2)/w:.8f} {(top+ph/2)/h:.8f} {pw/w:.8f} {ph/h:.8f}"
                    (output/'labels/train'/f'{name}.txt').write_text('\n'.join(lines+[line])+'\n')
                    added.append(dict(group=row['group'],split='train',source=row['path'],source_sha256=row['sha256'],
                        background=background.name,image=f'images/train/{name}.jpg',augmentation='reviewed_real_crop_composite'))
                    placed=True;break
                if not placed:raise ValueError('No non-overlapping background for reviewed pose')
    result={**manifest,'counts':{**manifest['counts'],'train':manifest['counts']['train']+len(added)},
        'records':manifest['records']+added,'pose_sources':records,'pose_seed':seed,
        'pose_unique_train_frames':sum(r['split']=='train' for r in records),
        'pose_unique_holdout_frames':sum(r['split']=='holdout' for r in records),
        'pose_augmentations':len(added),'pose_holdout_in_training':False}
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'data.yaml').write_text(f'path: {json.dumps(output.as_posix())}\ntrain: images/train\nval: images/val\nnames: {json.dumps(classes)}\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','root','reviews','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--copies',type=int,default=20)
    a=p.parse_args();r=export_pose(a.base,a.root,a.reviews,a.output,copies=a.copies)
    print(json.dumps({k:r[k] for k in ('counts','pose_unique_train_frames','pose_unique_holdout_frames','pose_augmentations')},indent=2))


if __name__=='__main__':main()
