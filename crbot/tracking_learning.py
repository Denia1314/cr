"""Reviewed temporal annotations, supervised association training and holdouts."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
from .atomic_file import atomic_write

FEATURES=('distance','iou','area_change','appearance','same_class','gap')


def descriptor(image,box):
    crop=image.crop(tuple(round(v*(image.width if i%2==0 else image.height)) for i,v in enumerate(box))).resize((16,16)).convert('RGB')
    hist=[0.]*24
    pixels=crop.tobytes()
    for offset in range(0,len(pixels),3):
        pixel=pixels[offset:offset+3]
        for channel,value in enumerate(pixel):
            hist[channel*8+min(7,value//32)]+=1/768
    return hist


def features(a,b,dt):
    x,y=a['bbox'],b['bbox']
    area=lambda z:(z[2]-z[0])*(z[3]-z[1])
    intersection=max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]))
    distance=math.hypot((x[0]+x[2]-y[0]-y[2])/2,(x[1]+x[3]-y[1]-y[3])/2)
    appearance=sum(min(v,w) for v,w in zip(a['appearance'],b['appearance']))
    return [distance*10,intersection/max(1e-9,area(x)+area(y)-intersection),
            min(5,abs(math.log(area(x)/area(y)))),appearance,float(a['card_id']==b['card_id']),min(3,dt)/3]


def probability(weights,values):
    score=weights[0]+sum(w*x for w,x in zip(weights[1:],values))
    return 1/(1+math.exp(-max(-50,min(50,score))))


def validate_rows(rows):
    seen=set();identities={};times={};splits={};hashes={}
    for row in rows:
        if row.get('reviewed') is not True:
            raise ValueError('存在未人工确认的帧；不能把自动跟踪结果当成真实身份标签')
        battle=str(row['battle']);split=row['split'];at=float(row['at'])
        if split not in ('train','val','test') or not math.isfinite(at):raise ValueError('无效分组或时间')
        if battle in splits and splits[battle]!=split:raise ValueError('同一对局泄漏到不同分组')
        splits[battle]=split
        if at<=times.get(battle,-math.inf):raise ValueError('帧时间必须递增')
        times[battle]=at
        path=Path(row['image']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes and hashes[digest]!=split:raise ValueError('相同图片泄漏到不同分组')
        hashes[digest]=split
        ids=set()
        for unit in row['units']:
            uid=str(unit['track_id']);box=unit['bbox'];side=unit['side']
            if not uid or uid in ids:raise ValueError('一帧中的单位编号必须唯一')
            ids.add(uid)
            if side not in (-1,1) or len(box)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in box) or box[0]>=box[2] or box[1]>=box[3]:raise ValueError('无效阵营或身体框')
            key=(battle,uid);identity=(unit['card_id'],side)
            if key in identities and identities[key]!=identity:raise ValueError('同一单位编号不能改变兵种或阵营')
            identities[key]=identity
        seen.add(split)
    if seen!={'train','val','test'}:raise ValueError('至少需要独立的训练、验证、测试对局')


def pairs(rows):
    validate_rows(rows)
    result={s:[] for s in ('train','val','test')};previous={}
    for row in rows:
        with Image.open(row['image']) as image:
            units=[dict(u,appearance=descriptor(image,u['bbox'])) for u in row['units']]
        prior=previous.get(row['battle'])
        if prior and 0<row['at']-prior['at']<=3:
            for a in prior['units']:
                for b in units:
                    if a['side']!=b['side']:continue
                    values=features(a,b,row['at']-prior['at'])
                    if values[0]>2:continue
                    result[row['split']].append((values,int(str(a['track_id'])==str(b['track_id']))))
        previous[row['battle']]=dict(at=row['at'],units=units)
    return result


def metrics(samples,weights,threshold=.9):
    tp=fp=fn=tn=0
    for x,y in samples:
        prediction=probability(weights,x)>=threshold
        tp+=bool(prediction and y);fp+=bool(prediction and not y)
        fn+=bool(not prediction and y);tn+=bool(not prediction and not y)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn))


def train(rows,output,epochs=500):
    import numpy as np
    if Path(output).exists():raise ValueError('候选文件已存在，请使用新输出路径以保留旧模型')
    samples=pairs(rows)
    for split,data in samples.items():
        if min(sum(y==label for x,y in data) for label in (0,1))<20:
            raise ValueError(f'{split} 至少需要 20 个同一单位样本和 20 个不同单位样本')
        if sum(not y and x[4]==1 for x,y in data)<10:
            raise ValueError(f'{split} 至少需要 10 个同兵种不同单位的对照样本')
    x=np.asarray([[1]+v for v,y in samples['train']],dtype=float);y=np.asarray([y for v,y in samples['train']],dtype=float)
    weights=np.zeros(x.shape[1]);balance=np.where(y==1,.5/(y==1).mean(),.5/(y==0).mean())
    best=None
    for epoch in range(max(1,min(5000,epochs))):
        predictions=1/(1+np.exp(-np.clip(x@weights,-50,50)))
        weights-=.15*(x.T@((predictions-y)*balance)/len(y)+.001*np.r_[0,weights[1:]])
        if epoch%10==0 or epoch==epochs-1:
            val=metrics(samples['val'],weights.tolist())
            rank=(val['precision']>=.98,val['recall'],val['precision'])
            if best is None or rank>best[0]:best=(rank,weights.copy())
    weights=best[1].tolist()
    val=metrics(samples['val'],weights);test=metrics(samples['test'],weights)
    qualified=all(m['precision']>=.98 and m['recall']>=.5 for m in (val,test))
    payload=dict(schema=1,features=list(FEATURES),weights=weights,threshold=.9,
        model_path=str(Path(output).resolve()),
        validation=val,test=test,qualified=qualified,battle_accepted=False,
        sample_counts={k:len(v) for k,v in samples.items()},
        hard_negative_counts={k:sum(not y and x[4]==1 for x,y in v) for k,v in samples.items()},
        data_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
    atomic_write(Path(output),(json.dumps(payload,ensure_ascii=False,indent=2)+'\n').encode())
    return payload


def load_model(path):
    model=json.loads(Path(path).read_text(encoding='utf-8'))
    if model.get('schema')!=1 or model.get('features')!=list(FEATURES) or model.get('qualified') is not True:
        raise ValueError('跟随模型未通过独立验证')
    if len(model['weights'])!=len(FEATURES)+1 or not all(math.isfinite(w) for w in model['weights']):raise ValueError('无效模型参数')
    if not .9<=float(model.get('threshold',0))<=1:raise ValueError('无效关联门槛')
    for split in ('validation','test'):
        m=model[split]
        if m['precision']<.98 or m['recall']<.5 or m['tp']+m['fn']<20 or m['tn']+m['fp']<20:raise ValueError('验证样本或指标未达标')
    if not all(model.get('hard_negative_counts',{}).get(s,0)>=10 for s in ('train','val','test')):
        raise ValueError('缺少同兵种不同单位的独立对照')
    return model


def refresh_review(path):
    """Replace only the generated UI, preserving its embedded frames/labels."""
    path=Path(path)
    source=path.read_text(encoding='utf-8')
    packets,_=json.JSONDecoder().raw_decode(source.split('const rows=',1)[1])
    from .tracking_cards import illustrated_cards
    template=(Path(__file__).parent/'tracking_review.html').read_text(encoding='utf-8')
    page=template.replace('__DATA__',json.dumps(packets,ensure_ascii=False).replace('</','<\\/'))
    page=page.replace('__CARDS__',json.dumps(illustrated_cards(),ensure_ascii=False).replace('</','<\\/'))
    atomic_write(path,page.encode())
    return len(packets)


def prepare(runs,output,captures='training/tracking-capture'):
    rows=[]
    logs=sorted(Path(captures).glob('*/*/frames.jsonl'),key=lambda p:p.stat().st_mtime)[-3:]
    for log in logs:
        frames=[]
        for line in log.read_text(encoding='utf-8').splitlines(keepends=True):
            if not line.endswith('\n'):continue
            row=json.loads(line)
            if Path(row['image']).is_file():frames.append(row)
        start=next((i for i,r in enumerate(frames) if r.get('proposals')),0)
        rows.extend(frames[max(0,start-2):max(0,start-2)+40])
    fallback_runs=sorted(Path(runs).glob('*'))[-3:] if len({r['battle'] for r in rows})<3 else []
    for run in fallback_runs:
        log=run/'events.jsonl'
        if not log.exists():continue
        battle=0;count=0
        for line in log.read_text(encoding='utf-8').splitlines(keepends=True):
            if not line.endswith('\n'):continue
            event=json.loads(line)
            if event.get('event','').startswith('battle_started'):battle+=1;count=0
            if battle and event.get('event')=='battle_action_pre_send_frame' and event.get('frame') and count<16:
                image=(run/event['frame']).resolve()
                if image.is_file():
                    rows.append(dict(battle=f'{run.name}:{battle}',at=event['timestamp_unix'],image=str(image),reviewed=False,units=[]));count+=1
    rows.sort(key=lambda r:(r['battle'],r['at']))
    groups=list(dict.fromkeys(r['battle'] for r in rows))
    if len(groups)<3:raise ValueError('需要至少三场保存了截图的对局')
    for row in rows:row['split']='test' if row['battle']==groups[-1] else 'val' if row['battle']==groups[-2] else 'train'
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'annotations.jsonl').exists():raise ValueError('标注目录已存在，请使用新输出目录以保留原标注')
    atomic_write(output/'annotations.jsonl',('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n').encode())
    packets=[]
    for row in rows:
        with Image.open(row['image']) as image:
            import io
            image.thumbnail((600,1000));buffer=io.BytesIO();image.convert('RGB').save(buffer,format='JPEG',quality=80)
        packets.append(dict(row,preview='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()))
    template=(Path(__file__).parent/'tracking_review.html').read_text(encoding='utf-8')
    from .tracking_cards import illustrated_cards
    page=template.replace('__DATA__',json.dumps(packets,ensure_ascii=False).replace('</','<\\/'))
    page=page.replace('__CARDS__',json.dumps(illustrated_cards(),ensure_ascii=False).replace('</','<\\/'))
    atomic_write(output/'review.html',page.encode())
    return dict(frames=len(rows),battles=len(groups),review=str((output/'review.html').resolve()),ready_for_training=False)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--runs',default='runs');p.add_argument('--output',default='training/tracking');p.add_argument('--captures',default='training/tracking-capture')
    t=sub.add_parser('train');t.add_argument('annotations');t.add_argument('--output',default='models/tracking/candidate.json');t.add_argument('--epochs',type=int,default=500)
    args=parser.parse_args()
    try:
        result=prepare(args.runs,args.output,args.captures) if args.command=='prepare' else train([json.loads(l) for l in Path(args.annotations).read_text(encoding='utf-8').splitlines() if l.strip()],args.output,args.epochs)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,KeyError,OSError) as error:
        parser.exit(2,f'{error}\n')


if __name__=='__main__':main()
