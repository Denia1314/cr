"""Conservative framed health-bar reading; an absent/ambiguous bar is unknown."""
from __future__ import annotations

import numpy as np


def read_unit_health(image, bbox, side=-1):
    w, h = image.size
    x1, y1, x2, y2 = bbox
    # Search above the detected body, including boxes that already include the bar.
    left, right = max(0, int(x1*w)-8), min(w, int(x2*w)+8)
    top, bottom = max(0, int(y1*h)-int(.045*h)), min(h, int((y1+.35*(y2-y1))*h))
    if right-left < 12 or bottom-top < 5:
        return {}
    pixels = np.asarray(image.crop((left, top, right, bottom)).convert("RGB")).astype(int)
    r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
    colored = ((r >= 200) & (r > g*1.15) & (r > b*.85) if side == -1 else
               (b >= 200) & (b > r*1.2) & (b > g*.9))
    team = ((r>g*1.15)&(r>b*.85) if side == -1 else (b>r*1.2)&(b>g*.9))
    dark = (np.max(pixels, axis=2) < 115) | (team & (np.max(pixels,axis=2)<170))
    readings = []
    # Require a complete rectangular dark frame and a contiguous colored fill.
    # Do not interpret a disappearing bar as a dead unit.
    for y in range(1, len(pixels)-3):
        for start in np.flatnonzero(colored[y] & ~np.roll(colored[y], 1)):
            if start < 1 or not (dark[y, start-1] or team[y,start-1]):
                continue
            end = start
            while end < right-left and colored[y, end]:
                end += 1
            if end-start < 2:
                continue
            frame = start-1
            stop = start
            while stop < right-left and dark[y-1, stop]:
                stop += 1
            if abs(left + (frame+stop-1)/2 - (x1+x2)*w/2) > max(6, (x2-x1)*w*.25):
                continue
            width = stop-frame-2
            if width < 12 or width > min(100, int(w*.20)) or end > stop-1:
                continue
            for height in range(2, min(9, len(pixels)-y)):
                if np.mean(dark[y+height, frame:stop]) < .9:
                    continue
                interior = colored[y:y+height, start:stop-1]
                if interior.size == 0 or not dark[y:y+height, stop-1].all():
                    continue
                if not (dark[y:y+height, frame] | team[y:y+height,frame]).all():
                    continue
                fill = end-start
                if np.mean(interior[:, :fill]) < .9 or (interior[:, fill:].size and np.mean(interior[:, fill:]) > .05):
                    continue
                if dark[y:y+height, end:stop-1].size and np.mean(dark[y:y+height, end:stop-1]) < .9:
                    continue
                readings.append((frame, y, round(fill/width, 3), stop, height))
                break
    if not readings:
        return _gradient_health(pixels, colored, team, left, top, w, h,
                                (x1+x2)*w/2, max(6,(x2-x1)*w*.25))
    if len(readings) != 1:
        return {}
    frame, y, fraction, stop, height = readings[0]
    return {"hp_fraction": fraction, "hp_confidence": .85,
            "hp_bar_bbox": [(left+frame)/w,(top+y-1)/h,(left+stop)/w,(top+y+height+1)/h]}


def _gradient_health(pixels, colored, team, left, top, w, h, center, tolerance):
    """Bright team fill above a dark team-colored remainder, including JPEG bars."""
    import cv2
    _,_,stats,_=cv2.connectedComponentsWithStats(colored.astype('uint8'),8)
    readings=[]
    light=pixels.max(axis=2)
    for x,y,bw,bh,area in stats[1:]:
        if bw<3 or bh<2 or bh>9 or bw<bh*2 or y<1 or y+bh>=len(pixels):continue
        if area/(bw*bh)<.75:continue
        stop=x+bw
        while stop<colored.shape[1] and np.mean(team[y:y+bh,stop])>=.75:
            stop+=1
        width=stop-x
        if width<12 or width>min(100,int(w*.2)) or stop>=colored.shape[1]:continue
        if abs(left+(x+stop)/2-center)>tolerance:continue
        if np.mean(light[y-1,x:stop]<190)<.9 or np.mean(light[y+bh,x:stop]<190)<.9:continue
        if width>bw and (np.mean(colored[y:y+bh,x+bw:stop])>.05 or np.mean(light[y:y+bh,x+bw:stop]<190)<.9):continue
        readings.append(dict(hp_fraction=round(float(bw/width),3),hp_confidence=.8,
                             hp_bar_bbox=[float((left+x)/w),float((top+y-1)/h),float((left+stop)/w),float((top+y+bh+1)/h)]))
    return readings[0] if len(readings)==1 else {}


def detect_unit_health_bars(image, tower_points=()):
    """Read visible framed team bars without claiming a card identity."""
    import cv2
    w,h = image.size
    scale = min(1., 720/w)
    small = image.resize((round(w*scale),round(h*scale))) if scale < 1 else image
    sw,sh = small.size
    rgb = np.asarray(small.convert('RGB')).astype(float)
    r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    results=[]
    for side, mask in ((-1,(r>170)&(r>g*1.15)&(r>b*.85)),(1,(b>170)&(b>r*1.2)&(b>g*.9))):
        mask[:int(sh*.20)]=False;mask[int(sh*.75):]=False
        count,_,stats,_ = cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
        candidates = sorted(stats[1:], key=lambda s:int(s[4]), reverse=True)[:80]
        for x,y,bw,bh,area in candidates:
            if bw<3 or bw>sw*.13 or bh<2 or bh>sh*.015 or bw<bh*2:
                continue
            cx,cy=(x+bw/2)/sw,y/sh
            if any(abs(cx-tx)<.09 and abs(cy-ty)<.04 for tx,ty in tower_points):
                continue
            reading=read_unit_health(small,(max(0,cx-.07),cy+.02,min(1,cx+.07),min(.8,cy+.08)),side)
            if not reading:continue
            box=reading['hp_bar_bbox'];px,py=(box[0]+box[2])/2,box[3]+.02
            if any(abs(px-d['x'])<.015 and abs(py-d['y'])<.015 for d in results):continue
            results.append(dict(side=side,x=px,y=py,**reading,position_source='health_bar_body_estimate'))
    return results
