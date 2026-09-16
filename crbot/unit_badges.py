"""Team evidence from white level digits on a red/blue badge, not body color."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def find_level_badges(image, *, arena_only=True):
    scale = min(1., 540 / image.width)
    small = image.convert("RGB")
    if scale < 1:
        small = small.resize((540, round(image.height*scale)), Image.Resampling.BILINEAR)
    rgb = np.asarray(small)
    height, width = rgb.shape[:2]
    r,g,b=rgb.astype(float).transpose(2,0,1)
    white = ((rgb.min(2)>160) & (rgb.max(2).astype(int)-rgb.min(2)<65)) | ((r>170)&(g>125)&(b<g*.65))
    if arena_only:
        white[:round(.18*height)] = False
        white[round(.78*height):] = False
        white[:,:round(.08*width)] = False
        white[:,round(.92*width):] = False
        # Champion ability control is UI, outside the visible fighting area.
        white[round(.68*height):,round(.75*width):] = False
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white.astype("uint8"))
    pixel = width / 540 if arena_only else 1.
    glyphs=[]
    for label,(x,y,w,h,area) in enumerate(stats[1:],1):
        if not (max(1,round(pixel))<=w<=13*pixel and 6*pixel<=h<=17*pixel and area>=5*pixel**2):
            continue
        glyphs.append((int(x),int(y),int(w),int(h),label))
    groups=[];remaining=set(range(len(glyphs)))
    while remaining:
        seed=remaining.pop();group=[seed];pending=[seed]
        while pending:
            x,y,w,h,_=glyphs[pending.pop()]
            for j in list(remaining):
                xx,yy,ww,hh,_=glyphs[j]
                gap=max(x,xx)-min(x+w,xx+ww)
                if -.15*min(w,ww)<=gap<=.7*max(h,hh) and abs(y-yy)<=max(1,2*pixel) and abs(h-hh)<=max(2,3*pixel):
                    remaining.remove(j);group.append(j);pending.append(j)
        groups.append([glyphs[j] for j in group])
    result=[]
    for group in groups:
        if not 1<=len(group)<=2:
            continue  # Four-digit tower health and text are not unit levels.
        x=min(g[0] for g in group);y=min(g[1] for g in group)
        right=max(g[0]+g[2] for g in group);bottom=max(g[1]+g[3] for g in group)
        w,h=right-x,bottom-y
        if w>1.6*h:
            continue
        if len(group)==1:
            # A lone solid highlight is not a numeral. Single digits need shape.
            gx,gy,gw,gh,label=group[0]
            occupancy=float((labels[gy:gy+gh,gx:gx+gw]==label).mean())
            if not .42<=gw/gh<=.95 or occupancy>.82:
                continue
        px=max(2,round(3*pixel));py=max(1,round(2*pixel))
        left,top=max(0,x-px),max(0,y-py)
        rgt,btm=min(width,right+px),min(height,bottom+py)
        patch=rgb[top:btm,left:rgt].astype(float)
        r,g,b=patch.transpose(2,0,1)
        red=(r>90)&(r>g*1.6)&(r>b*1.05)
        blue=(b>100)&(b>r*1.6)&(b>g*1.05)
        votes=[float(blue.mean()),float(red.mean())]
        index=int(np.argmax(votes));mask=(blue,red)[index]
        if votes[index]<.38 or votes[index]<3*max(.01,votes[1-index]):
            continue
        # Single blue highlights on the king's windows are not digits. Do not
        # mask red attackers or genuine double-digit badges in the king area.
        if arena_only and index==0 and len(group)==1 and .44<(x+right)/2/width<.56 and .645<(y+bottom)/2/height<.685:
            continue
        # Require actual background immediately above and below the digits.
        if mask[:py].mean()<.20 or mask[-py:].mean()<.20:
            continue
        ones=[]
        for gx,gy,gw,gh,label in group:
            glyph=labels[gy:gy+gh,gx:gx+gw]==label
            ones.append(gw/gh<=.48 and float(glyph.mean(axis=0).max())>=.85)
        level=11 if len(group)==2 and all(ones) else None
        result.append(dict(side=1 if index==0 else -1,level=level,
            x=(x+right)/2/width,y=(y+bottom)/2/height,
            bbox=[left/width,top/height,rgt/width,btm/height],
            confidence=.95,evidence="level_badge"))
    return result


def associate_badges(boxes, badges):
    """One badge per body and vice versa; conflicting nearby teams abstain."""
    options={}
    for i,box in enumerate(boxes):
        x1,y1,x2,y2=box;w=x2-x1;h=y2-y1
        for j,badge in enumerate(badges):
            x,y=badge["x"],badge["y"]
            if not (x1-.012<=x<=x2+.012 and y1-.035<=y<=y1+min(.026,h*.4)):
                continue
            score=abs(x-(x1+x2)/2)/max(.015,w)+abs(y-y1)/max(.018,h)
            options.setdefault(i,[]).append((score,j))
    pairs=[]
    for i,candidates in options.items():
        candidates.sort()
        best,j=candidates[0]
        if any(badges[k]["side"]!=badges[j]["side"] and score-best<.3 for score,k in candidates[1:]):
            continue
        pairs.append((best,i,j))
    assigned={};used=set()
    for _,i,j in sorted(pairs):
        if j not in used:
            assigned[i]=badges[j];used.add(j)
    return assigned
