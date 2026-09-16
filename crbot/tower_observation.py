"""Read tower bars and independently confirm visible princess-tower rubble."""
import numpy as np


def observe_tower_health(image, rois):
    if not rois:return (None,)*4
    if len(rois) not in (4,6):raise ValueError('four or six tower bar ROIs required')
    values=[]
    for index,roi in enumerate(rois):
        if len(roi)!=4 or not (0<=roi[0]<roi[2]<=1 and 0<=roi[1]<roi[3]<=1):
            raise ValueError('invalid tower bar ROI')
        box=(round(roi[0]*image.width),round(roi[1]*image.height),round(roi[2]*image.width),round(roi[3]*image.height))
        rgb=np.asarray(image.crop(box).convert('RGB')).astype(float)
        if rgb.shape[0]<2 or rgb.shape[1]<12:
            values.append(None);continue
        r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
        # Enemy fill is pink/magenta. Red arena carpet exposed after collapse
        # also satisfies r>g and must not become a fictitious surviving bar.
        fill=(g>125)&(b>150)&(r<g*.85) if index in (0,1,4) else (r>155)&(r>g*1.4)&(r>b*1.2)&(b>g*1.15)
        candidates=[]
        for row in fill:
            # Find a left-aligned filled run, tolerate only anti-aliasing pinholes.
            positions=np.flatnonzero(row)
            if len(positions)<2 or positions[0]>2:continue
            end=positions[-1]+1
            if row[positions[0]:end].mean()<.85:continue
            candidates.append(end/len(row))
        if len(candidates)<2 or max(candidates)-min(candidates)>.10:
            values.append(None)
        else:values.append(round(float(np.median(candidates)),4))
    return tuple(values)


def princess_rubble(image, point):
    """Conservative evidence for the standard arena's pale stone/gold rubble.

    Missing health text, an empty ROI, or a flat floor is insufficient. Unknown
    skins/effects abstain. This is a visual heuristic, not a trained classifier.
    """
    x,y=point
    box=tuple(round(v*s) for v,s in zip((x-.045,y-.015,x+.045,y+.020),image.size*2))
    rgb=np.asarray(image.crop(box).convert('RGB'),dtype=float)
    if min(rgb.shape[:2])<8:return False
    r,g,b=rgb.transpose(2,0,1)
    pale=(r>140)&(g>120)&(b>85)&(r>b*1.08)&(r<g*1.5)
    gold=(r>140)&(g>95)&(b<g*.8)
    return bool(pale.mean()>.50 and gold.mean()>.12 and rgb.mean(2).std()>20)


class TowerHealthTracker:
    """Destroyed towers stay zero until the explicit next-battle reset."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.destroyed=set()
        self.pending={}
        self.last_at=None

    def observe(self, image, rois, points, *, now):
        values=list(observe_tower_health(image,rois))
        if self.last_at is not None and now<self.last_at:
            self.reset()
        fresh=self.last_at is None or now>self.last_at
        self.last_at=now
        for index,point in enumerate(points[:4]):
            if index>=len(values):break
            if index in self.destroyed:
                values[index]=0.
                continue
            if values[index] is not None or not princess_rubble(image,point):
                self.pending.pop(index,None)
                continue
            if not fresh:continue
            first,last,count=self.pending.get(index,(now,now,0))
            if now-last>8:first,count=now,0
            count+=1
            self.pending[index]=(first,now,count)
            if count>=2 and now-first>=.20:
                self.destroyed.add(index)
                values[index]=0.
        return tuple(values)
