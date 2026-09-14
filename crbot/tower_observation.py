"""Read calibrated tower bar strips. Missing bars never imply destroyed towers."""
import numpy as np


def observe_tower_health(image, rois):
    if not rois:return (None,)*4
    if len(rois)!=4:raise ValueError('four tower bar ROIs required')
    values=[]
    for index,roi in enumerate(rois):
        if len(roi)!=4 or not (0<=roi[0]<roi[2]<=1 and 0<=roi[1]<roi[3]<=1):
            raise ValueError('invalid tower bar ROI')
        box=(round(roi[0]*image.width),round(roi[1]*image.height),round(roi[2]*image.width),round(roi[3]*image.height))
        rgb=np.asarray(image.crop(box).convert('RGB')).astype(float)
        if rgb.shape[0]<2 or rgb.shape[1]<12:
            values.append(None);continue
        r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
        fill=(g>125)&(b>150)&(r<g*.85) if index<2 else (r>155)&(r>g*1.4)&(r>b*1.2)
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
