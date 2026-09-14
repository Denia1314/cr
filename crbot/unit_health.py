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
    colored = ((r > 110) & (r > g*1.4) & (r > b*1.25) if side == -1 else
               (b > 110) & (b > r*1.3) & (b > g*1.1))
    dark = np.max(pixels, axis=2) < 90
    readings = []
    # Require a complete rectangular dark frame and a contiguous colored fill.
    # Do not interpret a disappearing bar as a dead unit.
    for y in range(1, len(pixels)-3):
        for start in np.flatnonzero(colored[y] & ~np.roll(colored[y], 1)):
            if start < 1 or not dark[y, start-1]:
                continue
            end = start
            while end < right-left and colored[y, end]:
                end += 1
            if end-start < 2:
                continue
            frame = start-1
            stop = frame
            while stop < right-left and dark[y-1, stop]:
                stop += 1
            if abs(left + (frame+stop-1)/2 - (x1+x2)*w/2) > max(6, (x2-x1)*w*.25):
                continue
            width = stop-frame-2
            if width < 12 or width > min(100, int(w*.20)) or end > stop-1:
                continue
            for height in range(2, min(9, len(pixels)-y)):
                if not dark[y+height, frame:stop].all():
                    continue
                interior = colored[y:y+height, start:stop-1]
                if interior.size == 0 or not dark[y:y+height, stop-1].all():
                    continue
                if not dark[y:y+height, frame].all():
                    continue
                fill = end-start
                if not interior[:, :fill].all() or interior[:, fill:].any():
                    continue
                if not dark[y:y+height, end:stop-1].all():
                    continue
                readings.append((frame, y, round(fill/width, 3)))
                break
    if len(readings) != 1:
        return {}
    return {"hp_fraction": readings[0][2], "hp_confidence": .85}
