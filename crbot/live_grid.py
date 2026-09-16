"""Read-only frame-by-frame position tracking; never feeds predicted actions.

Identity comes from a bounded-age planner observation. Optical flow measures
position changes in new pixels; it is not a new neural identity classification.
"""
import copy
import time
from threading import Event, Lock, Thread

import cv2
import numpy as np
from PIL import Image

from .battle_perception import _level_badge_candidates
from .vision import estimate_elixir


class GridTracker:
    def __init__(self, vision):
        self.vision=vision
        self.previous=None
        self.rows=[]
        self.model_key=None

    @staticmethod
    def small(image):
        width=min(480,image.width)
        return image.resize((width,round(image.height*width/image.width)),Image.Resampling.BILINEAR)

    def update(self, frame, model):
        image,base=model
        age=max(0.,frame.started-base['at'])
        small=self.small(frame.image)
        current=np.asarray(small.convert('L'))
        key=(base['revision'],base['at'],id(image))
        l,t,r,b=base['geometry']['bounds']
        if key!=self.model_key:
            self.rows=copy.deepcopy([e for e in base['entities'] if e.get('tower') or
                                    (age<=2 and e.get('age_s',0) is not None and e.get('age_s',0)<=.5)])
            self.previous=np.asarray(self.small(image).convert('L'))
            self.model_key=key
        previous=self.previous
        height,width=current.shape
        points=[];groups=[]
        if previous is not None and previous.shape==current.shape:
            for row in self.rows:
                if row.get('tower'):continue
                x=round((l+row['x']/18*(r-l))*width)
                y=round((t+row['y']/32*(b-t))*height)
                left,top=max(0,x-12),max(0,y-18)
                patch=previous[top:min(height,y+12),left:min(width,x+12)]
                corners=cv2.goodFeaturesToTrack(patch,8,.05,3) if patch.size else None
                if corners is not None:
                    start=len(points)
                    points.extend(corners.reshape(-1,2)+[left,top])
                    groups.append((row,start,len(points)))
        moved=set()
        if points:
            points=np.asarray(points,dtype=np.float32).reshape(-1,1,2)
            next_points,ok,error=cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(21,21),maxLevel=2)
            back,back_ok,_=cv2.calcOpticalFlowPyrLK(current,previous,next_points,None,winSize=(21,21),maxLevel=2)
            valid=(ok.ravel()!=0)&(back_ok.ravel()!=0)&(error.ravel()<30)&(np.linalg.norm(back-points,axis=2).ravel()<1.5)
            delta=(next_points-points).reshape(-1,2)
            for row,start,end in groups:
                accepted=delta[start:end][valid[start:end]]
                if len(accepted)<3:continue
                dx,dy=np.median(accepted,axis=0)
                if abs(dx)>width*.08 or abs(dy)>height*.08:continue
                row['x']+=float(dx)/width/(r-l)*18
                row['y']+=float(dy)/height/(b-t)*32
                if not (0<=row['x']<=18 and 0<=row['y']<=32):continue
                row.update(source='optical_flow',position_observed_at=frame.started,
                           identity_age_s=age,age_s=0.,hp=None,hp_fraction=None,hp_estimated=True,
                           path=[],target_uid=None)
                moved.add(id(row))
        self.rows=[row for row in self.rows if row.get('tower') or (age<=2 and id(row) in moved)]
        self.previous=current
        rows=copy.deepcopy(self.rows)
        # New enemy badges enter immediately, without waiting for a planner.
        for index,(x,y,_) in enumerate(_level_badge_candidates(small,pixel_scale=small.width/frame.image.width)):
            gx,gy=(x-l)/(r-l)*18,(y-t)/(b-t)*32
            if not (0<=gx<=18 and 0<=gy<=32):continue
            if any(e['side']==-1 and not e.get('tower') and abs(e['x']-gx)<1.2 and abs(e['y']-gy)<1.5 for e in rows):continue
            rows.append(dict(id=f'badge:{frame.sequence}:{index}',track_id=f'b{index+1}',x=gx,y=gy,side=-1,
                             card_id='unknown',identity_estimated=True,hp_estimated=True,
                             hp_fraction=None,source='current_frame_badge',age_s=0.,path=[]))
        # Static anchors are geometry, not freshly read tower health.
        for row in rows:
            if row.get('tower'):row.update(hp_fraction=None,hp_estimated=True,source='tower_anchor',path=[])
            row['cell']=[int(row['x']),int(row['y'])]
        packet={k:copy.deepcopy(base[k]) for k in ('geometry','bridges','width','height') if k in base}
        elixir,confidence=estimate_elixir(small,self.vision['elixir_roi'])
        packet.update(revision=frame.sequence,at=frame.started,entities=rows,elixir=elixir,
                      elixir_confidence=confidence,action=None,status='live_tracking',
                      model_revision=base['revision'],model_age_s=age,
                      decision='逐帧位置跟踪；兵种沿用最近识别，非逐帧完整推演')
        return frame.image,packet


class LiveGridStream:
    def __init__(self,frames,model_source,vision):
        self.frames,self.model_source=frames,model_source
        self.tracker=GridTracker(vision)
        self.stopped=Event();self.lock=Lock();self.result=None;self.error=None
        self.processed=0;self.interval=None;self.work_s=None
        self.thread=Thread(target=self.run,name='live-grid-tracking',daemon=True)

    def start(self):
        self.thread.start();return self

    def run(self):
        sequence=0;last=None
        try:
            while not self.stopped.is_set():
                try:frame=self.frames.get(sequence=sequence,timeout=.2)
                except TimeoutError:continue
                sequence=frame.sequence
                model=self.model_source()
                start=time.perf_counter()
                result=self.tracker.update(frame,model) if model else None
                if model is None:self.tracker.model_key=None
                finished=time.perf_counter()
                with self.lock:
                    self.result=result
                    if result:
                        self.processed+=1;self.work_s=finished-start
                        self.interval=None if last is None else finished-last;last=finished
        except Exception as exc:
            with self.lock:self.error=str(exc);self.result=None

    def latest(self):
        with self.lock:return self.result

    def status(self):
        with self.lock:
            return dict(processed=self.processed,actual_fps=1/self.interval if self.interval else None,
                        work_s=self.work_s,error=self.error,scope='visual_position_tracking_not_identity_inference')

    def close(self):
        self.stopped.set();self.thread.join()
