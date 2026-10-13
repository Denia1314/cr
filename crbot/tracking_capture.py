"""Bounded asynchronous same-frame capture for reviewed temporal training."""
import json
import queue
import threading
import uuid
from pathlib import Path


class TrackingCapture:
    def __init__(self,root,fps=2,limit=600):
        self.root=Path(root)/uuid.uuid4().hex
        self.period=1/max(.5,min(5,float(fps)));self.limit=max(1,min(1200,int(limit)))
        self.queue=queue.Queue(maxsize=2);self.stop=threading.Event()
        self.last={};self.counts={};self.saved=0;self.dropped=0;self.error=''
        self.thread=threading.Thread(target=self.run,daemon=True,name='tracking-capture');self.thread.start()

    def submit(self,image,at,generation,proposals):
        if generation<0 or self.counts.get(generation,0)>=self.limit or at-self.last.get(generation,-float('inf'))<self.period:return
        self.last[generation]=at;self.counts[generation]=self.counts.get(generation,0)+1
        try:self.queue.put_nowait((image,at,generation,[dict(p) for p in proposals]))
        except queue.Full:self.dropped+=1

    def run(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:image,at,generation,proposals=self.queue.get(timeout=.1)
            except queue.Empty:continue
            try:
                folder=self.root/str(generation);folder.mkdir(parents=True,exist_ok=True)
                path=folder/f'{self.saved:06d}.jpg';image.convert('RGB').save(path,quality=95)
                row=dict(battle=f'{self.root.name}:{generation}',at=at,image=str(path.resolve()),
                    reviewed=False,units=[],proposals=proposals,source='continuous_same_frame_capture')
                with (folder/'frames.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(row,ensure_ascii=False)+'\n')
                self.saved+=1
            except (OSError,ValueError) as error:self.error=str(error)

    def close(self):
        self.stop.set();self.thread.join(timeout=5)
