"""Read-only live screenshot / grid inspector; rendering stays on the UI thread."""
import json
import tkinter as tk
import time
import math
import gc
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont, ImageTk


@lru_cache(maxsize=8)
def font(size=14):
    try:return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)
    except OSError:return ImageFont.load_default()


@lru_cache(maxsize=512)
def text_tile(text,size,fill):
    face=font(size)
    box=face.getbbox(text)
    tile=Image.new('RGBA',(max(1,box[2]),max(1,box[3])))
    ImageDraw.Draw(tile).text((0,0),text,font=face,fill=fill)
    return tile


def cached_text(image,position,text,*,size,fill):
    tile=text_tile(text,size,fill)
    image.paste(tile,tuple(round(v) for v in position),tile)


def current_entities(packet):
    """Keep history in the audit packet, not in the current-observation layer."""
    for entity in packet.get('entities', []):
        if entity.get('tower'):
            yield entity
            continue
        if entity.get('source') in {'occluded', 'deployment_hypothesis'}:
            continue
        age = entity.get('age_s')
        if packet.get('status') != 'live_tracking' and age is not None and age >= .05:
            continue
        yield entity


def render_grid(packet, size=(450,700)):
    image=Image.new('RGB',size,'#101a2a');draw=ImageDraw.Draw(image)
    w,h=size;compact=h<280
    scale=max(.1,min((w-24)/18,(h-(50 if compact else 120))/32));ox=(w-18*scale)/2;oy=26 if compact else 60
    def point(x,y):return ox+x*scale,oy+y*scale
    elixir=packet.get('elixir')
    elixir_text='未知' if elixir is None else f'{elixir:.1f}'
    cached_text(image,(8,5 if compact else 10),f"方格战场 · 圣水 {elixir_text}",size=10 if compact else 14,fill='#eef4ff')
    live=packet.get('status')=='live_tracking'
    if not compact:cached_text(image,(12,32),'实时位置跟踪 · 兵种非逐帧重识别' if live else '当前观测：蓝=我方 红=敌方 黄框=身份未知',size=11,fill='#9fb2cb')
    draw.rectangle((*point(0,15),*point(18,17)),fill='#154567')
    for bridge in packet.get('bridges',[]):draw.rectangle((*point(bridge-1,15),*point(bridge+1,17)),fill='#967a50')
    for x in range(19):draw.line((*point(x,0),*point(x,32)),fill='#25344a')
    for y in range(33):draw.line((*point(0,y),*point(18,y)),fill='#25344a')
    boxes=[]
    visible = list(current_entities(packet))
    hidden = len(packet.get('entities', [])) - len(visible)
    for e in visible:
        x,y=point(e['x'],e['y']);color='#5daaff' if e['side']==1 else '#ff707d'
        if e.get('identity_estimated') and not e.get('tower'):color='#e7bd62'
        path=e.get('path') or []
        if len(path)>1:draw.line([point(*p) for p in path],fill=color,width=1)
        radius=max(4,scale*(.65 if e.get('tower') else .30))
        box=(x-radius,y-radius,x+radius,y+radius)
        destroyed=e.get('destroyed',False)
        draw.rectangle(box,outline=color,width=2,fill='#263047' if e.get('identity_estimated') or destroyed else color)
        if destroyed:
            draw.line(box,fill=color,width=2)
            draw.line((box[0],box[3],box[2],box[1]),fill=color,width=2)
        hp=e.get('hp_fraction')
        if hp is not None and not e.get('hp_estimated') and not destroyed:
            draw.line((x-radius,y-radius-4,x+radius,y-radius-4),fill='#4b5260',width=3)
            if hp>0:draw.line((x-radius,y-radius-4,x-radius+2*radius*hp,y-radius-4),fill='#b5ef89',width=3)
        label=str(e.get('track_id') or ('K' if e.get('tower_kind')=='king' else 'T'))
        if destroyed:label+='×'
        elif e.get('level') is not None:label+=f" L{e['level']}"
        if e.get('identity_estimated') or e.get('hp_estimated'):label+='?'
        if (e.get('attributes') or {}).get('air'):label+='↑'
        cached_text(image,(x+radius+2,y-radius),label,size=10,fill='#f0f4ff')
        boxes.append((box,e))
    action=packet.get('action') or {}
    if action.get('card_id'):
        geo=packet['geometry'];l,t,r,b=geo['bounds'];x,y=point((action['x']-l)/(r-l)*18,(action['y']-t)/(b-t)*32)
        draw.ellipse((x-9,y-9,x+9,y+9),outline='#ffe37e',width=3)
    error=packet.get('calibration',{}).get('mean_position_error_tiles')
    if not compact:cached_text(image,(12,h-44),f"兵种快照年龄：{packet.get('model_age_s',0):.1f} 秒" if live else f"位置预测误差：{error if error is not None else '等待可比观测'} 格",size=12,fill='#b9c6da')
    cached_text(image,(8,h-19),f'历史/部署假设 {hidden} 项未画入当前观测' if hidden else '路线为模型预测；黄框身份未知，点击查看',size=10 if compact else 12,fill='#b9c6da')
    return image,boxes


def compose_frame(screenshot,packet,w,h,mode):
    half=w//2 if mode=='compare' else 0
    layers=[];boxes=[]
    if half and screenshot is not None:
        scale=min(1.,half/screenshot.width,h/screenshot.height)
        shot=screenshot.resize((max(1,round(screenshot.width*scale)),max(1,round(screenshot.height*scale))),Image.Resampling.BILINEAR)
        layers.append(((half-shot.width)//2,(h-shot.height)//2,shot))
    if packet is not None:
        # Upload the actual portrait field, not wide empty side margins.
        grid_width=min(w-half,max(240,round(h*18/32)+64))
        left=half+(w-half-grid_width)//2
        grid,boxes=render_grid(packet,(grid_width,h))
        layers.append((left,0,grid))
        boxes=[((b[0]+left,b[1],b[2]+left,b[3]),e) for b,e in boxes]
    else:
        waiting=Image.new('RGB',(w-half,h),'#0c1321')
        draw=ImageDraw.Draw(waiting)
        draw.multiline_text(((w-half)//2,h//2),'等待 P1 战场数据\n进入对局后显示',font=font(11),
                            fill='#8195b0',anchor='mm',align='center',spacing=6)
        layers.append((half,0,waiting))
    return layers,boxes


class GridWorldWindow:
    def __init__(self,parent,source,*,embedded=False):
        self.embedded=embedded;self.mode='compare'
        self.source=source;self.last=None;self.boxes=[]
        self.selected_id=None
        self.period=1/30
        self.next_tick=time.perf_counter()
        self.image_item=None
        self.meter_started=self.next_tick
        self.updated_frames=0
        self.source_fps=0.
        self.packet=None
        # Collect previously closed Tk windows on their owning UI thread,
        # before allocating images on a worker can trigger cyclic collection.
        gc.collect()
        self.renderer=ThreadPoolExecutor(max_workers=1,thread_name_prefix="grid-render")
        self.pending_render=None
        self.ready_render=None
        self.present_after=self.next_tick
        self.window=tk.Frame(parent,bg='#0c1321') if embedded else tk.Toplevel(parent)
        if not embedded:self.window.title('方格战场 · 实时模型检查')
        self.timer_api=None
        try:
            import ctypes
            timer=ctypes.WinDLL('winmm')
            if timer.timeBeginPeriod(1)==0:self.timer_api=timer
        except (AttributeError,OSError):
            pass
        self.window.bind('<Destroy>',self.release_timer,add='+')
        if not embedded:
            self.window.geometry('1040x760');self.window.minsize(800,600)
        self.window.grid_rowconfigure(1,weight=1);self.window.grid_columnconfigure(0,weight=1)
        self.label=tk.Label(self.window,text='等待 P1 战场数据',anchor='w',
                            bg='#0c1321',fg='#9fb2cb',font=('Microsoft YaHei UI',8),width=1)
        self.label.grid(row=0,column=0,sticky='ew')
        self.canvas=tk.Canvas(self.window,bg='#0c1321',highlightthickness=0,width=300,height=160)
        self.canvas.grid(row=1,column=0,sticky='nsew')
        self.details=tk.Text(self.window,width=18,height=1,wrap='word',font=('Microsoft YaHei UI',9),
                             bg='#142137',fg='#eef4ff',insertbackground='#eef4ff',relief='flat')
        self.details.grid(row=1 if embedded else 0,column=1,rowspan=1 if embedded else 2,sticky='nsew')
        if embedded:
            self.details.grid_remove()
            self.close_details=tk.Button(self.window,text='收起属性 ×',command=self.hide_details,
                bg='#142137',fg='#9fb2cb',relief='flat',font=('Microsoft YaHei UI',8))
        self.canvas.bind('<Button-1>',self.select)
        self.after_id=self.window.after(0,self.tick)

    def tick(self):
        if not self.window.winfo_exists():return
        if not getattr(self,'embedded',False) or self.window.winfo_ismapped():
            if hasattr(self,'renderer'):self.refresh_async()
            else:self.refresh()
        now=time.perf_counter()
        elapsed=now-self.meter_started
        if elapsed >= 1:
            self.source_fps=self.updated_frames/elapsed
            self.updated_frames=0
            self.meter_started=now
            self.update_label()
        if hasattr(self,'renderer'):
            # Poll readiness separately from presentation. Waiting a whole
            # 33 ms for a just-finished render otherwise halves live FPS when
            # capture and UI clocks fall on opposite sides of a frame boundary.
            self.after_id=self.window.after(4,self.tick)
            return
        self.next_tick += self.period
        if self.next_tick < now-self.period:
            self.next_tick=now
        self.after_id=self.window.after(max(1,math.ceil((self.next_tick-now)*1000)),self.tick)

    def refresh_async(self):
        # Prepare the next image before uploading this one to Tk, so CPU
        # rendering overlaps the UI upload. Keep only one pending job.
        dimensions=(self.canvas.winfo_width(),self.canvas.winfo_height(),self.mode)
        ready=getattr(self,'ready_render',None)
        if self.pending_render is not None:
            future,key,packet=self.pending_render
            if future.done():
                self.pending_render=None
                combined,boxes=future.result()
                if key[2:]==dimensions:
                    ready=(key,packet,combined,boxes)
        if ready is not None and ready[0][2:]!=dimensions:ready=None
        if self.pending_render is None:
            screenshot,packet=self.source() or (None,None)
            key=(id(packet),id(screenshot),*dimensions)
            displayed=ready[0] if ready else self.last
            if key!=displayed:
                w,h=max(80,dimensions[0]),max(80,dimensions[1])
                future=self.renderer.submit(compose_frame,screenshot,packet,w,h,self.mode)
                self.pending_render=(future,key,packet)
        now=time.perf_counter()
        if ready and now>=getattr(self,'present_after',0):
            key,packet,combined,boxes=ready
            changed=packet is not None and (self.last is None or key[:2]!=self.last[:2])
            self.present(key,packet,combined,boxes,changed)
            self.present_after=max(getattr(self,'present_after',now)+self.period,now)
            ready=None
        self.ready_render=ready

    def refresh(self,force=False):
        source=self.source()
        if not source:source=(None,None)
        screenshot,packet=source
        key=(id(packet),id(screenshot),self.canvas.winfo_width(),self.canvas.winfo_height(),self.mode)
        if key==self.last and not force:return
        new_packet=packet is not None and (self.last is None or key[:2]!=self.last[:2])
        self.last=key;w=max(80,key[2]);h=max(80,key[3]);half=w//2 if self.mode=='compare' else 0
        combined,boxes=compose_frame(screenshot,packet,w,h,self.mode)
        self.present(key,packet,combined,boxes,new_packet)

    def present(self,key,packet,combined,boxes,new_packet):
        self.last=key
        photos=getattr(self,'photos',[])
        items=getattr(self,'image_items',[])
        while len(items)>len(combined):self.canvas.delete(items.pop());photos.pop()
        for index,(x,y,image) in enumerate(combined):
            if index<len(photos) and (photos[index].width(),photos[index].height())==image.size:
                photos[index].paste(image)
            else:
                photo=ImageTk.PhotoImage(image,master=self.canvas)
                if index<len(photos):photos[index]=photo
                else:photos.append(photo)
            if index==len(items):items.append(self.canvas.create_image(x,y,anchor='nw',image=photos[index]))
            else:
                self.canvas.coords(items[index],x,y)
                self.canvas.itemconfigure(items[index],image=photos[index])
        self.photos,self.image_items=photos,items
        self.photo=photos[0];self.image_item=items[0]
        self.boxes=boxes
        self.updated_frames += int(new_packet)
        self.packet=packet
        self.update_label()
        if self.selected_id is not None:
            hit=next((e for _,e in self.boxes if e.get('id')==self.selected_id),None)
            if hit is None:
                self.details_content=None
                self.details.delete('1.0','end');self.details.insert('end','选中单位已不在当前战场记录中')
            else:
                self.show_details(hit)

    def update_label(self):
        self.label_packet=self.packet
        if self.packet is not None:
            self.label.configure(text=f"目标 30 FPS · 新数据 {self.source_fps:.1f} FPS · 第 {self.packet['revision']} 帧 · 同帧截图与模型" +
                ('' if self.embedded else '\n'+self.packet.get('decision','')[:65]),wraplength=max(80,self.canvas.winfo_width()))
        else:self.label.configure(text='等待 P1 战场数据 · 游戏画面可独立查看')

    def hide_details(self):
        self.selected_id=None
        self.details.grid_remove();self.close_details.grid_remove()

    def release_timer(self,event):
        if event.widget is self.window:
            self.renderer.shutdown(wait=True,cancel_futures=True)
            if getattr(self,'after_id',None) is not None:
                self.window.after_cancel(self.after_id)
                self.after_id=None
            if self.timer_api is not None:
                self.timer_api.timeEndPeriod(1)
                self.timer_api=None

    def select(self,event):
        hit=next((e for b,e in reversed(self.boxes) if b[0]-5<=event.x<=b[2]+5 and b[1]-5<=event.y<=b[3]+5),None)
        if not hit:return
        self.selected_id=hit.get('id')
        if self.embedded:
            self.details.grid();self.close_details.grid(row=0,column=1,sticky='ew')
        self.show_details(hit)

    def show_details(self,hit):
        names={'card_id':'单位','side':'敌我（1我方/-1敌方）','hp':'模拟血量','hp_fraction':'血量比例',
               'hp_estimated':'血量为估计','identity_estimated':'身份为假设','source':'数据来源',
               'destroyed':'已摧毁','level':'观测等级','side_evidence':'敌我判断依据',
               'simulated_hp':'推演假设血量','simulated_hp_fraction':'推演假设血量比例',
               'confidence':'识别置信度','target_uid':'目标编号','attributes':'属性','path':'预测路径',
               'damage':'单次伤害','speed':'速度','reach':'射程','period':'攻击间隔','targets':'攻击层',
               'air':'空中单位','building_only':'仅攻击建筑','radius':'碰撞半径'}
        def translate(v):
            if isinstance(v,dict):return {names.get(k,k):translate(x) for k,x in v.items()}
            if isinstance(v,list):return [translate(x) for x in v]
            if isinstance(v,float):return round(v,4)
            return v
        content=json.dumps(translate(hit),ensure_ascii=False,indent=2)
        if content!=getattr(self,'details_content',None):
            self.details_content=content
            self.details.delete('1.0','end');self.details.insert('end',content)
