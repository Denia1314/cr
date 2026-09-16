"""Read-only live screenshot / grid inspector; rendering stays on the UI thread."""
import json
import tkinter as tk
import time
import math
from functools import lru_cache
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


def render_grid(packet, size=(450,700)):
    image=Image.new('RGB',size,'#101a2a');draw=ImageDraw.Draw(image)
    w,h=size;scale=min((w-40)/18,(h-120)/32);ox=(w-18*scale)/2;oy=60
    def point(x,y):return ox+x*scale,oy+y*scale
    cached_text(image,(12,10),f"方格战场 · 圣水 {packet.get('elixir',0):.1f} · {packet.get('status','')}",size=14,fill='#eef4ff')
    cached_text(image,(12,32),'蓝=我方  红=敌方  虚线/问号=估计',size=12,fill='#9fb2cb')
    draw.rectangle((*point(0,15),*point(18,17)),fill='#154567')
    for bridge in packet.get('bridges',[]):draw.rectangle((*point(bridge-1,15),*point(bridge+1,17)),fill='#967a50')
    for x in range(19):draw.line((*point(x,0),*point(x,32)),fill='#25344a')
    for y in range(33):draw.line((*point(0,y),*point(18,y)),fill='#25344a')
    boxes=[]
    for e in packet.get('entities',[]):
        x,y=point(e['x'],e['y']);color='#5daaff' if e['side']==1 else '#ff707d'
        path=e.get('path') or []
        if len(path)>1:draw.line([point(*p) for p in path],fill=color,width=1)
        radius=max(4,scale*(.65 if e.get('tower') else .30))
        box=(x-radius,y-radius,x+radius,y+radius)
        draw.rectangle(box,outline=color,width=2,fill='#263047' if e.get('identity_estimated') else color)
        hp=e.get('hp_fraction')
        if hp is not None:
            draw.line((x-radius,y-radius-4,x+radius,y-radius-4),fill='#4b5260',width=3)
            draw.line((x-radius,y-radius-4,x-radius+2*radius*hp,y-radius-4),fill='#b5ef89',width=3)
        label=str(e.get('track_id') or ('K' if e.get('tower_kind')=='king' else 'T'))
        if e.get('identity_estimated') or e.get('hp_estimated'):label+='?'
        if (e.get('attributes') or {}).get('air'):label+='↑'
        cached_text(image,(x+radius+2,y-radius),label,size=10,fill='#f0f4ff')
        boxes.append((box,e))
    action=packet.get('action') or {}
    if action.get('card_id'):
        geo=packet['geometry'];l,t,r,b=geo['bounds'];x,y=point((action['x']-l)/(r-l)*18,(action['y']-t)/(b-t)*32)
        draw.ellipse((x-9,y-9,x+9,y+9),outline='#ffe37e',width=3)
    error=packet.get('calibration',{}).get('mean_position_error_tiles')
    cached_text(image,(12,h-44),f"位置预测误差：{error if error is not None else '等待可比观测'} 格",size=12,fill='#b9c6da')
    cached_text(image,(12,h-24),'路线为模型预测；点击方块查看属性',size=12,fill='#b9c6da')
    return image,boxes


class GridWorldWindow:
    def __init__(self,parent,source):
        self.source=source;self.last=None;self.boxes=[]
        self.selected_id=None
        self.period=1/30
        self.next_tick=time.perf_counter()
        self.image_item=None
        self.meter_started=self.next_tick
        self.updated_frames=0
        self.source_fps=0.
        self.packet=None
        self.window=tk.Toplevel(parent);self.window.title('方格战场 · 实时模型检查')
        self.timer_api=None
        try:
            import ctypes
            timer=ctypes.WinDLL('winmm')
            if timer.timeBeginPeriod(1)==0:self.timer_api=timer
        except (AttributeError,OSError):
            pass
        self.window.bind('<Destroy>',self.release_timer,add='+')
        self.window.geometry('1040x760');self.window.minsize(800,600)
        self.window.grid_rowconfigure(1,weight=1);self.window.grid_columnconfigure(0,weight=1)
        self.label=tk.Label(self.window,text='等待 P1 战场数据；此窗口不会启动游戏或出牌',anchor='w')
        self.label.grid(row=0,column=0,sticky='ew')
        self.canvas=tk.Canvas(self.window,bg='#0c1321',highlightthickness=0)
        self.canvas.grid(row=1,column=0,sticky='nsew')
        self.details=tk.Text(self.window,width=18,wrap='word',font=('Microsoft YaHei UI',9))
        self.details.grid(row=0,column=1,rowspan=2,sticky='nsew')
        self.canvas.bind('<Button-1>',self.select)
        self.window.after(0,self.tick)

    def tick(self):
        if not self.window.winfo_exists():return
        self.refresh()
        now=time.perf_counter()
        elapsed=now-self.meter_started
        if elapsed >= 1:
            self.source_fps=self.updated_frames/elapsed
            self.updated_frames=0
            self.meter_started=now
            self.update_label()
        self.next_tick += self.period
        if self.next_tick < now-self.period:
            self.next_tick=now
        self.window.after(max(1,math.ceil((self.next_tick-now)*1000)),self.tick)

    def refresh(self,force=False):
        source=self.source()
        if not source:return
        screenshot,packet=source
        key=(id(packet),id(screenshot),self.canvas.winfo_width(),self.canvas.winfo_height())
        if key==self.last and not force:return
        new_packet=self.last is None or key[:2]!=self.last[:2]
        self.last=key;w=max(400,key[2]);h=max(400,key[3]);half=w//2
        grid,boxes=render_grid(packet,(half,h))
        combined=Image.new('RGB',(w,h),'#0c1321')
        shot=screenshot.copy();shot.thumbnail((half,h),Image.Resampling.BILINEAR);combined.paste(shot,((half-shot.width)//2,(h-shot.height)//2))
        combined.paste(grid,(half,0));self.photo=ImageTk.PhotoImage(combined)
        if self.image_item is None:
            self.image_item=self.canvas.create_image(0,0,anchor='nw',image=self.photo)
        else:
            self.canvas.itemconfigure(self.image_item,image=self.photo)
        self.boxes=[((b[0]+half,b[1],b[2]+half,b[3]),e) for b,e in boxes]
        self.updated_frames += int(new_packet)
        self.packet=packet
        self.update_label()
        if self.selected_id is not None:
            hit=next((e for _,e in self.boxes if e.get('id')==self.selected_id),None)
            if hit is None:
                self.details.delete('1.0','end');self.details.insert('end','选中单位已不在当前战场记录中')
            else:
                self.show_details(hit)

    def update_label(self):
        if self.packet is not None:
            self.label.configure(text=f"目标 30 FPS · 数据 {self.source_fps:.1f} FPS · 第 {self.packet['revision']} 帧 · 同帧截图与模型\n{self.packet.get('decision','')[:65]}",wraplength=max(400,self.canvas.winfo_width()))

    def release_timer(self,event):
        if event.widget is self.window and self.timer_api is not None:
            self.timer_api.timeEndPeriod(1)
            self.timer_api=None

    def select(self,event):
        hit=next((e for b,e in reversed(self.boxes) if b[0]-5<=event.x<=b[2]+5 and b[1]-5<=event.y<=b[3]+5),None)
        if not hit:return
        self.selected_id=hit.get('id')
        self.show_details(hit)

    def show_details(self,hit):
        names={'card_id':'单位','side':'敌我（1我方/-1敌方）','hp':'模拟血量','hp_fraction':'血量比例',
               'hp_estimated':'血量为估计','identity_estimated':'身份为假设','source':'数据来源',
               'confidence':'识别置信度','target_uid':'目标编号','attributes':'属性','path':'预测路径',
               'damage':'单次伤害','speed':'速度','reach':'射程','period':'攻击间隔','targets':'攻击层',
               'air':'空中单位','building_only':'仅攻击建筑','radius':'碰撞半径'}
        def translate(v):
            if isinstance(v,dict):return {names.get(k,k):translate(x) for k,x in v.items()}
            if isinstance(v,list):return [translate(x) for x in v]
            if isinstance(v,float):return round(v,4)
            return v
        self.details.delete('1.0','end');self.details.insert('end',json.dumps(translate(hit),ensure_ascii=False,indent=2))
