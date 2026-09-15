"""Read-only live screenshot / grid inspector; rendering stays on the UI thread."""
import json
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk


def font(size=14):
    try:return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)
    except OSError:return ImageFont.load_default()


def render_grid(packet, size=(450,700)):
    image=Image.new('RGB',size,'#101a2a');draw=ImageDraw.Draw(image)
    w,h=size;scale=min((w-40)/18,(h-120)/32);ox=(w-18*scale)/2;oy=60
    def point(x,y):return ox+x*scale,oy+y*scale
    draw.text((12,10),f"方格战场 · 圣水 {packet.get('elixir',0):.1f} · {packet.get('status','')}",font=font(),fill='#eef4ff')
    draw.text((12,32),'蓝=我方  红=敌方  虚线/问号=估计',font=font(12),fill='#9fb2cb')
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
        draw.text((x+radius+2,y-radius),label,font=font(10),fill='#f0f4ff')
        boxes.append((box,e))
    action=packet.get('action') or {}
    if action.get('card_id'):
        geo=packet['geometry'];l,t,r,b=geo['bounds'];x,y=point((action['x']-l)/(r-l)*18,(action['y']-t)/(b-t)*32)
        draw.ellipse((x-9,y-9,x+9,y+9),outline='#ffe37e',width=3)
    error=packet.get('calibration',{}).get('mean_position_error_tiles')
    draw.text((12,h-44),f"位置预测误差：{error if error is not None else '等待可比观测'} 格",font=font(12),fill='#b9c6da')
    draw.text((12,h-24),'路线为模型预测；点击方块查看属性',font=font(12),fill='#b9c6da')
    return image,boxes


class GridWorldWindow:
    def __init__(self,parent,source):
        self.source=source;self.last=None;self.boxes=[]
        self.window=tk.Toplevel(parent);self.window.title('方格战场 · 实时模型检查')
        self.window.geometry('1040x760');self.window.minsize(800,600)
        self.window.grid_rowconfigure(1,weight=1);self.window.grid_columnconfigure(0,weight=1)
        self.label=tk.Label(self.window,text='等待 P1 战场数据；此窗口不会启动游戏或出牌',anchor='w')
        self.label.grid(row=0,column=0,sticky='ew')
        self.canvas=tk.Canvas(self.window,bg='#0c1321',highlightthickness=0)
        self.canvas.grid(row=1,column=0,sticky='nsew')
        self.details=tk.Text(self.window,width=18,wrap='word',font=('Microsoft YaHei UI',9))
        self.details.grid(row=0,column=1,rowspan=2,sticky='nsew')
        self.canvas.bind('<Button-1>',self.select);self.canvas.bind('<Configure>',lambda _:self.refresh(True))
        self.window.after(300,self.tick)

    def tick(self):
        if not self.window.winfo_exists():return
        self.refresh();self.window.after(500,self.tick)

    def refresh(self,force=False):
        source=self.source()
        if not source:return
        screenshot,packet=source
        key=(id(packet),self.canvas.winfo_width(),self.canvas.winfo_height())
        if key==self.last and not force:return
        self.last=key;w=max(400,key[1]);h=max(400,key[2]);half=w//2
        grid,boxes=render_grid(packet,(half,h))
        combined=Image.new('RGB',(w,h),'#0c1321')
        shot=screenshot.copy();shot.thumbnail((half,h));combined.paste(shot,((half-shot.width)//2,(h-shot.height)//2))
        combined.paste(grid,(half,0));self.photo=ImageTk.PhotoImage(combined)
        self.canvas.delete('all');self.canvas.create_image(0,0,anchor='nw',image=self.photo)
        self.boxes=[((b[0]+half,b[1],b[2]+half,b[3]),e) for b,e in boxes]
        self.label.configure(text=f"第 {packet['revision']} 帧 · 同帧截图与模型 · {packet.get('decision','')[:65]}")

    def select(self,event):
        hit=next((e for b,e in reversed(self.boxes) if b[0]-5<=event.x<=b[2]+5 and b[1]-5<=event.y<=b[3]+5),None)
        if not hit:return
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
