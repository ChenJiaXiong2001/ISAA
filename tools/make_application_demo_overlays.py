from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(r'C:\Users\15637\PycharmProjects\ISAA')
SRC=ROOT/'outputs'/'ntu_rgb_samples'; RAW=SRC/'real_skeletons'; OUT=SRC/'application_demos'; OUT.mkdir(exist_ok=True)
font_path=Path(r'C:\Windows\Fonts\msyh.ttc')
font=ImageFont.truetype(str(font_path),30) if font_path.exists() else ImageFont.load_default()
small=ImageFont.truetype(str(font_path),17) if font_path.exists() else ImageFont.load_default()
samples=[('A010','拍手','拍手','0.9963'),('A012','写字','写字','0.8477'),('A029','玩手机或平板','掉落物品','0.6292')]
body_edges=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22),(0,5),(0,6)]
hand_edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
for aid,true_name,pred_name,prob in samples:
    d=np.load(RAW/f'{aid}.npz'); kp=np.asarray(d['keypoints'],np.float32); sc=np.asarray(d['scores'],np.float32)
    quality=np.nan_to_num(sc,nan=0).sum(axis=(1,2)); t=int(np.argmax(quality)); m=int(np.argmax(np.nan_to_num(sc[t],nan=0).sum(axis=1)))
    pts=kp[t,m]; scores=np.nan_to_num(sc[t,m],nan=0); valid=np.isfinite(pts).all(1)&(scores>0.05)
    im=Image.open(SRC/f'{aid}.png').convert('RGB'); w,h=im.size; canvas=im.copy(); draw=ImageDraw.Draw(canvas)
    xy=pts[valid]; minx,miny=xy.min(0); maxx,maxy=xy.max(0); pad=32
    x1=max(0,int(minx-pad)); y1=max(0,int(miny-pad)); x2=min(w-1,int(maxx+pad)); y2=min(h-1,int(maxy+pad))
    # Application-style person box and label.
    col='#00D084' if pred_name==true_name else '#FF8A00'
    draw.rectangle((x1,y1,x2,y2),outline=col,width=6)
    label=f'{aid} {pred_name}  {prob}'
    lb=draw.textbbox((0,0),label,font=font); lw=lb[2]-lb[0]+24; lh=lb[3]-lb[1]+18
    ly=max(0,y1-lh); draw.rectangle((x1,ly,x1+lw,ly+lh),fill=col); draw.text((x1+12,ly+7),label,font=font,fill='black')
    # Draw skeleton on the same RGB frame, using the same selected person.
    def p(i): return (int(pts[i,0]),int(pts[i,1]))
    for a,b in body_edges:
        if valid[a] and valid[b]: draw.line([p(a),p(b)],fill='#2563EB',width=5)
    for base,hand_col in ((91,'#00BFA5'),(112,'#7C3AED')):
        for a,b in hand_edges:
            if valid[base+a] and valid[base+b]: draw.line([p(base+a),p(base+b)],fill=hand_col,width=4)
    for i in list(range(23))+list(range(91,133)):
        if valid[i]:
            c='#00BFA5' if 91<=i<=111 else ('#7C3AED' if i>=112 else '#2563EB')
            x,y=p(i); draw.ellipse((x-6,y-6,x+6,y+6),fill=c,outline='white',width=2)
    draw.text((24,h-48),f'RTMW-133真实骨架  ·  frame={t}  ·  person={m}  ·  预测动作={aid} {pred_name}',font=small,fill='white',stroke_width=3,stroke_fill='black')
    canvas.save(OUT/f'{aid}_application_demo.png')
print(OUT)
