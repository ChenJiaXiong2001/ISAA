from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'C:\Users\15637\PycharmProjects\ISAA')
SRC = ROOT / 'outputs' / 'ntu_rgb_samples'
RAW = SRC / 'real_skeletons'
OUT = SRC / 'overlays'
OUT.mkdir(exist_ok=True)
font_path = Path(r'C:\Windows\Fonts\msyh.ttc')
font = ImageFont.truetype(str(font_path), 34) if font_path.exists() else ImageFont.load_default()
small = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
samples = [('A010','拍手','拍手','0.9963','A010.png'), ('A012','写字','写字','0.8477','A012.png'), ('A029','玩手机或平板','掉落物品','0.6292','A029.png')]
body = [(0.50,0.17),(0.47,0.20),(0.53,0.20),(0.45,0.23),(0.55,0.23),(0.43,0.30),(0.57,0.30),(0.39,0.40),(0.61,0.40),(0.35,0.50),(0.65,0.50),(0.46,0.50),(0.54,0.50),(0.43,0.63),(0.57,0.63),(0.42,0.77),(0.58,0.77),(0.39,0.84),(0.44,0.84),(0.48,0.84),(0.56,0.84),(0.61,0.84),(0.65,0.84)]
edges = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22)]
def hand(cx, cy, scale):
    pts=[(cx,cy),(cx-scale*.45,cy-scale*.55),(cx-scale*.65,cy-scale*.9),(cx-scale*.82,cy-scale*1.18),(cx-scale*.98,cy-scale*1.45),(cx-scale*.12,cy-scale*.55),(cx-scale*.12,cy-scale*.92),(cx-scale*.12,cy-scale*1.28),(cx-scale*.12,cy-scale*1.58),(cx+scale*.28,cy-scale*.52),(cx+scale*.48,cy-scale*.88),(cx+scale*.66,cy-scale*1.24),(cx+scale*.82,cy-scale*1.56),(cx+scale*.55,cy-scale*.12),(cx+scale*.8,cy-scale*.3),(cx+scale*1.04,cy-scale*.48),(cx+scale*1.28,cy-scale*.66),(cx+scale*.72,cy+scale*.14),(cx+scale*.98,cy+scale*.14),(cx+scale*1.24,cy+scale*.14),(cx+scale*1.5,cy+scale*.14)]
    es=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
    return pts, es
for aid, true_name, pred_name, prob, fn in samples:
    data=np.load(RAW/f'{aid}.npz'); kp=np.asarray(data['keypoints'],dtype=np.float32); sc=np.asarray(data['scores'],dtype=np.float32)
    quality=np.nan_to_num(sc,nan=0).sum(axis=(1,2)); t=int(np.argmax(quality)); m=int(np.argmax(np.nan_to_num(sc[t],nan=0).sum(axis=1)))
    real=kp[t,m]; scores=np.nan_to_num(sc[t,m],nan=0)
    im=Image.open(SRC/fn).convert('RGB')
    W,H=im.size; canvas=Image.new('RGB',(W+520,H+80),'white'); canvas.paste(im,(0,80)); d=ImageDraw.Draw(canvas)
    d.rectangle((0,0,1440,80),fill='#12304A'); d.text((28,20),f'预测动作：{aid} {pred_name}   Top-1 {prob}',font=font,fill='white')
    ox,oy=W+18,80; panel_w=480; panel_h=H; d.rectangle((ox,oy,ox+panel_w,oy+panel_h),fill='#F7F9FB')
    valid=np.isfinite(real).all(axis=1) & (scores>0.05); sx=panel_w/W; sy=panel_h/H
    pts=[(ox+float(x)*sx,oy+float(y)*sy) for x,y in real]
    for a,b in edges:
        if valid[a] and valid[b]: d.line([pts[a],pts[b]],fill='#60758A',width=3)
    for base in (91,112):
        for a,b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]:
            if valid[base+a] and valid[base+b]: d.line([pts[base+a],pts[base+b]],fill='#0B8F82' if base==91 else '#2563EB',width=3)
    for i,(x,y) in enumerate(pts):
        if valid[i] and (i<23 or i>=91):
            col='#0B8F82' if 91<=i<=111 else '#2563EB'; d.ellipse((x-5,y-5,x+5,y+5),fill=col); d.text((x+6,y-8),str(i),font=small,fill=col)
    d.text((ox+18,oy+H-62),'真实RTMW-133 NPZ骨架',font=font,fill='#12304A')
    d.text((ox+18,oy+H-28),'编号为133点全身索引',font=small,fill='#1F2937')
    canvas.save(OUT/f'{aid}_skeleton_overlay.png')
print(OUT)
