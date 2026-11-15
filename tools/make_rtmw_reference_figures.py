from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(r'C:\Users\15637\PycharmProjects\ISAA')
OUT=ROOT/'outputs'/'rtmw_reference_figures'; OUT.mkdir(exist_ok=True)
fp=Path(r'C:\Windows\Fonts\msyh.ttc')
font=ImageFont.truetype(str(fp),28) if fp.exists() else ImageFont.load_default()
small=ImageFont.truetype(str(fp),18) if fp.exists() else ImageFont.load_default()
def node(d,x,y,label,color,r=7):
    d.ellipse((x-r,y-r,x+r,y+r),fill=color,outline='white',width=2); d.text((x+9,y-10),label,font=small,fill=color)
def line(d,a,b,color,w=3): d.line([a,b],fill=color,width=w)
im=Image.new('RGB',(1200,760),'white'); d=ImageDraw.Draw(im); d.text((46,30),'RTMW / COCO-WholeBody 133-point layout',font=font,fill='#12304A')
body=[(600,125),(565,155),(635,155),(540,190),(660,190),(530,245),(670,245),(470,335),(730,335),(420,430),(780,430),(555,390),(645,390),(520,520),(680,520),(500,650),(700,650),(450,690),(500,710),(550,710),(750,690),(800,710),(850,710)]
edges=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22)]
for a,b in edges: line(d,body[a],body[b],'#5B7083',4)
for i,p in enumerate(body): node(d,*p,str(i),'#2F80ED' if i<17 else '#F2994A')
for base,col,cx in [(91,'#00A896',420),(112,'#7B61FF',780)]:
    hp=[(cx,295),(cx-35,265),(cx-55,235),(cx-75,205),(cx-95,175),(cx-10,265),(cx-10,230),(cx-10,195),(cx-10,160),(cx+25,265),(cx+45,230),(cx+65,195),(cx+85,160),(cx+45,285),(cx+80,265),(cx+115,245),(cx+150,225),(cx+55,305),(cx+90,305),(cx+125,305),(cx+160,305)]
    for a,b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]: line(d,hp[a],hp[b],col,3)
    for i,p in enumerate(hp): node(d,*p,str(base+i),col,5)
face=[(600,130),(575,145),(560,170),(565,195),(585,215),(615,220),(640,205),(650,180),(645,150),(625,135)]
for i in range(len(face)): line(d,face[i],face[(i+1)%len(face)],'#F2994A',2)
for i,p in enumerate(face): node(d,*p,str(23+i*7),'#F2994A',4)
d.text((60,715),'身体17点 · 足部6点 · 面部68点 · 左右手各21点',font=small,fill='#12304A'); im.save(OUT/'figure4_rtmw_133_layout.png')
im=Image.new('RGB',(1200,650),'white'); d=ImageDraw.Draw(im); d.text((46,30),'RTMW hand keypoints: 21 points per hand',font=font,fill='#12304A')
def hand(cx,col,base,title):
    pts=[(cx,390),(cx-45,335),(cx-75,290),(cx-105,245),(cx-135,200),(cx-12,330),(cx-12,285),(cx-12,240),(cx-12,195),(cx+30,330),(cx+55,285),(cx+80,240),(cx+105,195),(cx+70,360),(cx+110,335),(cx+150,310),(cx+190,285),(cx+90,390),(cx+135,390),(cx+180,390),(cx+225,390)]
    es=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
    for a,b in es: line(d,pts[a],pts[b],col,4)
    for i,p in enumerate(pts): node(d,*p,str(i),col,7)
    d.text((cx-90,500),f'{title}  全身索引 {base}–{base+20}',font=font,fill=col)
hand(260,'#00A896',91,'左手'); hand(760,'#7B61FF',112,'右手'); d.text((140,575),'0腕点 · 1–4拇指 · 5–8食指 · 9–12中指 · 13–16无名指 · 17–20小指',font=small,fill='#12304A'); im.save(OUT/'figure5_rtmw_hands.png')
print(OUT)
