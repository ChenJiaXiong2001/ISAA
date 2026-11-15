"""Render observed RTMW landmarks and server-verified project recognition."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'outputs/ntu_rgb_samples'
OUT = SRC/'project_pool_application_20261019'
OUT.mkdir(exist_ok=True)
result = json.loads((ROOT/'.paper_reference_revision/pool_sample_recognition.json').read_text(encoding='utf-8'))
NAMES = {'A010':'拍手', 'A012':'写字', 'A029':'玩手机或平板'}
font = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 25)
small = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 16)
BODY = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22)]
HAND = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
FACE = []
for a,b in [(23,39),(40,44),(45,49),(50,53),(54,58),(59,64),(65,70),(71,82),(83,90)]:
    FACE += [(i,i+1) for i in range(a,b)]
for a,b in [(59,64),(65,70),(71,82),(83,90)]:
    FACE.append((a,b))

for row in result['samples']:
    aid, recognized = row['true_action'], row['recognized_action']
    raw = np.load(SRC/'real_skeletons'/f'{aid}.npz')
    kp, sc = raw['keypoints'], raw['scores']
    frame_indices = raw['frame_indices'].astype(int)
    boxes = raw['bboxes']
    lookup = {int(fi):ti for ti,fi in enumerate(frame_indices)}
    cap = cv2.VideoCapture(str(SRC/'videos'/f'{aid}.avi'))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best_t = int(np.nan_to_num(sc, nan=0).sum((1,2)).argmax())
    best_frame = int(frame_indices[best_t])
    # Fixed crop keeps the full actor and surrounding RGB context for the whole clip.
    crop = (730, 105, 1330, 955)
    writer = cv2.VideoWriter(str(OUT/f'{aid}_project_recognition.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (600,850))
    for fi in range(total):
        ok, frame = cap.read()
        if not ok: break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        if fi in lookup:
            ti=lookup[fi]
            mi=int(np.nan_to_num(sc[ti],nan=0).sum(1).argmax())
            pts, scores = kp[ti,mi], sc[ti,mi]
            valid=np.isfinite(pts).all(1)&(np.nan_to_num(scores,nan=0)>.05)
            point=lambda i:tuple(int(v) for v in pts[i])
            for edges,col,width in [(BODY,'#2D6CE3',3),(FACE,'#FFCF55',1)]:
                for a,b in edges:
                    if valid[a] and valid[b]: draw.line([point(a),point(b)],fill=col,width=width)
            for base,col in [(91,'#00E0A0'),(112,'#CD86FF')]:
                for a,b in HAND:
                    if valid[base+a] and valid[base+b]: draw.line([point(base+a),point(base+b)],fill=col,width=2)
            for i in range(133):
                if valid[i]:
                    x,y=point(i)
                    r=1 if 23<=i<=90 else 2
                    col='#FFCF55' if 23<=i<=90 else ('#00E0A0' if 91<=i<=111 else '#CD86FF' if i>=112 else '#2D6CE3')
                    draw.ellipse((x-r,y-r,x+r,y+r),fill=col)
            bbox=boxes[ti,mi]
            if np.isfinite(bbox).all():
                x1,y1,x2,y2=map(int,bbox)
                draw.rectangle((x1,y1,x2,y2),outline='#00D48B',width=3)
                label=f'{recognized} {NAMES[recognized]} {row["confidence"]:.4f}'
                bb=draw.textbbox((0,0),label,font=font)
                lx=max(crop[0]+8,min(x1,crop[2]-(bb[2]+20)))
                ly=max(crop[1]+40,y1-42)
                draw.rectangle((lx,ly,lx+bb[2]+18,ly+37),fill='#00D48B')
                draw.text((lx+8,ly+2),label,font=font,fill='#071C2B')
        display=image.crop(crop)
        d=ImageDraw.Draw(display)
        d.rectangle((0,0,600,36),fill='#12304A')
        d.text((12,7),'本项目原池化模型  RTMW-133  离线整段识别',font=small,fill='white')
        d.rectangle((0,814,600,850),fill='#12304A')
        d.text((12,823),f'真实视频及同名骨架  帧 {fi+1}/{total}',font=small,fill='white')
        writer.write(cv2.cvtColor(np.asarray(display),cv2.COLOR_RGB2BGR))
        if fi==best_frame:
            display.save(OUT/f'{aid}_project_recognition.png')
            image.save(OUT/f'{aid}_full_rgb_overlay.png')
    cap.release(); writer.release()
    print(aid, recognized, f'{row["confidence"]:.4f}', total, 'frames')
(OUT/'recognition_provenance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
