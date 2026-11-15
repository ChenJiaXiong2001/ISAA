from pathlib import Path
import cv2, numpy as np

ROOT=Path(r'C:\Users\15637\PycharmProjects\ISAA')
VID=ROOT/'outputs'/'ntu_rgb_samples'/'videos'; RAW=ROOT/'outputs'/'ntu_rgb_samples'/'real_skeletons'; OUT=ROOT/'outputs'/'ntu_rgb_samples'/'application_videos'; OUT.mkdir(exist_ok=True)
samples=[('A010','拍手','拍手','0.9963'),('A012','写字','写字','0.8477'),('A029','玩手机或平板','掉落物品','0.6292')]
body_edges=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22),(0,5),(0,6)]
hand_edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
for aid,true_name,rec_name,prob in samples:
    d=np.load(RAW/f'{aid}.npz'); kp=np.asarray(d['keypoints'],np.float32); sc=np.asarray(d['scores'],np.float32)
    cap=cv2.VideoCapture(str(VID/f'{aid}.avi')); fps=cap.get(cv2.CAP_PROP_FPS) or 30; w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc=cv2.VideoWriter_fourcc(*'mp4v'); writer=cv2.VideoWriter(str(OUT/f'{aid}_application_demo.mp4'),fourcc,fps,(w,h))
    T=kp.shape[0]
    for fi in range(n):
        ok,frame=cap.read()
        if not ok: break
        ti=min(T-1,round(fi*(T-1)/max(1,n-1))); quality=np.nan_to_num(sc[ti],nan=0).sum(axis=1); mi=int(np.argmax(quality)); pts=kp[ti,mi]; scores=np.nan_to_num(sc[ti,mi],nan=0); valid=np.isfinite(pts).all(1)&(scores>.05)
        xy=pts[valid];
        if len(xy):
            x1,y1=xy.min(0).astype(int); x2,y2=xy.max(0).astype(int); pad=25; x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(w-1,x2+pad); y2=min(h-1,y2+pad)
            col=(0,208,132) if rec_name==true_name else (0,138,255); cv2.rectangle(frame,(x1,y1),(x2,y2),col,4)
            label=f'{aid} {rec_name}  {prob}'; cv2.rectangle(frame,(x1,max(0,y1-42)),(x1+430,max(0,y1-4)),col,-1); cv2.putText(frame,label,(x1+10,max(28,y1-12)),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),2,cv2.LINE_AA)
            def p(i): return (int(pts[i,0]),int(pts[i,1]))
            for a,b in body_edges:
                if valid[a] and valid[b]: cv2.line(frame,p(a),p(b),(235,80,37),4)
            for base,color in ((91,(165,0,181)),(112,(126,49,196))):
                for a,b in hand_edges:
                    if valid[base+a] and valid[base+b]: cv2.line(frame,p(base+a),p(base+b),color,3)
            for i in list(range(23))+list(range(91,133)):
                if valid[i]:
                    color=(165,0,181) if 91<=i<=111 else ((126,49,196) if i>=112 else (235,80,37)); cv2.circle(frame,p(i),6,color,-1)
        cv2.putText(frame,f'RTMW-133 | frame {fi+1}/{n} | 识别动作：{aid} {rec_name}',(24,h-24),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA)
        writer.write(frame)
    cap.release(); writer.release()
print(OUT)
