"""Local camera client for fully remote action inference.

Only camera capture, JPEG encoding and display run on Windows.  The SSH
process performs detection, pose extraction, 100-frame buffering, HF/BF
accuracy-ratio fusion and independent CTR-GCN classification on the remote GPU.
"""
from __future__ import annotations

import argparse
import base64
import cv2
import numpy as np
import pickle
import shlex
import struct
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.realtime_pool_camera import NTU60_NAMES

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"


def draw_unicode(frame, lines, *, x=10, y=10, color=(0, 255, 0), size=22):
    """Draw Chinese text through Pillow; cv2.putText has no CJK glyphs."""
    if not lines:
        return frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT_PATH, size)
    except OSError:
        font = ImageFont.load_default()
    for line in lines:
        draw.text((x, y), line, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
        y += size + 6
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return frame


def draw_person_boxes(frame, boxes, color):
    """Draw one detector's boxes without placing text beside the people."""
    if not boxes:
        return frame
    h, w = frame.shape[:2]
    for raw_box in boxes:
        box = np.asarray(raw_box, dtype=np.float32).reshape(-1)
        if box.size != 4 or not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    return frame


def draw_status_panel(frame, lines, *, x=10, y=10, size=17):
    """Draw one fixed readout panel so labels from the two models cannot collide."""
    if not lines:
        return frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT_PATH, size)
    except OSError:
        font = ImageFont.load_default()
    clean_lines = [str(line) for line in lines if str(line)]
    max_width = min(frame.shape[1] - 2 * x, 620)
    measured = [draw.textbbox((0, 0), line, font=font) for line in clean_lines]
    text_width = min(max_width - 20, max((box[2] - box[0] for box in measured), default=0))
    line_height = size + 5
    panel_height = len(clean_lines) * line_height + 14
    panel_width = max(220, text_width + 20)
    draw.rounded_rectangle((x, y, x + panel_width, y + panel_height),
                           radius=7, fill=(12, 12, 12), outline=(210, 210, 210), width=1)
    for index, line in enumerate(clean_lines):
        draw.text((x + 10, y + 7 + index * line_height), line, font=font, fill=(255, 255, 255))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return frame


REMOTE_PROGRAM = r'''import argparse, os, pickle, struct, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from collections import deque
# The remote torch wheel ships CUDA/cuDNN libraries under the user site. Put
# those directories first before importing torch; otherwise an older system
# cuDNN can be selected and fail during the first convolution.
_nvidia = '/home/xiong/.local/lib/python3.10/site-packages/nvidia'
_cuda_libs = []
if os.path.isdir(_nvidia):
  for _name in os.listdir(_nvidia):
    _lib = os.path.join(_nvidia, _name, 'lib')
    if os.path.isdir(_lib): _cuda_libs.append(_lib)
# PyTorch 2.14 may request a fully versioned cuDNN 9.24 engine-runtime
# filename, while the installed wheel exposes the same binary under its SONAME
# only. Create local aliases before importing torch so every fresh SSH process
# can initialize cuDNN without a manual server-side repair.
_cudnn_lib = os.path.join(_nvidia, 'cudnn', 'lib')
_cudnn_target = os.path.join(_cudnn_lib, 'libcudnn_engines_runtime_compiled.so.9')
if os.path.isfile(_cudnn_target):
  for _alias in ('libcudnn_engines_runtime_compiled.so',
                 'libcudnn_engines_runtime_compiled.so.9.24',
                 'libcudnn_engines_runtime_compiled.so.9.24.0'):
    _alias_path = os.path.join(_cudnn_lib, _alias)
    if not os.path.exists(_alias_path):
      try:
        os.symlink(_cudnn_target, _alias_path)
      except OSError:
        pass
if _cuda_libs:
  _filtered = [x for x in os.environ.get('LD_LIBRARY_PATH','').split(':') if x and '/usr/local/cuda' not in x and '/usr/local/lib64' not in x and '/nvidia/' not in x.lower()]
  _desired = ':'.join(_cuda_libs + _filtered)
  if os.environ.get('ISAA_CUDNN_ENV_FIXED') != '1' and _desired != os.environ.get('LD_LIBRARY_PATH',''):
    os.environ['LD_LIBRARY_PATH'] = _desired
    os.environ['ISAA_CUDNN_ENV_FIXED'] = '1'
    _argv = getattr(sys, 'orig_argv', None)
    if _argv: os.execvpe(sys.executable, [sys.executable, *_argv[1:]], os.environ)
protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno()); sys.stdout = sys.stderr
import cv2, numpy as np, torch
# FACE_PAIR_HELPERS

def repair_pool_anchor_gaps(frames, max_gap=5):
    """Apply the frozen bounded shoulder/hip interpolation on the server."""
    values = np.stack(tuple(frames), axis=0).astype(np.float32, copy=True)
    for person in range(values.shape[1]):
      for joint in (5, 6, 11, 12):
        valid = np.isfinite(values[:, person, joint, :2]).all(axis=-1) & (values[:, person, joint, 2] > 0)
        for index in range(values.shape[0]):
          if valid[index]: continue
          left=index-1
          while left >= 0 and not valid[left]: left-=1
          right=index+1
          while right < values.shape[0] and not valid[right]: right+=1
          if left < 0 or right >= values.shape[0] or right-left-1 > max_gap: continue
          alpha=(index-left)/float(right-left)
          values[index, person, joint] = values[left, person, joint]*(1-alpha)+values[right, person, joint]*alpha
          valid[index]=True
    return [values[index] for index in range(values.shape[0])]

def coco17_to_ntu25_independent(points, scores):
    """Build NTU25 from a fresh COCO17 pose extracted by the CTR branch."""
    p = np.asarray(points, dtype=np.float32)
    c = np.asarray(scores, dtype=np.float32)
    valid = np.isfinite(p).all(axis=1) & np.isfinite(c) & (c > 0)
    if p.shape != (17, 2) or c.shape != (17,) or int(valid.sum()) < 5:
        return np.zeros((25, 3), dtype=np.float32), False
    def mid(a, b):
        values = [p[i] for i in (a, b) if valid[i]]
        return np.mean(values, axis=0) if values else np.zeros(2, dtype=np.float32)
    hip = mid(11, 12)
    shoulder = mid(5, 6)
    if not valid[11] and not valid[12]: hip = np.mean(p[valid], axis=0)
    if not valid[5] and not valid[6]: shoulder = hip.copy()
    extent = np.ptp(p[valid], axis=0)
    scale = max(float(extent.max()), 1.0)
    q = np.zeros((17, 3), dtype=np.float32)
    q[:, 0] = (p[:, 0] - hip[0]) / scale
    q[:, 1] = (hip[1] - p[:, 1]) / scale
    q[:, 2] = 0.15 * q[:, 1]
    q[~valid] = 0
    out = np.zeros((25, 3), dtype=np.float32)
    out[0] = 0
    out[1] = 0.25 * (q[5] + q[6])
    out[2] = 0.5 * (q[5] + q[6])
    out[3] = q[0] if valid[0] else 0.5 * (q[1] + q[2])
    out[4], out[5], out[6] = q[5], q[7], q[9]
    out[8], out[9], out[10] = q[6], q[8], q[10]
    out[12], out[13], out[14] = q[11], q[13], q[15]
    out[16], out[17], out[18] = q[12], q[14], q[16]
    out[20] = 0.5 * (q[5] + q[6])
    out[7] = out[6] + 0.35 * (out[6] - out[5])
    out[11] = out[10] + 0.35 * (out[10] - out[9])
    out[15] = out[14] + 0.35 * (out[14] - out[13])
    out[19] = out[18] + 0.35 * (out[18] - out[17])
    out[21] = out[6] + 0.55 * (out[6] - out[5])
    out[22] = out[6] + 0.25 * (out[6] - out[5])
    out[23] = out[10] + 0.55 * (out[10] - out[9])
    out[24] = out[10] + 0.25 * (out[10] - out[9])
    return out, True

class IndependentYOLOPose25:
    """Separate pose network; it never reads RTMW-133 keypoints."""
    def __init__(self, model_path, device, conf, imgsz, max_persons=2):
        from ultralytics import YOLO
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                '独立 CTR-GCN 姿态权重不存在: ' + model_path +
                '; 请把 yolo11n-pose.pt 放到远端 models 目录')
        self.model = YOLO(model_path)
        self.device = None if str(device) == 'cpu' else device
        self.conf, self.imgsz, self.max_persons = float(conf), int(imgsz), int(max_persons)
    def __call__(self, frame, frame_id, captured_at):
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                     device=self.device, verbose=False)
        if not results or results[0].keypoints is None:
            return {'keypoints3d': np.zeros((self.max_persons,25,3), dtype=np.float32),
                    'bboxes': np.full((self.max_persons, 4), np.nan, dtype=np.float32),
                    'bbox_scores': np.zeros((self.max_persons,), dtype=np.float32),
                    'count': 0, 'frame_id': frame_id, 'captured_at': captured_at}
        data = results[0].keypoints.data.detach().cpu().numpy().astype(np.float32)
        if data.ndim != 3 or data.shape[1:] != (17,3):
            raise RuntimeError('independent pose output shape is '+str(data.shape))
        order = np.argsort(-data[...,2].mean(axis=1))[:self.max_persons]
        boxes = np.full((self.max_persons, 4), np.nan, dtype=np.float32)
        box_scores = np.zeros((self.max_persons,), dtype=np.float32)
        detected_boxes = getattr(results[0], 'boxes', None)
        xyxy = getattr(detected_boxes, 'xyxy', None)
        conf = getattr(detected_boxes, 'conf', None)
        if xyxy is not None:
            xyxy = xyxy.detach().cpu().numpy().astype(np.float32)
        if conf is not None:
            conf = conf.detach().cpu().numpy().astype(np.float32)
        out = np.zeros((self.max_persons,25,3), dtype=np.float32); count = 0
        for slot, index in enumerate(order):
            pose, ok = coco17_to_ntu25_independent(data[index,:,:2], data[index,:,2])
            if xyxy is not None and int(index) < len(xyxy) and xyxy[index].shape == (4,):
                boxes[slot] = xyxy[index]
                box_scores[slot] = float(conf[index]) if conf is not None and int(index) < len(conf) else 1.0
            elif np.isfinite(data[index, :, :2]).all(axis=1).any():
                valid = data[index, :, 2] > 0
                points = data[index, valid, :2]
                if len(points):
                    boxes[slot] = (points[:, 0].min(), points[:, 1].min(),
                                   points[:, 0].max(), points[:, 1].max())
                    box_scores[slot] = float(data[index, :, 2].mean())
            if ok:
                out[slot], count = pose, count + 1
        return {'keypoints3d': out, 'bboxes': boxes, 'bbox_scores': box_scores,
                'count': count, 'frame_id': frame_id, 'captured_at': captured_at}

def read_exact(s, n):
    b=b''
    while len(b)<n:
        x=s.read(n-len(b))
        if not x: return None
        b+=x
    return b
def read_packet(s):
    h=read_exact(s,4)
    if h is None: return None
    n=struct.unpack('!I',h)[0]
    if n>32*1024*1024: raise ValueError('packet too large')
    return pickle.loads(read_exact(s,n))
def write_packet(x):
    b=pickle.dumps(x,protocol=5); protocol_output.write(struct.pack('!I',len(b))+b); protocol_output.flush()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pose-project',required=True); p.add_argument('--isaa-project',required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--yolo-model',required=True)
    p.add_argument('--ctrgcn-pose-model',required=True)
    p.add_argument('--ctrgcn-pose-conf',type=float,default=.25); p.add_argument('--ctrgcn-pose-imgsz',type=int,default=640)
    p.add_argument('--face-project',required=True)
    p.add_argument('--face-checkpoint',required=True)
    p.add_argument('--face-summary',required=True)
    p.add_argument('--ctrgcn-checkpoint',required=True)
    p.add_argument('--yolo-conf',type=float,default=.35); p.add_argument('--yolo-iou',type=float,default=.5)
    p.add_argument('--yolo-imgsz',type=int,default=640); p.add_argument('--kpt-thr',type=float,default=.1)
    p.add_argument('--pool-person-threshold',type=float,default=.35)
    p.add_argument('--pool-body-threshold',type=float,default=.25)
    p.add_argument('--timeout',type=float,default=45.); p.add_argument('--infer-every',type=int,default=1)
    a=p.parse_args(); sys.path[:0]=[a.pose_project,a.face_project,a.isaa_project]
    from ntu_rtmw.compat import patch_runtime
    patch_runtime()
    from ntu_rtmw.camera import PoseWorker
    from ntu_rtmw.constants import RTMDET_CONFIG, RTMDET_WEIGHTS_PATH, RTMW_CONFIG, RTMW_WEIGHTS_PATH
    device=torch.device(a.device)
    torch.set_num_threads(4)
    # Load the face-pair package from its isolated deployment before importing
    # camera helpers, which prepend the main project root to sys.path.
    pair_model, pair_weight, pair_info = load_face_pair_camera(
        a.face_checkpoint, a.face_summary, device)
    from tools.realtime_pool_camera import filter_people, RecognitionWindow, make_input, NTU60_NAMES
    from tools.ctrgcn25_pose import CTRGCN25Window, make_input as make25
    from isaa.models.original_ctrgcn import OriginalCTRGCN, build_official_ntu_adjacency
    ctr_ckpt=torch.load(a.ctrgcn_checkpoint,map_location='cpu',weights_only=False)
    ctr_state_dict=ctr_ckpt.get('model',ctr_ckpt)
    if 'data_bn.weight' not in ctr_state_dict: raise RuntimeError('remote CTR-GCN checkpoint is not an official raw state_dict')
    ctr=OriginalCTRGCN(num_classes=60,num_point=25,num_person=2,graph=build_official_ntu_adjacency())
    ctr.load_state_dict(ctr_state_dict,strict=True); ctr=ctr.to(device).eval()
    independent_pose = IndependentYOLOPose25(
        a.ctrgcn_pose_model, a.device, a.ctrgcn_pose_conf,
        a.ctrgcn_pose_imgsz, max_persons=2)
    pose_args=argparse.Namespace(pose2d=RTMW_CONFIG,pose2d_weights=str(RTMW_WEIGHTS_PATH),det_model=RTMDET_CONFIG,det_weights=str(RTMDET_WEIGHTS_PATH),device=a.device,det_backend='yolo',yolo_model=a.yolo_model,yolo_conf=a.yolo_conf,yolo_iou=a.yolo_iou,yolo_imgsz=a.yolo_imgsz,pose_input='crop',crop_margin=0.,max_persons=2,bbox_thr=.3,kpt_thr=a.kpt_thr,tracking_distance=120.,temporal_min_frames=2,temporal_max_jump=150.,temporal_min_keypoints=5,filter_output_to_bbox=True,output_bbox_margin=0.)
    pose=PoseWorker(pose_args).start()
    pose_executor=ThreadPoolExecutor(max_workers=1)
    pool_state=RecognitionWindow(100,3); ctr_state=CTRGCN25Window(100,2)
    last_pose=None; last_cls=0.; remote_pose_seq=-1; cached={}; people=None
    try:
      while True:
        req=read_packet(sys.stdin.buffer)
        if req is None: break
        frame=cv2.imdecode(np.frombuffer(req['jpeg'],dtype=np.uint8),cv2.IMREAD_COLOR)
        if frame is None: raise RuntimeError('remote JPEG decode failed')
        # Run the independent CTR-GCN pose network concurrently with RTMW.
        # The two models still receive the same raw image independently.
        pose.submit(frame)
        independent_future = pose_executor.submit(
            independent_pose, frame, req['frame_id'], req['captured_at'])
        deadline=time.monotonic()+a.timeout
        while time.monotonic()<deadline:
          latest,busy,error=pose.snapshot()
          if error is not None: raise RuntimeError(str(error))
          if latest is not None and latest is not last_pose:
            last_pose=latest; remote_pose_seq += 1
            people=filter_people(latest,frame.shape,person_threshold=a.pool_person_threshold,body_threshold=a.pool_body_threshold,min_body_points=6)
            pool_state.observe(people,remote_pose_seq)
            break
          time.sleep(.002)
        else:
          raise TimeoutError('RTMW inference timed out')
        independent_result = independent_future.result()
        ctr_state.observe(independent_result)
        # Clear each branch as soon as its own current observation is absent.
        if not pool_state.complete:
          for key in ('best','weighted','geometric_mix','HF','BF'):
            cached.pop(key,None); cached.pop(key+'_info',None)
        ctr_ready = ctr_state.complete and independent_result['count'] > 0
        if not ctr_ready:
          cached.pop('ctrgcn25',None); cached.pop('ctrgcn25_info',None)
        if time.monotonic()-last_cls>=1.0 and (pool_state.complete or ctr_ready):
          if pool_state.complete:
            x,m=make_input(repair_pool_anchor_gaps(pool_state.frames),window=100,max_persons=2)
            cached.update(face_pair_top1(pair_model,pair_weight,x.to(device),m.to(device)))
            for key, info in pair_info.items(): cached[key+'_info']=info
          if ctr_ready:
            x25=make25(ctr_state.frames,int(ctr_ckpt.get('args',{}).get('window_size',64)),2).to(device)
            with torch.inference_mode(): cp=ctr(x25).softmax(-1)[0]
            confidence,index=cp.max(-1)
            cached['ctrgcn25']=[(int(index),float(confidence))]
            cached['ctrgcn25_info']={'validation_top1':None,
              'note':'RGB-derived NTU25; no labeled webcam accuracy measured'}
          last_cls=time.monotonic()
        pool_person_boxes=[]
        if people is not None:
          for bbox, score in zip(people.get('bboxes', []), people.get('bbox_scores', [])):
            bbox=np.asarray(bbox, dtype=np.float32)
            if bbox.shape == (4,) and np.isfinite(bbox).all() and float(score) > 0:
              pool_person_boxes.append([float(v) for v in bbox])
        ctr_person_boxes=[]
        for bbox, score in zip(independent_result.get('bboxes', []), independent_result.get('bbox_scores', [])):
          bbox=np.asarray(bbox, dtype=np.float32)
          if bbox.shape == (4,) and np.isfinite(bbox).all() and float(score) > 0:
            ctr_person_boxes.append([float(v) for v in bbox])
        out={'frame_id':req['frame_id'],'captured_at':req['captured_at'],'action':dict(cached) or None,
             'pool_count':len(pool_state.frames),'ctrgcn_count':len(ctr_state.frames),
             'ctrgcn_valid_frames':int(ctr_state.valid_frames),
             'ctrgcn_pose_source':'independent_yolo11n_pose_coco17_to_ntu25',
             'pool_pose_source':'independent RTMW-133 + YOLO person boxes',
             'pool_person_boxes':pool_person_boxes,
             'ctrgcn_person_boxes':ctr_person_boxes,
             # Compatibility alias for older local clients: it always means
             # the best-model/RTMW boxes, never the CTR-GCN boxes.
             'person_boxes':pool_person_boxes}
        write_packet(out)
    finally:
      pose_executor.shutdown(wait=True)
      pose.stop()
if __name__=='__main__':
  try: main()
  except Exception as e:
    traceback.print_exc(file=sys.stderr); raise
'''

REMOTE_PROGRAM = REMOTE_PROGRAM.replace(
    '# FACE_PAIR_HELPERS', (ROOT / 'tools/camera_face_pair.py').read_text(encoding='utf-8'))


def _packet(v):
    b = pickle.dumps(v, protocol=5)
    return struct.pack("!I", len(b)) + b


def _read_packet(stream):
    h = stream.read(4)
    if not h:
        return None
    while len(h) < 4:
        part = stream.read(4 - len(h))
        if not part:
            raise ConnectionError('truncated packet header')
        h += part
    n = struct.unpack("!I", h)[0]
    payload = b''
    while len(payload) < n:
        part = stream.read(n - len(payload))
        if not part:
            raise ConnectionError('truncated packet payload')
        payload += part
    return pickle.loads(payload)


class RemoteActionClient:
    def __init__(self, args):
        self.args=args
        self.pending=deque()
        self.batch_size=max(1, int(getattr(args, 'batch_size', 32)))
        self.max_pending=max(self.batch_size, int(getattr(args, 'max_pending_frames', 600)))
        self.batch_wait=max(0., float(getattr(args, 'batch_wait_ms', 80)) / 1000.)
        self.dropped_frames=0
        self.submitted_frames=0; self.processed_frames=0; self.in_flight=0
        self.latest=None; self.error=None; self.busy=False; self.stop=False
        self.cv=threading.Condition(); self.proc=None; self.stderr_tail=[]
        self.thread=threading.Thread(target=self._run,daemon=True)

    def start(self): self.thread.start(); return self
    def submit(self, frame, frame_id):
        ok,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,self.args.jpeg_quality])
        if not ok:
            with self.cv: self.dropped_frames += 1
            return False
        captured_at=time.perf_counter()
        with self.cv:
            if len(self.pending) >= self.max_pending:
                self.dropped_frames += 1
                return False
            self.pending.append((jpeg.tobytes(),frame_id,captured_at))
            self.submitted_frames += 1
            self.cv.notify_all()
            return True
    def snapshot(self):
        with self.cv: return self.latest,self.busy,self.error
    def stats(self):
        with self.cv:
            return {'submitted':self.submitted_frames, 'processed':self.processed_frames,
                    'dropped':self.dropped_frames, 'queued':len(self.pending),
                    'in_flight':self.in_flight}
    def close(self):
        with self.cv: self.stop=True; self.cv.notify_all()
        if self.proc and self.proc.poll() is None: self.proc.terminate()
        self.thread.join(5)
    def _drain_stderr(self):
        while True:
            data=self.proc.stderr.read1(4096)
            if not data: return
            self.stderr_tail.append(data.decode('utf-8','replace'))
            self.stderr_tail=self.stderr_tail[-12:]
    def _run(self):
        try:
            a=self.args; code=base64.b64encode(REMOTE_PROGRAM.encode()).decode()
            remote=[a.remote_python,'-u','-c',f"import base64;exec(base64.b64decode('{code}'))",'--pose-project',a.remote_pose_project,'--isaa-project',a.remote_isaa_project,'--device',a.remote_device,'--yolo-model',a.remote_yolo_model,'--ctrgcn-pose-model',a.remote_ctrgcn_pose_model,'--face-project',a.remote_face_project,'--face-checkpoint',a.remote_face_checkpoint,'--face-summary',a.remote_face_summary,'--ctrgcn-checkpoint',a.remote_ctrgcn_checkpoint,'--yolo-conf',str(a.yolo_conf),'--yolo-iou',str(a.yolo_iou),'--yolo-imgsz',str(a.yolo_imgsz),'--pool-person-threshold',str(a.pool_person_threshold),'--pool-body-threshold',str(a.pool_body_threshold),'--ctrgcn-pose-conf',str(a.ctrgcn_pose_conf),'--ctrgcn-pose-imgsz',str(a.ctrgcn_pose_imgsz),'--timeout',str(a.remote_timeout)]
            command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-p',str(a.ssh_port)]
            if a.ssh_key: command += ['-i',a.ssh_key]
            command += [f'{a.ssh_user}@{a.ssh_host}',' '.join(shlex.quote(str(x)) for x in remote)]
            self.proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            threading.Thread(target=self._drain_stderr,daemon=True).start()
            while True:
                with self.cv:
                    while not self.pending and not self.stop: self.cv.wait()
                    if self.stop: return
                    batch_deadline=time.monotonic()+self.batch_wait
                    while len(self.pending)<self.batch_size and not self.stop:
                        remaining=batch_deadline-time.monotonic()
                        if remaining<=0: break
                        self.cv.wait(remaining)
                    if self.stop: return
                    batch=[]
                    while self.pending and len(batch) < self.batch_size:
                        batch.append(self.pending.popleft())
                    self.busy=True
                    self.in_flight=len(batch)
                packets=[_packet({'frame_id':fid,'captured_at':captured_at,'jpeg':jpeg})
                         for jpeg,fid,captured_at in batch]
                self.proc.stdin.write(b''.join(packets)); self.proc.stdin.flush()
                for _ in batch:
                    out=_read_packet(self.proc.stdout)
                    if out is None: raise RuntimeError(''.join(self.stderr_tail)[-4000:])
                    with self.cv:
                        self.latest=out
                        self.error=None
                        self.processed_frames += 1
                        self.in_flight -= 1
                        self.cv.notify_all()
                with self.cv:
                    self.busy=False
                    self.in_flight=0
                    self.cv.notify_all()
        except Exception as e:
            with self.cv: self.error=e; self.busy=False; self.in_flight=0; self.cv.notify_all()


def _format_action_lines(action):
    """Show only Top1, separating live confidence from validation accuracy."""
    specs = (
        ('best', '最新最佳模型（几何融合）'),
        ('ctrgcn25', 'CTR-GCN'),
        ('weighted', '最佳模型原始算术融合'),
        ('HF', 'HF 手部+面部'),
        ('BF', 'BF 身体+面部'),
    )
    lines = []
    report = []
    seen = set()
    for key, label in specs:
        if key in seen:
            continue
        seen.add(key)
        values = action.get(key) or []
        info = action.get(key + '_info') or {}
        validation = info.get('validation_top1')
        header = label
        values_text = []
        for rank, pair in enumerate(values[:1], start=1):
            index, confidence = int(pair[0]), float(pair[1])
            name = NTU60_NAMES[index] if 0 <= index < len(NTU60_NAMES) else f'A{index + 1:03d}'
            values_text.append(f'Top{rank} {name} {confidence:.1%}')
        suffix = f' | 验证准确率 {float(validation):.2%}' if validation is not None else ''
        line = header + (' '+values_text[0] if values_text else ' 等待100帧') + suffix
        lines.append(line)
        report.append(line)
    return lines, '; '.join(report)


def _compact_top1(action, key, label):
    values = (action or {}).get(key) or []
    if not values:
        return f'{label}：等待100帧'
    index, confidence = int(values[0][0]), float(values[0][1])
    name = NTU60_NAMES[index] if 0 <= index < len(NTU60_NAMES) else f'A{index + 1:03d}'
    return f'{label}：{name} {confidence:.1%}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--ssh-host', default='100.123.254.107')
    p.add_argument('--ssh-user', default='xiong')
    p.add_argument('--ssh-port', type=int, default=22)
    p.add_argument('--ssh-key', default=str(Path.home() / '.ssh' / 'isaa_codex'))
    p.add_argument('--remote-python', default='/home/xiong/venvs/isaa-camera/bin/python')
    p.add_argument('--remote-pose-project', default='/home/xiong/Desktop/nturgbd_skeleton_extraction')
    p.add_argument('--remote-isaa-project', default='/home/xiong/Desktop/ISAA')
    p.add_argument('--remote-face-project',
                   default='/home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103/code')
    p.add_argument('--remote-face-checkpoint',
                   default='/home/xiong/Desktop/ISAA/outputs/best_face_pair_stable_100_20261103/20261103-052121-a73ca490/best.pt')
    p.add_argument('--remote-face-summary',
                   default='/home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103/verification/best_epoch83_accuracy_weighted/summary.json')
    p.add_argument('--remote-device', default='cuda:0')
    p.add_argument('--remote-yolo-model',
                   default='/home/xiong/Desktop/nturgbd_skeleton_extraction/models/yolov8n.pt')
    p.add_argument('--remote-ctrgcn-pose-model',
                   default='/home/xiong/Desktop/nturgbd_skeleton_extraction/models/yolo11n-pose.pt')
    p.add_argument('--remote-ctrgcn-checkpoint',
                   default='/home/xiong/Desktop/ISAA/runtime_ctrgcn25_raw.pt')
    p.add_argument('--remote-timeout', type=float, default=60)
    p.add_argument('--jpeg-quality', type=int, default=70)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=360)
    p.add_argument('--capture-fps', type=float, default=12.0,
                   help='camera sampling rate; 0 means capture as fast as possible')
    p.add_argument('--yolo-conf', type=float, default=.35)
    p.add_argument('--yolo-iou', type=float, default=.5)
    p.add_argument('--yolo-imgsz', type=int, default=640)
    p.add_argument('--pool-person-threshold', type=float, default=.35,
                   help='RTMW branch accepted person-box confidence')
    p.add_argument('--pool-body-threshold', type=float, default=.25,
                   help='RTMW branch accepted body-keypoint confidence')
    p.add_argument('--ctrgcn-pose-conf', type=float, default=.25)
    p.add_argument('--ctrgcn-pose-imgsz', type=int, default=640)
    p.add_argument('--batch-size', type=int, default=32,
                   help='maximum JPEG frames written to SSH together')
    p.add_argument('--max-pending-frames', type=int, default=600,
                   help='bounded client queue; excess new frames are counted as dropped')
    p.add_argument('--batch-wait-ms', type=float, default=80,
                   help='maximum initial wait to fill a transport batch')
    p.add_argument('--kpt-thr', type=float, default=.1)
    p.add_argument('--fullscreen', action=argparse.BooleanOptionalAction, default=True)
    a=p.parse_args(); cap=cv2.VideoCapture(a.camera,cv2.CAP_DSHOW)
    if not cap.isOpened(): cap=cv2.VideoCapture(a.camera)
    if not cap.isOpened(): raise RuntimeError('Cannot open camera')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,a.height)
    client=RemoteActionClient(a).start(); i=0; last_report=None
    window_name='ISAA remote full action inference'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if a.fullscreen:
      try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
      except cv2.error:
        pass
    capture_period = 1.0 / a.capture_fps if a.capture_fps > 0 else 0.0
    next_capture = time.perf_counter()
    try:
      while True:
        if capture_period:
          now = time.perf_counter()
          if now < next_capture:
            time.sleep(next_capture - now)
          now = time.perf_counter()
          next_capture = now + capture_period if now > next_capture + capture_period else next_capture + capture_period
        ok,frame=cap.read()
        if not ok: raise RuntimeError('camera read failed')
        latest,busy,error=client.snapshot()
        client.submit(frame,i)
        stats=client.stats()
        overlay_lines = []
        pool_person_boxes = latest.get('pool_person_boxes', latest.get('person_boxes', [])) if latest else []
        ctr_person_boxes = latest.get('ctrgcn_person_boxes', []) if latest else []
        if error is not None:
          if last_report != ('error', str(error)): print('远端推理失败:', error, flush=True); last_report=('error',str(error))
          overlay_lines = ['SSH远端推理失败', str(error)[:70]]
        elif latest and latest.get('action'):
          action=latest['action']
          overlay_lines = [_compact_top1(action,'best','最新最佳模型'),
                           _compact_top1(action,'ctrgcn25','CTR-GCN 原论文')]
          _, report_text = _format_action_lines(action)
          report=('action', repr(action))
          if report != last_report:
            print('远端动作:', report_text, flush=True); last_report=report
        else:
          pool_count=latest.get('pool_count',0) if latest else 0
          ctr_count=latest.get('ctrgcn_count',0) if latest else 0
          overlay_lines = [f'收集帧 RTMW {pool_count}/100 | CTR-GCN {ctr_count}/100']
          if latest and max(pool_count,ctr_count) and max(pool_count,ctr_count) % 10 == 0:
            report=('count',pool_count,ctr_count)
            if last_report != report:
              print(f'远端有效帧: RTMW {pool_count}/100, CTR-GCN {ctr_count}/100', flush=True)
              last_report=report
        if latest:
          processed_id=int(latest.get('frame_id',-1))
          lag=max(0,i-processed_id)
          age=max(0.,time.perf_counter()-float(latest.get('captured_at',time.perf_counter())))
          overlay_lines.append(f'处理帧 {processed_id} | 落后 {lag} 帧 / {age:.1f} 秒')
          overlay_lines.append(f'窗口 RTMW {latest.get("pool_count",0)}/100 | CTR-GCN {latest.get("ctrgcn_count",0)}/100')
        overlay_lines.append(f'已提交 {stats["submitted"]} | 已返回 {stats["processed"]} | 排队 {stats["queued"]+stats["in_flight"]} | 丢帧 {stats["dropped"]}')
        overlay_lines.append('绿色框：最佳模型 RTMW | 橙色框：独立 CTR-GCN')
        if latest and lag <= 5:
          draw_person_boxes(frame,pool_person_boxes,(40,220,40))
          draw_person_boxes(frame,ctr_person_boxes,(0,165,255))
        draw_status_panel(frame,overlay_lines,size=17)
        cv2.imshow(window_name,frame)
        if cv2.waitKey(1)&0xff in (ord('q'),27): break
        i+=1
    except KeyboardInterrupt:
      pass
    finally:
      stats=client.stats()
      print('传输统计: 提交 {submitted}, 返回 {processed}, 排队 {queued}, 丢帧 {dropped}'.format(**stats), flush=True)
      client.close(); cap.release(); cv2.destroyAllWindows()

if __name__=='__main__':
  try:
    main()
  except KeyboardInterrupt:
    pass
