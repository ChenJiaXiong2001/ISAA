import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const W = 1280, H = 720;
const buildDir = 'C:/Users/15637/PycharmProjects/ISAA/.ppt_build';
const outDir = 'C:/Users/15637/PycharmProjects/ISAA/outputs';
const draft = path.join(buildDir, 'research_study_draft.pptx');
const finalPath = path.join(outDir, '高精度全身骨架细粒度动作识别研究汇报.pptx');
await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outDir, { recursive: true });

const deck = Presentation.create({ slideSize: { width: W, height: H } });
const C = { bg: 'F7F9FC', navy: '17324D', blue: '2F6B9A', teal: '2A9D8F', orange: 'E58F3A', text: '203040', muted: '617384', light: 'E8EEF4', white: 'FFFFFF', red: 'C95C5C' };
const font = 'Aptos';

function box(slide, x, y, w, h, fill='none', line='none', radius='rounded-lg') {
  const cfg = { geometry: radius === 'none' ? 'rect' : 'roundRect', position:{left:x,top:y,width:w,height:h}, fill, line:{style:'solid',fill:line,width:line==='none'?0:1} };
  if (radius !== 'none') cfg.borderRadius = radius;
  return slide.shapes.add(cfg);
}
function text(slide, str, x, y, w, h, size=22, color=C.text, bold=false, align='left') {
  const s = slide.shapes.add({ geometry:'textbox', position:{left:x,top:y,width:w,height:h}, fill:'none', line:{style:'solid',fill:'none',width:0} });
  s.text = str; s.text.style = { fontFamily:font, fontSize:size, color, bold, align, verticalAlign:'middle' }; return s;
}
function title(slide, str, kicker='本研究') {
  text(slide, kicker.toUpperCase(), 72, 34, 400, 24, 13, C.blue, true);
  text(slide, str, 72, 64, 1120, 54, 32, C.navy, true);
  box(slide, 72, 128, 1136, 2, C.blue, 'none', 'none');
}
function footer(slide, n) { text(slide, `本研究  ·  ${n}`, 1090, 682, 118, 18, 11, C.muted, false, 'right'); }
function bullet(slide, str, x, y, w, size=20, color=C.text) {
  text(slide, '•', x, y, 18, 24, size+2, C.blue, true); text(slide, str, x+24, y, w-24, 46, size, color, false);
}
function pill(slide, str, x, y, w, fill, color=C.white) { box(slide,x,y,w,34,fill,'none','rounded-xl'); text(slide,str,x+10,y+3,w-20,26,14,color,true,'center'); }
function node(slide, label, x, y, w, h, fill, fs=18) { box(slide,x,y,w,h,fill,'none','rounded-xl'); text(slide,label,x+10,y+8,w-20,h-16,fs,C.white,true,'center'); }
function connector(slide,x1,y1,x2,y2,color=C.muted) { slide.shapes.add({ geometry:'line', position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.max(2,Math.abs(x2-x1)),height:Math.max(2,Math.abs(y2-y1))}, line:{style:'solid',fill:color,width:2,beginArrowType:'none',endArrowType:'triangle'} }); }

// 1 cover
{
 const s=deck.slides.add(); s.background.fill=C.bg;
 box(s,0,0,W,H,C.bg,'none','none');
 box(s,72,106,10,420,C.blue,'none','none');
 text(s,'基于高精度全身骨架的\n细粒度人体动作识别研究',120,120,900,190,46,C.navy,true);
 text(s,'研究背景、数据处理、技术依据与方法设计',124,336,780,42,22,C.muted);
 pill(s,'NTU RGB+D  ·  RTMW-133  ·  时空图建模',124,430,440,C.teal);
 text(s,'研究汇报',124,602,220,30,16,C.blue,true); text(s,'2026',1060,602,100,30,16,C.muted,false,'right');
}

// 2 problem
{
 const s=deck.slides.add(); s.background.fill=C.bg; title(s,'研究问题：整体姿态相似时，局部细节决定分类');
 box(s,72,176,520,400,C.white,'D7E0E8','rounded-xl');
 text(s,'传统少节点骨架',104,204,380,32,22,C.navy,true);
 bullet(s,'能够描述身体整体姿态和大范围运动',104,260,420,19);
 bullet(s,'对手腕、手指和面部细节表达有限',104,326,420,19);
 bullet(s,'相似动作容易被映射到同一类别',104,392,420,19);
 box(s,636,176,572,400,'EAF4F2','none','rounded-xl');
 text(s,'本研究要回答的问题',672,204,440,32,22,C.teal,true);
 text(s,'高精度全身骨架能否提供稳定的局部动作证据，帮助模型区分身体姿态相近的动作？',672,268,460,112,29,C.navy,true);
 text(s,'关键变量',672,414,180,26,16,C.muted,true);
 pill(s,'手部',672,454,100,C.blue); pill(s,'面部',786,454,100,C.orange); pill(s,'时间阶段',900,454,132,C.teal);
 footer(s,2);
}

// 3 research flow
{
 const s=deck.slides.add(); s.background.fill=C.bg; title(s,'研究流程：从视频到可解释的动作分类');
 const labels=[['RGB 视频','NTU RGB+D'],['人物检测','定位目标人物'],['RTMW-133','133 点全身骨架'],['质量控制','缺失、跳变、槽位'],['分支建模','身体、手部、面部'],['动作分类','NTU60 XSub']];
 const xs=[72,260,448,636,824,1012];
 labels.forEach((a,i)=>{ node(s,a[0],xs[i],250,150,78,i===2?C.blue:(i===4?C.teal:C.navy),18); text(s,a[1],xs[i],344,150,44,15,C.muted,false,'center'); if(i<labels.length-1) connector(s,xs[i]+150,289,xs[i+1]-12,289,C.blue); });
 text(s,'研究同时处理两类问题：输入数据必须稳定，模型必须能够利用局部时空信息。',142,470,980,58,24,C.navy,true,'center');
 footer(s,3);
}

// 4 data and methods
{
 const s=deck.slides.add(); s.background.fill=C.bg; title(s,'数据与技术依据');
 const rows=[
 ['NTU RGB+D','动作视频、标签与标准划分','当前使用 NTU60 XSub'],
 ['COCO-WholeBody / RTMW','133 点全身关键点定义与姿态估计','从 RGB 视频重新提取骨架'],
 ['ST-GCN','空间图卷积与时间建模','手部和面部局部分支'],
 ['CTR-GCN','按通道细化骨架拓扑','身体主体分支'],
 ['自适应图与区域分析','学习动态关系并分析区域贡献','作为方法设计和解释依据']
 ];
 const x=[72,300,730], widths=[210,400,478];
 ['数据或方法','原有作用','本研究中的使用方式'].forEach((h,i)=>{box(s,x[i],176,widths[i],42,C.navy,'none','none'); text(s,h,x[i]+14,185,widths[i]-28,24,16,C.white,true);});
 rows.forEach((r,ri)=>{ const y=218+ri*76; r.forEach((v,i)=>{box(s,x[i],y,widths[i],76,ri%2?'F0F4F7':C.white,'D7E0E8','none'); text(s,v,x[i]+14,y+12,widths[i]-28,52,16,i===0?C.navy:C.text,i===0);});});
 text(s,'本研究组合已有技术，并针对 133 点全身骨架重新安排数据处理和分支结构。',72,602,1050,34,19,C.muted);
 footer(s,4);
}

// 5 extraction
{
 const s=deck.slides.add(); s.background.fill=C.bg; title(s,'骨架提取：RTMW-133 保留局部信息');
 box(s,72,180,460,380,C.white,'D7E0E8','rounded-xl'); text(s,'为什么重新提取',104,210,360,30,22,C.navy,true);
 bullet(s,'Kinect 少量节点难以表达手指、手腕和面部变化',104,268,370,18);
 bullet(s,'RGB 视频可重新生成统一的 133 点全身表示',104,348,370,18);
 bullet(s,'坐标、置信度、人物框和槽位共同保存',104,428,370,18);
 box(s,580,180,628,380,'EDF5F3','none','rounded-xl'); text(s,'每个样本的处理链路',616,210,400,30,22,C.teal,true);
 const ys=[274,340,406,472]; const ls=['人物检测','人物框裁剪与匹配','RTMW-L 输出 133 点','NPZ 骨架与质量元数据'];
 ls.forEach((v,i)=>{node(s,v,650,ys[i],380,48,i===2?C.blue:C.teal,17); if(i<3) connector(s,840,ys[i]+48,840,ys[i+1],C.muted);});
 footer(s,5);
}

// 6 quality
{
 const s=deck.slides.add(); s.background.fill=C.bg; title(s,'质量控制：先修复输入，再比较模型');
 box(s,72,176,490,404,C.white,'D7E0E8','rounded-xl'); text(s,'检查项目',104,208,300,30,22,C.navy,true);
 ['人物数量与槽位稳定性','身体节点完整性','NaN、低置信度与连续缺失','相邻帧关键点和人物框跳变'].forEach((v,i)=>bullet(s,v,104,270+i*60,400,18));
 box(s,610,176,598,404,'FFF5E9','none','rounded-xl'); text(s,'困难样本重新检测结果',648,208,430,30,22,C.orange,true);
 text(s,'同一批 100 个质量检查失败样本',648,254,420,26,16,C.muted);
 text(s,'5%',668,328,130,66,48,C.red,true,'center'); text(s,'85%',930,328,130,66,48,C.teal,true,'center');
 text(s,'RTMDet-tiny\n通过率',650,402,165,55,17,C.muted,false,'center'); text(s,'YOLO26-X\n通过率',912,402,165,55,17,C.muted,false,'center');
 connector(s,804,360,906,360,C.orange);
 text(s,'平均启发式质量分：80.521 → 96.149',680,492,470,34,21,C.navy,true,'center');
 footer(s,6);
}

// 7 model
{
 const s=deck.slides.add(); s.background.fill=C.bg; title(s,'模型结构：身体主体与局部区域分支建模');
 text(s,'RTMW-133',72,214,170,32,22,C.navy,true,'center'); box(s,72,260,170,120,C.navy,'none','rounded-xl'); text(s,'133 点\n全身骨架',88,282,138,72,24,C.white,true,'center');
 connector(s,242,320,330,250,C.blue); connector(s,242,320,330,450,C.blue);
 node(s,'身体 22 点',350,214,220,72,C.blue,20); text(s,'CTR-GCN 风格主体分支',350,300,220,32,16,C.muted,false,'center');
 node(s,'双手 42 点 + 面部 6 token',350,414,320,72,C.teal,18); text(s,'ST-GCN 风格局部分支',350,500,320,32,16,C.muted,false,'center');
 connector(s,570,250,760,360,C.blue); connector(s,670,450,760,380,C.teal);
 box(s,760,306,210,100,'EAF4F2','none','rounded-xl'); text(s,'有效节点池化\n与特征投影',776,326,178,62,20,C.navy,true,'center');
 connector(s,970,356,1060,356,C.blue);
 box(s,1060,306,150,100,C.orange,'none','rounded-xl'); text(s,'动作\n分类',1072,326,126,62,22,C.white,true,'center');
 footer(s,7);
}

// 8 adjustments
{
 const s=deck.slides.add(); s.background.fill=C.bg; title(s,'方法调整：为什么这样设计');
 const cards=[
 ['133 点输入','保留手部、面部和足部细节','应对少节点输入的局部信息缺失',C.blue],
 ['分支建模','身体、手部和面部分别处理','匹配不同区域的尺度、拓扑和噪声',C.teal],
 ['面部 6 token','压缩 68 个面部点','降低密集节点的成本和噪声',C.orange],
 ['有效性 Mask','只统计有效节点和人物','减少缺失点对特征池化的干扰',C.navy]
 ];
 cards.forEach((a,i)=>{ const x=72+(i%2)*570, y=186+Math.floor(i/2)*190; box(s,x,y,520,150,C.white,'D7E0E8','rounded-xl'); box(s,x,y,8,150,a[3],'none','none'); text(s,a[0],x+28,y+22,420,28,21,a[3],true); text(s,a[1],x+28,y+62,440,25,18,C.navy,true); text(s,a[2],x+28,y+98,450,32,16,C.muted); });
 footer(s,8);
}

// 9 results
{
 const s=deck.slides.add(); s.background.fill=C.bg; title(s,'当前基线结果：分类链路已经跑通');
 const metrics=[['89.60%','验证 Top-1',C.blue],['98.82%','验证 Top-5',C.teal],['99.84%','训练 Top-1',C.orange],['907,690','参数量',C.navy]];
 metrics.forEach((m,i)=>{const x=72+i*282; box(s,x,190,250,145,C.white,'D7E0E8','rounded-xl'); text(s,m[0],x+16,220,218,55,34,m[2],true,'center'); text(s,m[1],x+16,290,218,25,16,C.muted,false,'center');});
 box(s,72,390,1136,150,'EEF3F7','none','rounded-xl'); text(s,'实验设置',104,418,170,26,18,C.navy,true); text(s,'NTU60 XSub  ·  RTMW-133 raw x/y/score  ·  64 帧  ·  65 轮  ·  seed 1',104,462,1000,30,21,C.text);
 text(s,'训练与验证差距约 9.24 个百分点，说明模型能收敛，但仍存在明显过拟合。',104,594,1050,30,19,C.muted);
 footer(s,9);
}

// 10 next
{
 const s=deck.slides.add(); s.background.fill=C.bg; title(s,'下一步：验证局部信息是否真正带来提升');
 const steps=[['数据版本','manifest、清洗记录和骨架准确性检查'],['模型消融','body-only、local-only、去手部、去面部、去 CTR'],['输入与融合','raw、relative、运动通道、拼接与门控'],['解释与效率','区域/时间遮挡，多 seed、显存、速度和 FLOPs']];
 steps.forEach((a,i)=>{const y=184+i*96; pill(s,String(i+1).padStart(2,'0'),82,y,54,C.blue); text(s,a[0],164,y+2,200,28,21,C.navy,true); text(s,a[1],382,y+2,740,30,19,C.text); box(s,164,y+48,960,1,'D7E0E8','none','none');});
 box(s,72,594,1136,52,C.navy,'none','rounded-xl'); text(s,'研究目标：明确哪些区域、哪些时间阶段和哪些处理方式真正提升相似动作的识别能力。',96,605,1088,30,20,C.white,true,'center');
 footer(s,10);
}

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(draft);
console.log(JSON.stringify({ draft, finalPath, slides: deck.slides.items.length }));
