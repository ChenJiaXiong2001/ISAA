import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT = 'C:/Users/15637/PycharmProjects/ISAA';
const SKILL_DIR = 'C:/Users/15637/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const BUILD = path.join(ROOT, '.ppt_build_publication');
const FINAL_DIR = path.join(ROOT, 'outputs');
const DRAFT = path.join(BUILD, '原池化100帧与CTR-GCN发表汇报_草稿.pptx');
const FINAL = path.join(FINAL_DIR, '原池化100帧与CTR-GCN发表汇报.pptx');
const ASSETS = path.join(ROOT, 'docs/研究报告图表_原池化CTR对比');
const POOL_TOP1 = 94.6400;
const POOL_TOP5 = 99.5891;
const SINGLE_TOP1 = 95.4802;
const SINGLE_TOP5 = 99.6688;
const SINGLE_EPOCH = 23;
const POOL_PARAMS = '1,115,716';
const SINGLE_PARAMS = '1,447,550';

await fs.mkdir(BUILD, { recursive: true });
await fs.mkdir(FINAL_DIR, { recursive: true });

const W = 1280, H = 720;
const C = {
  bg: 'F7F8FA', navy: '102A43', blue: '1F6FEB', teal: '138A72', orange: 'C2410C',
  red: 'B42318', text: '243B53', muted: '627D98', line: 'CBD5E1', paleBlue: 'EAF2FF',
  paleTeal: 'EAF7F3', paleOrange: 'FFF1E8', white: 'FFFFFF', gray: 'EEF2F6', dark: '0F172A'
};
const FONT = 'Aptos';

const deck = Presentation.create({ slideSize: { width: W, height: H } });

async function imageBytes(p) {
  const bytes = await fs.readFile(p);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function shape(slide, geometry, x, y, w, h, fill = 'none', line = 'none', radius = 'rounded-xl') {
  const cfg = { geometry, position: { left: x, top: y, width: w, height: h }, fill, line: { style: 'solid', fill: line, width: line === 'none' ? 0 : 1 } };
  if (radius && radius !== 'none') cfg.borderRadius = radius;
  return slide.shapes.add(cfg);
}
function rect(slide, x, y, w, h, fill = 'none', line = 'none') { return shape(slide, 'rect', x, y, w, h, fill, line, 'none'); }
function round(slide, x, y, w, h, fill = 'none', line = 'none', radius = 'rounded-xl') { return shape(slide, 'roundRect', x, y, w, h, fill, line, radius); }
function text(slide, str, x, y, w, h, size = 20, color = C.text, bold = false, align = 'left') {
  const s = slide.shapes.add({ geometry: 'textbox', position: { left: x, top: y, width: w, height: h }, fill: 'none', line: { style: 'solid', fill: 'none', width: 0 } });
  s.text = str;
  s.text.style = { fontFamily: FONT, fontSize: size, color, bold, align, verticalAlign: 'middle' };
  return s;
}
function line(slide, x1, y1, x2, y2, color = C.line, width = 2, arrow = false) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = Math.max(2, Math.hypot(dx, dy));
  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  return slide.shapes.add({
    geometry: 'line',
    position: { left: cx - length / 2, top: cy - 1, width: length, height: 2, rotation: angle },
    line: { style: 'solid', fill: color, width, beginArrowType: 'none', endArrowType: arrow ? 'triangle' : 'none' }
  });
}
function title(slide, heading, section, page) {
  text(slide, section.toUpperCase(), 72, 30, 400, 22, 12, C.blue, true);
  text(slide, heading, 72, 57, 1100, 52, 31, C.navy, true);
  rect(slide, 72, 122, 1136, 2, C.blue);
  text(slide, `发表汇报  ·  ${page}`, 1080, 684, 128, 18, 11, C.muted, false, 'right');
}
function pill(slide, label, x, y, w, fill, color = C.white, size = 14) {
  round(slide, x, y, w, 32, fill, 'none', 'rounded-lg');
  text(slide, label, x + 8, y + 2, w - 16, 28, size, color, true, 'center');
}
async function addImage(slide, p, x, y, w, h, alt, fit = 'contain') {
  return slide.images.add({ blob: await imageBytes(p), contentType: 'image/png', alt, fit, position: { left: x, top: y, width: w, height: h } });
}
function bullet(slide, label, x, y, w, size = 19, color = C.text) {
  text(slide, '•', x, y, 18, 28, size + 3, C.blue, true);
  text(slide, label, x + 22, y, w - 22, 42, size, color, false);
}
function note(slide, source) { slide.speakerNotes.text = `证据来源：${source}`; slide.speakerNotes.setVisible(false); }

// 1. Cover
{
  const s = deck.slides.add(); s.background.fill = C.bg;
  rect(s, 0, 0, W, H, C.bg);
  rect(s, 72, 92, 10, 456, C.blue);
  text(s, '原池化100帧输入\n与 CTR-GCN 对比', 120, 128, 800, 150, 47, C.navy, true);
  text(s, '基于 RTMW-133 全身二维骨架的 NTU60 XSub 实验分析', 124, 316, 780, 40, 22, C.muted);
  pill(s, '图结构  ·  训练曲线  ·  类别分析  ·  受控实验边界', 124, 404, 510, C.teal, C.white, 15);
  text(s, '发表汇报版', 124, 612, 230, 32, 17, C.blue, true);
  text(s, '2026.10', 1040, 612, 110, 32, 17, C.muted, false, 'right');
  note(s, 'docs/原池化100帧与CTR-GCN对比研究报告.md');
}

// 2. One-slide answer
{
  const s = deck.slides.add(); s.background.fill = C.bg; title(s, '核心结论', '结论先行', 2);
  round(s, 72, 178, 530, 370, C.paleBlue, 'none');
  text(s, '原池化100帧', 106, 214, 450, 38, 25, C.blue, true);
  text(s, '94.6400%', 106, 266, 450, 78, 57, C.navy, true);
  text(s, '统一评估 Top-1  ·  日志最佳第98轮 94.6461%', 108, 360, 490, 30, 17, C.muted);
  text(s, '1,115,716 参数', 108, 416, 430, 30, 18, C.text, true);
  round(s, 654, 178, 554, 370, C.paleOrange, 'none');
  text(s, '论文版单流 Joint CTR-GCN', 690, 214, 470, 38, 25, C.orange, true);
  text(s, '95.4802%', 690, 266, 450, 78, 57, C.navy, true);
  text(s, '验证 Top-1  ·  最佳第23逻辑 epoch', 692, 360, 450, 30, 18, C.muted);
  text(s, `${SINGLE_PARAMS} 参数`, 692, 416, 430, 30, 18, C.text, true);
  text(s, '单流验证高出0.8402个百分点。两组都使用100帧，但训练遍历数、batch size、图映射和读出协议不同；10-clip独立测试仍待完成。', 92, 594, 1110, 44, 18, C.muted, false, 'center');
  note(s, 'outputs/evaluations/pool_weighted_100_cosine_20261016/summary.json; outputs/evaluations/single_stream_ctr_gcn_20261019/epochs.csv');
}

// 3. Research question and data
{
  const s = deck.slides.add(); s.background.fill = C.bg; title(s, '研究问题与输入', '研究对象', 3);
  round(s, 72, 170, 510, 410, C.white, C.line);
  text(s, '整体姿态相似时，局部细节决定分类', 106, 202, 430, 40, 24, C.navy, true);
  bullet(s, '少节点骨架难以表达手腕、手指和面部变化', 106, 278, 420, 18);
  bullet(s, 'RTMW-133 保留全身局部信息，但噪声和缺失更明显', 106, 352, 420, 18);
  bullet(s, '模型需要同时处理主体运动与局部时空变化', 106, 426, 420, 18);
  round(s, 632, 170, 576, 410, C.paleTeal, 'none');
  text(s, 'NTU RGB+D 60 · XSub', 668, 202, 420, 32, 24, C.teal, true);
  text(s, '39,627', 668, 274, 180, 54, 38, C.navy, true);
  text(s, '训练样本', 668, 330, 180, 25, 16, C.muted);
  text(s, '16,306', 904, 274, 180, 54, 38, C.navy, true);
  text(s, '验证样本', 904, 330, 180, 25, 16, C.muted);
  pill(s, 'RTMW-133', 668, 408, 140, C.blue); pill(s, '最多2人', 824, 408, 110, C.navy); pill(s, '60类', 950, 408, 88, C.orange);
  text(s, '输入基础通道：raw x / y / score + 有效性掩码', 668, 490, 480, 30, 18, C.text, true);
  note(s, 'outputs/evaluations/pool_weighted_100_cosine_20261016/config.json');
}

// 4. Pipeline
{
  const s = deck.slides.add(); s.background.fill = C.bg; title(s, '研究流程', '数据到模型', 4);
  const labels = [
    ['RGB视频', 'NTU RGB+D'], ['人物检测', '定位目标人物'], ['RTMW-133', '133点全身骨架'],
    ['质量控制', '缺失、跳变、槽位'], ['分支建模', '身体 / 双手 / 面部'], ['动作分类', '60类 logits']
  ];
  const xs = [72, 258, 444, 630, 816, 1002];
  labels.forEach((a, i) => {
    round(s, xs[i], 244, 150, 82, i === 2 ? C.blue : (i === 4 ? C.teal : C.navy), 'none', 'rounded-lg');
    text(s, a[0], xs[i] + 10, 257, 130, 30, 17, C.white, true, 'center');
    text(s, a[1], xs[i], 356, 150, 46, 15, C.muted, false, 'center');
    if (i < labels.length - 1) line(s, xs[i] + 150, 285, xs[i + 1] - 12, 285, C.blue, 2, true);
  });
  text(s, '输入稳定性与局部建模必须同时成立，才能解释后续精度差异。', 164, 506, 950, 42, 25, C.navy, true, 'center');
  note(s, 'docs/原池化100帧与CTR-GCN对比研究报告.docx');
}

// 5. Body graph with every selected RTMW node labelled
{
  const s = deck.slides.add(); s.background.fill = C.bg; title(s, '图结构：身体22个节点全部标注', '模型基础', 5);
  text(s, 'RTMW索引 + 语义名称', 84, 166, 430, 26, 18, C.blue, true);
  const nodes = {
    n0:[300,215,'0 鼻'], n5:[300,275,'5 左肩'], n6:[470,275,'6 右肩'], n7:[250,350,'7 左肘'], n8:[520,350,'8 右肘'], n9:[205,430,'9 左腕'], n10:[565,430,'10 右腕'],
    n11:[350,350,'11 左髋'], n12:[430,350,'12 右髋'], n13:[320,440,'13 左膝'], n14:[460,440,'14 右膝'], n15:[285,540,'15 左踝'], n16:[495,540,'16 右踝'],
    n17:[245,620,'17 左脚尖1'], n18:[285,630,'18 左脚尖2'], n19:[325,620,'19 左脚跟'], n20:[455,620,'20 右脚尖1'], n21:[495,630,'21 右脚尖2'], n22:[535,620,'22 右脚跟'], n56:[385,235,'56 面部代表点'], n91:[205,480,'91 左手根/腕'], n112:[565,480,'112 右手根/腕']
  };
  const edges = [['n0','n5'],['n0','n6'],['n5','n7'],['n7','n9'],['n9','n91'],['n6','n8'],['n8','n10'],['n10','n112'],['n5','n11'],['n6','n12'],['n11','n12'],['n11','n13'],['n13','n15'],['n15','n17'],['n15','n18'],['n15','n19'],['n12','n14'],['n14','n16'],['n16','n20'],['n16','n21'],['n16','n22'],['n0','n56']];
  edges.forEach(([a,b]) => line(s, nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1], C.line, 3));
  Object.entries(nodes).forEach(([k,[x,y,label]]) => { const fill = label.includes('手根') ? C.orange : (label.startsWith('0 ') || label.startsWith('56 ') ? C.blue : C.navy); round(s, x - 15, y - 15, 30, 30, fill, 'none', 'rounded-full'); text(s, label, x + 18, y - 13, label.length > 7 ? 120 : 88, 26, 12, C.text, true); });
  round(s, 760, 176, 430, 404, C.white, C.line);
  text(s, '节点分区', 792, 204, 180, 28, 22, C.navy, true);
  const legend = ['0：鼻；5–10：肩、肘、腕','11–16：髋、膝、踝','17–22：足部6点','56：面部代表点','91 / 112：左右手根/腕','左右手内部另用21点拓扑'];
  legend.forEach((v,i)=>bullet(s,v,792,260+i*46,350,16));
  text(s, '连线只表示项目图中的显式身体边；手指内部拓扑在下一页展开。', 792, 530, 350, 42, 16, C.muted, false);
  text(s, '节点标签采用 RTMW 物理索引，避免只画抽象圆点。', 170, 674, 940, 26, 19, C.navy, true, 'center');
  note(s, 'isaa/models/backbones/ctrgcn.py; isaa/layouts/rtmw_133.py');
}

// 6. Both hand graphs with all 21 local nodes
{
  const s = deck.slides.add(); s.background.fill = C.bg; title(s, '图结构：左右手21点拓扑与节点编号', '模型基础', 6);
  text(s, '左手 L0–L20（RTMW 91–111）', 88, 166, 430, 28, 21, C.teal, true);
  text(s, '右手 R0–R20（RTMW 112–132）', 752, 166, 440, 28, 21, C.teal, true);
  function handGraph(baseX, baseY, prefix, rtBase) {
    const scale = 1.65;
    const raw = [[0,0],[0,-46],[0,-92],[0,-138],[0,-184],[52,-32],[66,-76],[72,-120],[72,-164],[92,0],[114,-38],[124,-80],[124,-122],[78,32],[108,24],[120,0],[120,-24],[44,52],[68,54],[82,40],[86,16]];
    const pts = raw.map(([x,y]) => [x * scale, y * scale]);
    const handEdges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]];
    handEdges.forEach(([a,b])=>line(s, baseX+pts[a][0], baseY+pts[a][1], baseX+pts[b][0], baseY+pts[b][1], C.line, 3));
    pts.forEach(([dx,dy],i)=>{ const x=baseX+dx,y=baseY+dy; round(s,x-13,y-13,26,26,i===0?C.blue:C.teal,'none','rounded-full'); text(s, `${prefix}${i}`, x-22, y+17, 44, 20, 11, C.text, true, 'center'); });
  }
  handGraph(150, 500, 'L', 91); handGraph(800, 500, 'R', 112);
  text(s, 'RTMW: 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111', 92, 570, 500, 30, 11, C.muted, false, 'center');
  text(s, 'RTMW: 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132', 742, 570, 500, 30, 11, C.muted, false, 'center');
  round(s, 440, 610, 400, 46, C.paleTeal, 'none', 'rounded-lg'); text(s, '共享 CTR-GCN 权重 · 20条显式手指边/只手', 456, 620, 368, 22, 16, C.teal, true, 'center');
  text(s, '节点标签用 L0–L20 / R0–R20，底部给出对应 RTMW 全身索引。', 242, 674, 796, 26, 18, C.navy, true, 'center');
  note(s, 'isaa/layouts/rtmw_133.py::_hand_edges()');
}

// 7. Model information flow image
{
  const s = deck.slides.add(); s.background.fill = C.bg; title(s, '原池化100帧模型', '三分支信息流', 7);
  await addImage(s, path.join(ASSETS, '原池化模型结构示意.png'), 72, 162, 1136, 490, '原池化100帧三分支模型结构示意', 'contain');
  text(s, '100帧是输入时间窗口，池化发生在骨干输出后。', 72, 662, 740, 26, 17, C.muted);
  note(s, 'docs/研究报告图表_原池化CTR对比/原池化模型结构示意.png');
}

// 8. Single-stream CTR-GCN comparison diagram
{
  const s = deck.slides.add(); s.background.fill = C.bg; title(s, '论文版单流 Joint CTR-GCN', '结构对照', 8);
  text(s, '原池化100帧', 116, 178, 260, 28, 22, C.blue, true, 'center');
  text(s, '单流 Joint CTR-GCN', 820, 178, 260, 28, 22, C.orange, true, 'center');
  const left = [['身体/双手/面部', '分支化 + 掩码池化'], ['输入时间', '100帧置信度压缩'], ['读出', '192+96 → 288维']];
  const right = [['Joint stream', 'x/y/score'], ['图映射', '65 → 27 → 11'], ['骨干', 'unit_ctrgcn + mstcn']];
  for (let i = 0; i < 3; i++) {
    const y = 246 + i * 110;
    round(s, 84, y, 330, 76, C.paleBlue, C.line, 'rounded-lg'); text(s, left[i][0], 104, y + 10, 290, 26, 19, C.navy, true); text(s, left[i][1], 104, y + 42, 290, 24, 16, C.muted);
    round(s, 866, y, 330, 76, C.paleOrange, C.line, 'rounded-lg'); text(s, right[i][0], 886, y + 10, 290, 26, 19, C.navy, true); text(s, right[i][1], 886, y + 42, 290, 24, 16, C.muted);
  }
  line(s, 470, 408, 810, 408, C.orange, 3, true); pill(s, '容量与时间分辨率同时变化', 520, 380, 240, C.orange, C.white, 15);
  text(s, '单流模型约1,447,550参数，验证 Top-1 95.4802%。', 214, 612, 850, 34, 21, C.navy, true, 'center');
  note(s, 'outputs/evaluations/single_stream_ctr_gcn_20261019/config.json; outputs/evaluations/pool_weighted_100_cosine_20261016/config.json');
}

// 9. Protocol fairness
{
  const s = deck.slides.add(); s.background.fill = C.bg; title(s, '实验协议：差异必须写清楚', '可比性边界', 8);
  const rows = [
    ['输入时间', '100帧置信度压缩', '100帧 Joint stream'],
    ['图结构', '分支化区域图', '65 → 27 → 11 progressive'],
    ['训练计划', '100轮', '24逻辑epoch × repeat 5'],
    ['batch size', '32', '128训练 / 32验证'],
    ['编译训练', 'torch.compile required', 'Inductor fullgraph required']
  ];
  const x = [72, 350, 780], widths = [240, 390, 428];
  ['比较项', '原池化100帧', '单流 Joint CTR-GCN'].forEach((h, i) => { rect(s, x[i], 178, widths[i], 42, C.navy); text(s, h, x[i] + 12, 187, widths[i] - 24, 24, 16, C.white, true); });
  rows.forEach((r, ri) => { const y = 220 + ri * 68; r.forEach((v, i) => { rect(s, x[i], y, widths[i], 68, ri % 2 ? C.gray : C.white, C.line); text(s, v, x[i] + 12, y + 10, widths[i] - 24, 48, 16, i === 0 ? C.navy : C.text, i === 0); }); });
  round(s, 72, 580, 1136, 56, C.paleOrange, 'none', 'rounded-lg'); text(s, '当前差异是协议级结果，不能直接归因于池化或 CTR-GCN 的单一因素。', 92, 592, 1096, 30, 19, C.orange, true, 'center');
  note(s, 'outputs/evaluations/pool_weighted_100_cosine_20261016/config.json; outputs/evaluations/single_stream_ctr_gcn_20261019/config.json');
}

// 10. Performance chart
{
  const s = deck.slides.add(); s.background.fill = C.bg; title(s, '总体性能：单流 Joint CTR-GCN 验证领先', '结果', 10);
  await addImage(s, path.join(ASSETS, '性能对比.png'), 72, 164, 720, 390, '原池化100帧与单流 Joint CTR-GCN Top-1性能对比图', 'contain');
  round(s, 842, 180, 366, 330, C.white, C.line);
  text(s, '关键指标', 874, 208, 260, 30, 21, C.navy, true);
  const metrics = [['Top-1', '94.6400%', '95.4802%'], ['Top-5', '99.5891%', '99.6688%'], ['参数量', '1,115,716', '1,447,550'], ['独立测试', '已完成', '运行中']];
  metrics.forEach((m, i) => { const y = 260 + i * 54; text(s, m[0], 874, y, 96, 24, 16, C.muted); text(s, m[1], 974, y, 106, 24, 17, C.blue, true, 'right'); text(s, m[2], 1088, y, 110, 24, 17, C.orange, true, 'right'); });
  text(s, '蓝：原池化100帧  ·  橙：单流 Joint CTR-GCN', 856, 532, 330, 24, 14, C.muted, false, 'center');
  text(s, '单流数值是训练验证最佳值，10-clip测试结果待补充。', 88, 596, 1080, 30, 18, C.muted, false, 'center');
  note(s, 'outputs/evaluations/pool_weighted_100_cosine_20261016/summary.json; outputs/evaluations/single_stream_ctr_gcn_20261019/epochs.csv; docs/研究报告图表_原池化CTR对比/性能对比.png');
}

// 11. Training curves
{
  const s = deck.slides.add(); s.background.fill = C.bg; title(s, '训练曲线：单流模型的验证优势', '结果', 11);
  await addImage(s, path.join(ASSETS, '池化100帧训练与逐类变化.png'), 72, 168, 680, 430, '原池化100帧训练与验证曲线', 'contain');
  await addImage(s, path.join(ASSETS, '单流CTR-GCN训练曲线.png'), 780, 168, 428, 430, '单流 Joint CTR-GCN训练与验证曲线', 'contain');
  round(s, 72, 610, 1136, 46, C.paleBlue, 'none', 'rounded-lg'); text(s, '原池化：统一评估94.6400%  ·  单流 Joint CTR-GCN：验证95.4802%  ·  两者训练计划不完全相同', 94, 620, 1092, 26, 16, C.navy, true, 'center');
  note(s, 'outputs/evaluations/pool_weighted_100_cosine_20261016/training_analysis.png; outputs/evaluations/single_stream_ctr_gcn_20261019/epochs.csv');
}

// 12. Per-class and interpretation
{
  const s = deck.slides.add(); s.background.fill = C.bg; title(s, '逐类结果：局部动作仍是主要难点', '结果解释', 12);
  await addImage(s, path.join(ASSETS, '池化100帧逐类准确率.png'), 72, 170, 700, 430, '原池化100帧逐类准确率图', 'contain');
  round(s, 828, 182, 380, 350, C.white, C.line);
  text(s, '观察', 858, 210, 160, 28, 22, C.navy, true);
  bullet(s, '原池化弱项：阅读、玩手机或平板、打喷嚏或咳嗽', 858, 266, 310, 16);
  bullet(s, '原池化强项：跳起、摇头、跌倒、握手、一起走近', 858, 354, 310, 16);
  bullet(s, '单流 CTR-GCN 的逐类测试差异待10-clip评估完成', 858, 442, 310, 16);
  text(s, '时间覆盖与平均压缩可能同时影响类别表现。', 100, 620, 1080, 26, 18, C.muted, false, 'center');
  note(s, 'outputs/evaluations/pool_weighted_100_cosine_20261016/summary.json; outputs/evaluations/single_stream_ctr_gcn_20261019/epochs.csv');
}

// 13. Limits and next experiment
{
  const s = deck.slides.add(); s.background.fill = C.bg; title(s, '研究边界与下一步', '结论与计划', 13);
  round(s, 72, 176, 520, 388, C.white, C.line);
  text(s, '当前结论', 106, 208, 250, 30, 23, C.blue, true);
  bullet(s, '原池化100帧是当前已完成记录中的准确率候选基线', 106, 274, 430, 17);
  bullet(s, '单流 Joint CTR-GCN 当前验证 Top-1 为95.4802%', 106, 360, 430, 17);
  bullet(s, '10-clip独立测试仍在服务器运行', 106, 446, 430, 17);
  round(s, 646, 176, 562, 388, C.paleTeal, 'none');
  text(s, '后续实验', 680, 208, 250, 30, 23, C.teal, true);
  const next = ['完成论文版单流10-clip测试评估', '统一100帧与训练遍历预算', '多随机种子报告均值与标准差', '身体/双手/面部与时间遮挡实验'];
  next.forEach((v, i) => { pill(s, String(i + 1).padStart(2, '0'), 680, 274 + i * 58, 60, C.teal, C.white, 14); text(s, v, 758, 276 + i * 58, 388, 28, 17, C.navy, true); });
  text(s, '完成这些对照后，才能把“结构效果”和“训练预算效果”分开。', 148, 626, 984, 28, 20, C.navy, true, 'center');
  note(s, 'docs/原池化100帧与CTR-GCN对比研究报告.md');
}

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(DRAFT);
console.log(JSON.stringify({ draft: DRAFT, final: FINAL, slides: deck.slides.items.length, skillDir: SKILL_DIR }));
