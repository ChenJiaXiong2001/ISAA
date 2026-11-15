"""Generate the evidence appendices and new Word report without editing old reports."""
from pathlib import Path
import csv, json, re, hashlib
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
MD=ROOT/'docs/ISAA完整研究报告_20261011.md'
OUT=MD.with_suffix('.docx')
BUILD=ROOT/'docs/report_assets_20261011'
BUILD.mkdir(exist_ok=True)
PROX='outputs/evaluations/best_hand_body_proximity_20261011'
FULL='outputs/evaluations/full_gcn_20261010'
def load(p): return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def rows(p):
    with (ROOT/p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
pr=rows(PROX+'/per_class_comparison.csv')
fr=rows(FULL+'/per_class_comparison.csv')
rr=rows('outputs/evaluations/baseline_vs_routed/per_class_comparison.csv')
assert len(pr)==len(fr)==len(rr)==60
fm={r['action_id']:r for r in fr};rm={r['action_id']:r for r in rr}
assert sum(int(r['samples']) for r in pr)==16306
assert sum(int(r['baseline_correct']) for r in pr)==15180
assert sum(int(r['new_correct']) for r in pr)==15172
assert sum(int(r['full_correct']) for r in fr)==15067
assert sum(int(r['baseline_correct']) for r in fr)==15179
assert sum(int(r['routed_correct']) for r in rr)==15049
for r in pr:
    a=r['action_id']
    assert r['samples']==fm[a]['samples']==rm[a]['samples']
    assert abs(int(r['new_correct'])/int(r['samples'])*100-float(r['new_top1_percent']))<1e-9
text=MD.read_text(encoding='utf-8').split('## 附录 A')[0].rstrip()
append='''

## 附录 A 全部 60 类验证结果

单位为百分数，样本数对应实际保留的验证集。B 为原最佳统一复测，R 为软门控，F 为完整 GCN，P 为最佳结构加 proximity。B 与 R、P 配对；F 的原最佳配对基线在下述差异类少一个正确，其余类别相同，总计为 15,179。此一个样本差异不作为结构效应。名称沿用项目评估文件，逐类数字按真实类别计分，即该类召回率。[E2] 至 [E5]

| 动作 | 名称 | 样本 | B | R | F | P |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
'''
diffs=[]
for r in pr:
    a=r['action_id'];f=fm[a];route=rm[a]
    if r['baseline_correct']!=f['baseline_correct']:diffs.append(a)
    vals=[r['baseline_top1_percent'],route['routed_accuracy_percent'],f['full_top1_percent'],r['new_top1_percent']]
    append+=f"| {a} | {r['name']} | {r['samples']} | "+' | '.join(f'{float(v):.2f}' for v in vals)+' |\n'
append+='\n完整 GCN 配对基线与 B 的差异类：'+', '.join(diffs)+'。总体 Top-1 用正确数总和除以 16,306，而非逐类百分数简单平均。B、R、F、P 总正确数依次为 15,180、15,049、15,067、15,172，60 类样本数和正确数总和已核对。\n'
append+='''

## 附录 B 索引和术语速查

### B.1 全身节点索引

0 鼻；1、2 左右眼；3、4 左右耳；5、6 左右肩；7、8 左右肘；9、10 左右腕；11、12 左右髋；13、14 左右膝；15、16 左右踝。17 至 22 为 6 个足部点，23 至 90 为 68 个面部点，91 至 111 为左手，112 至 132 为右手。索引为零基，不能将 NTU 官方 25 点编号直接套用。

每只手局部 0 为腕点；1 至 4 拇指，5 至 8 食指，9 至 12 中指，13 至 16 无名指，17 至 20 小指。每链沿腕到指尖排列，相邻链点相连，腕连接各手指链根。左手局部 0 全局为 91，右手为 112；身体腕 9/10 与手部腕是不同索引表达，即使接近也不能假定数值完全一致。

历史 32 区域将头部 0 至 4 聚合，身体其他代表与足部各成区域，面部 23 至 90 为一组；每只手非指尖为一组，五个指尖各成一组。历史 25 节点语义映射中存在重复 RTMW 索引，因为若干 NTU 语义关节共享观测点。这是映射约定，不表示多了真实观测，也不能在当前 22 节点模型中擅自复制。

### B.2 核心术语

| 术语 | 本项目中的含义 |
| --- | --- |
| skeleton 骨架 | 时序二维关键点与置信度，不是人体网格 |
| raw | 原始 x、y、score，不自动统一尺度 |
| torso | 肩髋参考集合；body 分支范围更广 |
| relative | 最佳减中心；交互另作尺度归一化 |
| mask | 有效观测布尔量，与学习门值不同 |
| topology | 信息传播关系矩阵，可包含学习关系 |
| token | 面部固定点组表示，不是器官标签 |
| gate | 特征系数；在骨干后则不节省骨干计算 |
| route | 决定执行路径，硬路由在手骨干前分样本 |
| supernet | 多种宽度共用一组大网络权重 |
| calibration | 估计 BN 或阈值等部署状态，须隔离官方验证 |
| distillation | 用教师分布训练学生，不保证同等准确率 |
| ablation | 应标明推理移除或独立训练 |
| registered parameters | 注册参数量，未必全在当前前向使用 |
| MACs | 乘加数；参数、宽度代理、延迟都不是 MACs |
| paired comparison | 同一视频两模型结果逐条对应 |

## 附录 C 配置数值与损失权重

完整 GCN 三分支默认通道均为 64×4、128×3、256×3，步长均为 1、1、1、1、2、1、1、2、1、1；body 和 hand 为 CTR，face 为 ST，face dropout=0。proximity 的 width=64、radius_on=0.40、radius_off=0.48、confidence_threshold=0。共同训练参数见第 10 章。

| 自适应宽度配置 | 数值 |
| --- | ---: |
| stage_maxima / group_size | 64、128、256 / 8 |
| router_hidden | 128 |
| confidence / fallback threshold | 0.80 / 0.45 |
| topk | 3 |
| supernet / gate / route epochs | 40 / 10 / 15 |
| supernet / gate / route lr | 0.10 / 0.01 / 0.01 |
| distill_weight / temperature | 0.5 / 2.0 |
| width_penalty / search_cost_weight | 0.01 / 0.05 |
| search_steps | 12 |
| search positives / negatives | 64 / 64 |
| shared_samples_per_class | 16 |
| search_accuracy_tolerance | 0.02 |
| search_loss_tolerance | 0.03 |
| routing_accuracy_tolerance | 0.002 |
| min_class_samples / bn_calibration_batches | 5 / 4 |
| calibration_fraction | 0.20 |

accuracy_tolerance 按小数表示，0.02 为 2 个百分点，0.002 为 0.2 个百分点；loss_tolerance 为损失数值容差。width_penalty 与 search_cost_weight 分别用于软宽度学习和搜索成本偏好，不能合成同一个速度系数。这些为默认值，不是已证明最优的参数。

硬手路由的 body_loss_weight 与 no_hand_loss_weight 均为 1，routing_distill_weight=0.5，distill_temperature=2；辅助头示例为 10 轮、lr0.01、warmup2、零基衰减7和9。需求 margin0、temperature0.1、调用阈值0.5和置信度回退0.8分别定义收益映射与部署决策，不能与训练蒸馏温度混用。

## 附录 D 证据清单和权重身份

以下为相对于 ISAA 根目录的路径，E6 指相邻提取项目。JSON 与 CSV 为实验事实依据，方法对应源码；历史 Markdown 或 README 中的旧状态不覆盖新的完成记录。

'''
evidence=[
('E1','历史 BodyLocal','backup_train_baseline/config.json；status.json；epochs.csv'),
('E2','原最佳模型与训练','outputs/evaluations/baseline_best_epoch61/summary.json；'+PROX+'/baseline_config.json；baseline_status.json；baseline_epochs.csv'),
('E3','软门控与配对','outputs/evaluations/routed_best_epoch64/summary.json；outputs/evaluations/baseline_vs_routed/comparison_summary.json；per_class_comparison.csv'),
('E4','完整 GCN',FULL+'/config.json；status.json；epochs.csv；summary.json；comparison_summary.json；run_evidence.json；per_class_comparison.csv'),
('E5','新 proximity 与分支移除',PROX+'/config.json；status.json；epochs.csv；summary.json；comparison_summary.json；branch_ablation.json；run_evidence.json；per_class_comparison.csv'),
('E6','提取与质量','../nturgbd_skeleton_extraction/PUBLICATION_SKELETON_EXTRACTION_RECORD.md'),
('E7','最佳与历史结构','isaa/models/body_local_fusion.py；rtmw_local_ctr.py；isaa/layouts/rtmw_133.py；isaa/graph/adjacency.py'),
('E8','完整骨干','isaa/models/body_local_full_gcn.py；isaa/models/backbones/ctrgcn.py；stgcn.py；isaa/models/original_ctrgcn.py；official_stgcn.py'),
('E9','交互定义','isaa/models/best_hand_body_proximity.py；hand_body_proximity.py；configs/hand_body_proximity.json'),
('E10','硬手路由','docs/class_hand_routing.md；isaa/models/class_hand_routed.py；isaa/class_hand_training.py'),
('E11','自适应宽度','docs/class_adaptive_gcn.md；configs/class_adaptive_gcn.json；isaa/models/class_adaptive_gcn.py；isaa/adaptive_width_training.py；outputs/adaptive_width_smoke/20261011-041656-b0328bef/status.json'),
('E12','加载训练评估','isaa/data/rtmw_zip_dataset.py；isaa/data/transforms.py；isaa/train.py；isaa/utils/experiment.py；tools/evaluate_best.py')]
for eid,name,path in evidence:append+=f'### {eid} {name}\n\n{path}\n\n'
append+='### D.1 Checkpoint SHA256\n\n'
for label,path in [('原最佳','outputs/evaluations/baseline_best_epoch61/summary.json'),('软门控','outputs/evaluations/routed_best_epoch64/summary.json'),('完整 GCN',FULL+'/summary.json'),('新 proximity',PROX+'/summary.json')]:
    s=load(path);append+=f"{label} checkpoint：{s['checkpoint']}\n\nSHA256：{s['checkpoint_sha256']}\n\n"
append+='### D.2 本地关键证据哈希\n\n这些哈希用于识别本报告的证据版本，不代替完整数据 ZIP 哈希。\n\n'
manifest={}
for rel in [PROX+'/summary.json',PROX+'/branch_ablation.json',PROX+'/comparison_summary.json',PROX+'/per_class_comparison.csv',FULL+'/summary.json',FULL+'/comparison_summary.json',FULL+'/per_class_comparison.csv','outputs/evaluations/baseline_vs_routed/per_class_comparison.csv']:
    digest=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest();manifest[rel]=digest
    append+=rel+'\n\nSHA256：'+digest+'\n\n'
text+=append
hist=rows('backup_train_baseline/epochs.csv')
assert round(float(max(hist,key=lambda r:float(r['val_top1']))['val_top1'])*16306)==14610
(BUILD/'evidence_checks.json').write_text(json.dumps({'samples':16306,'correct':[15180,15049,15067,15172],'full_paired_baseline':15179,'difference_classes':diffs,'evidence_sha256':manifest},ensure_ascii=False,indent=2),encoding='utf-8')

canvas=Image.new('RGB',(1800,890),'white');d=ImageDraw.Draw(canvas)
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',29)
small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
bold=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',34)
d.text((45,18),'最佳 wide-relative 模型的信息流',font=bold,fill='black')
def box(rect,ls):
    d.rounded_rectangle(rect,12,fill='#F3F5F7',outline='#7D8790',width=2)
    x,y,_,_=rect
    for j,line in enumerate(ls):d.text((x+18,y+16+j*43),line,font=font,fill='black')
def arrow(a,b):
    d.line([a,b],fill='#53616E',width=4);x,y=b
    d.polygon([(x,y),(x-13,y-8),(x-13,y+8)],fill='#53616E')
box((40,115,410,350),['全身 133 点','raw x、y、score','B×3×64×133×2','清洗与有效性掩码'])
box((510,90,1090,265),['身体 22 点 · 原始三通道','10 层 CTR · 48 / 96 / 192','时间 64 → 32 → 16'])
box((510,300,1090,500),['左右手各 21 点 · 共享权重','相对 xy、score、跨手距离/方向','6 层 CTR · 32 / 64 / 96','时间保持 64 · 投影到 48'])
box((510,545,1090,720),['面部 68 点 → 6 个固定 token','相对 xy 与 score','4 层 LocalST · 24 / 48'])
box((1210,110,1760,290),['逐帧掩码节点池化','身体时间保持 16','固定人物/时间平均 → 192'])
box((1210,390,1760,660),['左右手固定平均','手和脸线性对齐至 16 帧','手与脸固定平均','固定人物/时间平均 → 96','192 + 96 → 288 → 60 类'])
for a,b in [((410,175),(510,175)),((410,220),(510,400)),((410,265),(510,630)),((1090,175),(1210,175)),((1090,400),(1210,450)),((1090,630),(1210,590))]:arrow(a,b)
d.text((45,790),'注意：节点池化有掩码；最终人物和时间平均不按有效人数与帧数重新归一化。',font=small,fill='black')
ARCH=BUILD/'best_architecture.png';canvas.save(ARCH)
figures={
'### 6.1 分支选择与网络尺寸':(ARCH,'图 1 最佳模型输入 分支和融合示意'),
'### 11.5 训练曲线与完整 GCN 异常':(ROOT/FULL/'training_curves.png','图 2 完整 GCN 与原最佳模型的训练记录'),
'### 12.3 新 proximity 的类间重新分配':(ROOT/PROX/'class_changes.png','图 3 新 proximity 逐类提升与下降'),
'### 13.1 推理移除的实验定义':(ROOT/PROX/'training_curves.png','图 4 新 proximity 与原最佳模型的训练记录')}
text=re.sub(r'^!\[图 [1-4] .*?\]\(.*?\)\s*$', '', text, flags=re.M)
for heading,(path,caption) in figures.items():
    relative=Path('..')/path.relative_to(ROOT) if path.parent!=BUILD else Path(BUILD.name)/path.name
    text=text.replace(heading,heading+'\n\n!['+caption+']('+relative.as_posix()+')',1)
MD.write_text(text,encoding='utf-8')

doc=Document();sec=doc.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=Inches(.72);sec.bottom_margin=Inches(.72)
sec.left_margin=Inches(.78);sec.right_margin=Inches(.78)
sec.footer_distance=Inches(.28)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
    st=doc.styles[name];st.font.name='Microsoft YaHei';st.font.color.rgb=RGBColor(0,0,0)
    rf=st.element.get_or_add_rPr().find(qn('w:rFonts'))
    if rf is not None:rf.set(qn('w:eastAsia'),'Microsoft YaHei')
normal=doc.styles['Normal'];normal.font.size=Pt(11)
normal.paragraph_format.line_spacing=1.16;normal.paragraph_format.space_after=Pt(6)
for name,size in [('Title',25),('Heading 1',17),('Heading 2',13),('Heading 3',11.5)]:
    st=doc.styles[name];st.font.size=Pt(size);st.font.bold=True
    st.paragraph_format.space_before=Pt(12);st.paragraph_format.space_after=Pt(7)
    st.paragraph_format.keep_with_next=True
doc.styles['Caption'].font.size=Pt(9)
foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
doc.core_properties.title='ISAA 全身骨架动作识别研究报告'
doc.core_properties.subject='数据 方法 定义 实验与复现';doc.core_properties.author='ISAA'
for border in list(doc.styles.element.iter(qn('w:pBdr'))):
    border.getparent().remove(border)
def runs(p,s):
    for part in re.split(r'(\*\*.*?\*\*)',s):
        r=p.add_run(part[2:-2] if part.startswith('**') else part);r.bold=part.startswith('**')
def table(data):
    t=doc.add_table(rows=1,cols=len(data[0]));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    n=len(data[0])
    widths={7:[.55,1.75,.55,1.02,1.02,1.02,1.03],6:[1.93,.63,1.05,1.05,1.18,1.10],5:[1.75,1.12,1.12,1.53,1.42],4:[1.55,1.78,1.78,1.83],3:[1.6,2.8,2.54],2:[2.1,4.84]}[n]
    for col,w in zip(t.columns,widths):col.width=Inches(w)
    for i,row in enumerate(data):
        cells=t.rows[0].cells if i==0 else t.add_row().cells
        trpr=t.rows[i]._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
        if i==0:trpr.append(OxmlElement('w:tblHeader'))
        for j,(c,s) in enumerate(zip(cells,row)):
            c.width=Inches(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr=c._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
            for side in ['top','left','bottom','right']:
                edge=OxmlElement('w:'+side)
                for k,v in [('val','single'),('sz','4'),('color','D9D9D9')]:edge.set(qn('w:'+k),v)
                borders.append(edge)
            tcpr.append(borders);margins=OxmlElement('w:tcMar')
            for side,val in [('top','65'),('bottom','65'),('left','80'),('right','80')]:
                m=OxmlElement('w:'+side);m.set(qn('w:w'),val);m.set(qn('w:type'),'dxa');margins.append(m)
            tcpr.append(margins)
            if i==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E8EDF2');tcpr.append(shade)
            p=c.paragraphs[0];p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.08
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 or (n==7 and j==1) else WD_ALIGN_PARAGRAPH.CENTER
            r=p.add_run(s);r.font.size=Pt(9.5);r.bold=i==0
    doc.add_paragraph().paragraph_format.space_after=Pt(0)
lines=text.splitlines();i=0;compact_evidence=False
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('!['):i+=1;continue
    if line.startswith('|'):
        data=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=[s.strip() for s in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',c) for c in cells):data.append(cells)
            i+=1
        table(data);continue
    if line.startswith(chr(96)*3):
        p=doc.add_paragraph();i+=1
        while i<len(lines) and not lines[i].startswith(chr(96)*3):
            cmd=lines[i]
            chunks=re.split(r' (?=--)',cmd) if len(cmd)>100 and cmd.startswith('python ') else [cmd]
            wrapped=(' '+chr(92)+'\n  ').join(chunks)
            r=p.add_run(wrapped+'\n');r.font.name='Consolas';r.font.size=Pt(8.5);i+=1
        i+=1;continue
    if line.startswith('> 公式'):
        prefix,formula=line[2:].split('：',1)
        p=doc.add_paragraph();p.paragraph_format.keep_together=True
        r=p.add_run(prefix+'  ');r.font.size=Pt(10)
        math=OxmlElement('m:oMath');mr=OxmlElement('m:r');mt=OxmlElement('m:t')
        mt.text=formula;mr.append(mt);math.append(mr);p._p.append(math);i+=1;continue
    if line.startswith('#'):
        level=len(line)-len(line.lstrip('#'));title=line[level:].strip()
        if title.startswith('D.2 本地关键证据哈希'):compact_evidence=True
        if level==1:p=doc.add_paragraph(title,style='Title')
        else:
            p=doc.add_heading(title,level=min(level-1,3))
            if level==2 and title.startswith('附录 A'):p.paragraph_format.page_break_before=True
        if line in figures:
            path,caption=figures[line];ip=doc.add_paragraph();ip.paragraph_format.keep_with_next=True
            ip.add_run().add_picture(str(path),width=Inches(6.84))
            cp=doc.add_paragraph(caption,style='Caption');cp.paragraph_format.keep_with_next=True
        i+=1;continue
    p=doc.add_paragraph();runs(p,line)
    if 'SHA256：' in line:
        for r in p.runs:r.font.size=Pt(9)
    if compact_evidence and (line.startswith('outputs/') or line.startswith('SHA256：')):
        p.paragraph_format.line_spacing=1.05
        p.paragraph_format.space_after=Pt(4)
        p.paragraph_format.keep_with_next=line.startswith('outputs/')
        for r in p.runs:r.font.size=Pt(9 if line.startswith('SHA256：') else 10)
    i+=1
doc.save(OUT)
print(json.dumps({'docx':str(OUT),'markdown':str(MD),'characters':len(text),'paragraphs':len(doc.paragraphs),'tables':len(doc.tables),'images':len(doc.inline_shapes),'difference_classes':diffs,'evidence_checks':'passed'},ensure_ascii=False))
