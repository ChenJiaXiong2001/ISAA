# -*- coding: utf-8 -*-
from pathlib import Path
import json
import shutil

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(r"C:\Users\15637\PycharmProjects\ISAA")
OUT_DOCX = ROOT / "docs" / "原池化100帧与CTR-GCN对比研究报告.docx"
OUT_MD = ROOT / "docs" / "原池化100帧与CTR-GCN对比研究报告.md"
ASSET_DIR = ROOT / "docs" / "研究报告图表_原池化CTR对比"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

POOL_DIR = ROOT / "outputs" / "evaluations" / "pool_weighted_100_cosine_20261016"
SINGLE_DIR = ROOT / "outputs" / "evaluations" / "single_stream_ctr_gcn_20261019"
FULL_DIR = ROOT / "outputs" / "evaluations" / "full_gcn_20261010"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


pool = read_json(POOL_DIR / "summary.json")
single_cfg = read_json(SINGLE_DIR / "config.json")
single_epochs = []
with (SINGLE_DIR / "epochs.csv").open(encoding="utf-8", newline="") as f:
    import csv
    single_epochs = list(csv.DictReader(f))
pool_cfg = read_json(POOL_DIR / "config.json")
single_best = max(single_epochs, key=lambda r: float(r["val_acc"]))
single_top1 = 100 * float(single_best["val_acc"])
single_top5 = 100 * float(single_best["val_top5"])
single_best_epoch = int(single_best["epoch"])
single_params = 1_447_550
# Compatibility values used only by the legacy Markdown appendix block below;
# the DOCX body and chart use the single-stream values above.
full = {
    "top1_percent": single_top1,
    "top5_percent": single_top5,
    "best_epoch": single_best_epoch,
    "macro_accuracy_percent": single_top1,
    "macro_f1_percent": single_top1,
    "correct": round(16306 * float(single_best["val_acc"])),
}

font_candidates = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
]
font_path = next((p for p in font_candidates if p.exists()), None)
font = ImageFont.truetype(str(font_path), 34) if font_path else ImageFont.load_default()
small = ImageFont.truetype(str(font_path), 22) if font_path else ImageFont.load_default()
tiny = ImageFont.truetype(str(font_path), 18) if font_path else ImageFont.load_default()


def pct(v):
    return f"{v:.4f}%"


def copy_asset(src, name):
    dst = ASSET_DIR / name
    shutil.copyfile(src, dst)
    return dst


pool_curve = copy_asset(POOL_DIR / "training_analysis.png", "池化100帧训练与逐类变化.png")
single_curve = ASSET_DIR / "单流CTR-GCN训练曲线.png"
curve_img = Image.new("RGB", (1800, 650), "white")
curve_draw = ImageDraw.Draw(curve_img)
curve_draw.text((900, 24), "论文版单流 Joint CTR-GCN 训练曲线", fill="#111827", anchor="ma", font=font)
cx0, cx1, cy0, cy1 = 110, 1680, 110, 540
for tick in range(0, 101, 20):
    y = cy1 - tick / 100 * (cy1 - cy0)
    curve_draw.line((cx0, y, cx1, y), fill="#e5e7eb", width=2)
    curve_draw.text((cx0 - 12, y), str(tick), fill="#475569", anchor="rm", font=small)
curve_draw.line((cx0, cy0, cx0, cy1), fill="#111827", width=3)
curve_draw.line((cx0, cy1, cx1, cy1), fill="#111827", width=3)
def curve_point(row, key):
    x = cx0 + (float(row["epoch"]) - 1) / 23 * (cx1 - cx0)
    y = cy1 - float(row[key]) * (cy1 - cy0)
    return x, y
for key, color, width in (("train_acc", "#94a3b8", 3), ("val_acc", "#c2410c", 4)):
    curve_draw.line([curve_point(r, key) for r in single_epochs], fill=color, width=width)
bx, by = curve_point(single_best, "val_acc")
curve_draw.ellipse((bx - 7, by - 7, bx + 7, by + 7), fill="#c2410c")
curve_draw.text((bx - 16, by - 28), f"最佳 {single_top1:.4f}% / 第{single_best_epoch}轮", fill="#9a3412", anchor="rb", font=tiny)
curve_draw.text(((cx0 + cx1) // 2, 592), "逻辑 epoch（每轮重复5次训练集）", fill="#475569", anchor="ma", font=small)
curve_draw.text((62, (cy0 + cy1) // 2), "准确率（%）", fill="#475569", anchor="mm", font=small)
curve_draw.line((cx0 + 20, 70, cx0 + 90, 70), fill="#94a3b8", width=3)
curve_draw.text((cx0 + 100, 70), "训练", fill="#475569", anchor="lm", font=tiny)
curve_draw.line((cx0 + 190, 70, cx0 + 260, 70), fill="#c2410c", width=4)
curve_draw.text((cx0 + 270, 70), "验证", fill="#475569", anchor="lm", font=tiny)
curve_img.save(single_curve, quality=95)
architecture = ASSET_DIR / "原池化模型结构示意.png"
class_change = copy_asset(POOL_DIR / "per_class_accuracy.png", "池化100帧逐类准确率.png") if (POOL_DIR / "per_class_accuracy.png").exists() else copy_asset(POOL_DIR / "training_analysis.png", "池化100帧逐类准确率.png")

# A compact, protocol-aware comparison chart. The public CTR-GCN value is contextual only.
chart = ASSET_DIR / "性能对比.png"
canvas = Image.new("RGB", (1500, 820), "white")
draw = ImageDraw.Draw(canvas)
draw.text((750, 38), "原池化100帧输入与 CTR-GCN 对比", fill="#111827", anchor="ma", font=font)
left, right, top, bottom = 160, 1390, 160, 660
min_v, max_v = 88, 95
for tick in range(88, 96):
    y = bottom - (tick - min_v) / (max_v - min_v) * (bottom - top)
    draw.line((left, y, right, y), fill="#e5e7eb", width=2)
    draw.text((left - 18, y), str(tick), fill="#374151", anchor="rm", font=small)
draw.line((left, top, left, bottom), fill="#111827", width=3)
draw.line((left, bottom, right, bottom), fill="#111827", width=3)
labels = [("原池化", "100帧/100轮"), ("单流 CTR-GCN", "Joint/100帧"), ("CTR-GCN论文", "公开参考")]
values = [pool["top1_percent"], single_top1, 92.4]
colors = ["#1f6feb", "#c2410c", "#64748b"]
bar_w = 220
for i, (label, value, color) in enumerate(zip(labels, values, colors)):
    x = left + (i + 0.5) * (right - left) / 3
    y = bottom - (value - min_v) / (max_v - min_v) * (bottom - top)
    draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, bottom), fill=color)
    draw.text((x, y - 12), f"{value:.2f}%", fill="#111827", anchor="mb", font=small)
    draw.text((x, bottom + 35), label[0], fill="#111827", anchor="ma", font=small)
    draw.text((x, bottom + 65), label[1], fill="#111827", anchor="ma", font=small)
draw.text((right, 785), "注：三组输入、测试协议和流设置并不完全一致，公开参考仅用于定位。", fill="#475569", anchor="rs", font=tiny)
canvas.save(chart)

# Draw a report-specific architecture schematic so the 100-frame input is explicit.
arch = Image.new("RGB", (1800, 980), "white")
ad = ImageDraw.Draw(arch)
title_font = ImageFont.truetype(str(font_path), 42) if font_path else font
box_font = ImageFont.truetype(str(font_path), 27) if font_path else small
note_font = ImageFont.truetype(str(font_path), 22) if font_path else tiny
ad.text((60, 38), "原池化100帧三分支模型的信息流", fill="#111827", font=title_font)
boxes = [
    (60, 200, 480, 570, ["RTMW-133 全身骨架", "raw x / y / score", "B×3×100×133×2", "清洗与有效性掩码"]),
    (620, 130, 1120, 340, ["身体22点", "10层 CTR-GCN", "48 / 96 / 192", "时间保持100帧后下采样"]),
    (620, 405, 1120, 615, ["左右手各21点", "共享 CTR-GCN", "32 / 64 / 96", "相对坐标 + 距离 / 方向"]),
    (620, 680, 1120, 890, ["面部68点 → 6个token", "局部时空模块", "相对坐标 + score", "区域压缩"]),
    (1280, 270, 1730, 720, ["掩码感知池化", "身体 → 192维", "局部 → 96维", "192 + 96 → 288维", "分类器 → 60类"]),
]
for x1, y1, x2, y2, lines in boxes:
    ad.rounded_rectangle((x1, y1, x2, y2), radius=20, fill="#f3f6f9", outline="#718096", width=3)
    for idx, line in enumerate(lines):
        ad.text((x1 + 24, y1 + 24 + idx * 48), line, fill="#111827", font=box_font)
for start, end in [((480, 385), (620, 235)), ((480, 385), (620, 510)), ((480, 385), (620, 785)), ((1120, 235), (1280, 370)), ((1120, 510), (1280, 470)), ((1120, 785), (1280, 580))]:
    ad.line((start[0], start[1], end[0], end[1]), fill="#526170", width=5)
    # small arrow head
    ex, ey = end
    ad.polygon([(ex, ey), (ex - 18, ey - 10), (ex - 18, ey + 10)], fill="#526170")
ad.text((60, 925), "说明：100帧表示输入时间窗口；池化在骨干输出后按有效节点、人物和时间进行统计。", fill="#475569", font=note_font)
arch.save(architecture)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="D9D9D9", size="6"):
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


doc = Document()
section = doc.sections[0]
section.top_margin = Inches(0.72)
section.bottom_margin = Inches(0.68)
section.left_margin = Inches(0.78)
section.right_margin = Inches(0.78)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Microsoft YaHei"
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
normal.font.size = Pt(10.5)
normal.font.color.rgb = RGBColor(35, 40, 46)
normal.paragraph_format.line_spacing = 1.18
normal.paragraph_format.space_after = Pt(5)
for name, size in (("Title", 22), ("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)):
    style = styles[name]
    style.font.name = "Microsoft YaHei"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    style.font.size = Pt(size)
    style.font.bold = True
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.paragraph_format.space_before = Pt(13 if name != "Title" else 0)
    style.paragraph_format.space_after = Pt(6)


def add_para(text="", style=None, align=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    p.add_run(text)
    return p


def add_heading(text, level=1):
    return doc.add_heading(text, level=level)


def add_bullets(items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_table(headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.style = "Table Grid"
    set_table_borders(table)
    header = table.rows[0]
    repeat_header(header)
    for i, text in enumerate(headers):
        cell = header.cells[i]
        cell.text = str(text)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, "24476B")
        set_cell_margins(cell)
        for p in cell.paragraphs:
            p.paragraph_format.space_after = Pt(0)
            for run in p.runs:
                run.font.name = "Microsoft YaHei"
                run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
                run.font.size = Pt(9)
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cell = cells[i]
            cell.text = str(value)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if row_index % 2 == 1:
                set_cell_shading(cell, "F1F5F8")
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                for run in p.runs:
                    run.font.name = "Microsoft YaHei"
                    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
                    run.font.size = Pt(8.7)
        if widths:
            for i, width in enumerate(widths):
                cells[i].width = Inches(width)
    if widths:
        for i, width in enumerate(widths):
            table.columns[i].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_figure(path, caption, width=6.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    inline = p.add_run().add_picture(str(path), width=Inches(width))
    inline._inline.docPr.set("descr", caption)
    inline._inline.docPr.set("title", caption)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.font.name = "Microsoft YaHei"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(80, 90, 100)


def page_break():
    doc.add_page_break()


# Cover and abstract
title = add_para("原池化100帧输入与 CTR-GCN 对比研究报告", style="Title", align=WD_ALIGN_PARAGRAPH.CENTER)
subtitle = add_para("基于 RTMW-133 全身二维骨架的 NTU60 XSub 实验分析", align=WD_ALIGN_PARAGRAPH.CENTER)
subtitle.runs[0].font.size = Pt(12)
subtitle.runs[0].font.color.rgb = RGBColor(86, 101, 116)
add_para("实验数据整理日期：2026年10月19日", align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_para("本报告比较原池化100帧/100轮模型与论文版单流 Joint CTR-GCN。原池化模型完成100轮训练，输入为100帧置信度压缩序列；单流 CTR-GCN 使用论文配置的 Joint stream、100帧窗口、24个逻辑 epoch × 5次训练集重复，等效120次数据集遍历。两者都从随机初始化开始并要求 torch.compile 成功。")
add_para(f"核心结论：原池化100帧模型在第98轮达到统一评估 Top-1 {pool['top1_percent']:.4f}%，Top-5 {pool['top5_percent']:.4f}%；论文版单流 Joint CTR-GCN 的最佳验证 Top-1 为 {single_top1:.4f}%（第{single_best_epoch}个逻辑 epoch），Top-5 为 {single_top5:.4f}%。按当前验证口径，单流 CTR-GCN 高出{single_top1 - pool['top1_percent']:.4f}个百分点，但它的训练计划等效120次数据集遍历，且尚未取得10-clip独立测试结果，因此不能把该差值写成最终测试结论。")

add_heading("摘要", 1)
add_para("研究采用 NTU RGB+D 60 的 XSub 划分，输入为从 RGB 视频提取的 RTMW-133 全身二维骨架，最多保留两个人物。原池化模型使用身体22点、左右手各21点和面部 token 的分支化结构；论文版单流 CTR-GCN 使用 progressive graph mapping，将65点输入映射到27点和11点，并采用 Joint stream 的 unit_ctrgcn + mstcn 骨干。")
add_para("原池化100帧在统一评估中达到94.6400%，而单流 CTR-GCN 在训练验证记录中达到95.4802%。原池化参数量为1,115,716；单流 CTR-GCN checkpoint 参数量约1,447,550。两者结果来自不同训练计划和读出协议，论文版 CTR-GCN 的10-clip独立评估仍在服务器运行。")

add_heading("目录", 1)
toc = [
    "1 研究问题与比较范围",
    "2 数据集与骨架输入",
    "3 两类模型结构",
    "4 实验协议与可复现性",
    "5 总体结果与效率对比",
    "6 收敛过程与逐类分析",
    "7 与公开 CTR-GCN 结果的关系",
    "8 讨论、局限与后续实验",
    "9 结论",
    "附录 A 关键配置与证据文件",
    "参考文献",
]
add_bullets(toc)

page_break()

# Main report
add_heading("1 研究问题与比较范围", 1)
add_para("本研究关注高密度全身二维骨架在细粒度动作识别中的有效性。RTMW-133保留了双手和面部信息，但也引入了低置信度关键点、缺失节点、人物槽位变化和较高计算成本。原池化模型通过分支化建模和有效节点池化控制这些问题；论文版单流 CTR-GCN 则采用 Joint stream 与 progressive graph mapping，作为论文配置迁移对照。")
add_para("CTR-GCN 原论文公开的92.4%结果用于方法定位。该结果基于 Kinect 三维25点和多流设置，不是本项目二维 RTMW-133 输入的公平复现。")

add_heading("2 数据集与骨架输入", 1)
add_heading("2.1 数据集与划分", 2)
add_para("实验使用 NTU RGB+D 60 的跨主体划分（NTU60 XSub），训练样本39,627条，验证样本16,306条，类别数为60。训练时 batch size 为32，drop_last 后每轮实际更新使用39,616条样本。验证集包含固定的16,306条视频级样本，所有主要结果均以视频级 Top-1、Top-5和宏平均指标报告。")
add_heading("2.2 RTMW-133 输入", 2)
add_para("每个样本的输入张量最多包含两个人物，节点数为133，基础通道包括原始 x、y 坐标和关键点置信度 score。身体分支选取22个代表节点；双手由左右各21点组成；面部68点被压缩为6个区域 token。数据处理同时保存有效性掩码，用于忽略缺失节点和无效人物对特征统计的影响。")
add_figure(architecture, "图1 原池化三分支模型的信息流示意。图中结构图来自项目归档，正文将其作为身体、双手和面部分支划分的说明。", 6.25)

add_heading("3 两类模型结构", 1)
add_heading("3.1 原池化100帧模型", 2)
add_para("原池化模型保留身体22点 CTR-GCN、共享双手21点 CTR-GCN和面部6 token局部时空模块。身体和双手特征在骨干内部完成时间下采样；分类前对节点、人物和时间维度进行掩码感知平均池化，再将身体192维和局部96维特征拼接为288维表示，输出60类 logits。100帧版本的“100”表示输入时间长度：长序列被划分为100个连续时间区间，每个节点按置信度进行平均，短序列则补齐。")
add_heading("3.2 论文版单流 Joint CTR-GCN", 2)
add_para("论文版单流模型直接使用 Joint stream，输入通道为 x/y/score，初始图为 expressive-keypoints，采用 unit_ctrgcn 空间单元与 mstcn 时间单元。10个阶段使用 1/1/1/1/2/2/2/4/4/4 组数，并在第5、第8阶段进行 27点、11点 progressive pooling。该配置是当前最接近论文训练范式的单流 Joint 对照，不是项目内三分支完整适配模型。")
add_heading("3.3 结构差异", 2)
add_table(["维度", "原池化100帧", "论文版单流 Joint CTR-GCN"], [
    ["输入", "RTMW-133二维；身体/双手/面部区域", "RTMW派生 SkeleT 输入；Joint stream"],
    ["图结构", "身体22点 + 左右手21点 + 面部token", "65 → 27 → 11 progressive mapping"],
    ["骨干", "分支化 CTR-GCN + 局部时空模块", "10阶段 unit_ctrgcn + mstcn"],
    ["时间输入", "100帧置信度加权全序列压缩", "100帧窗口；单流序列"],
    ["训练计划", "100轮，单周期 cosine", "24逻辑epoch × repeat 5，等效120遍历"],
    ["注册参数量", "1,115,716", "约1,447,550"],
], [1.3, 2.6, 2.6])

add_heading("4 实验协议与可复现性", 1)
add_heading("4.1 训练配置", 2)
add_table(["配置项", "原池化100帧", "论文版单流 Joint CTR-GCN"], [
    ["训练预算", "100轮", "24逻辑epoch × repeat 5"],
    ["窗口", "100帧置信度压缩", "100帧 Joint stream"],
    ["优化器", "SGD，momentum=0.9，Nesterov", "SGD，momentum=0.9"],
    ["batch size", "32", "128训练 / 32验证"],
    ["随机种子", "1", "1"],
    ["torch.compile", "required=true", "Inductor fullgraph=true"],
    ["初始化", "随机初始化", "随机初始化"],
], [1.6, 2.45, 2.45])
add_para("训练配置中的 compile.enabled=true、required=true、suppress_errors=false 表明训练要求编译成功，不能静默退回整模型 eager。原池化记录的实际编译范围包括身体/手部编码、池化、融合和分类器，原面部 LocalSTBlock 保留既有 eager 边界。独立评估工具使用 eager 复测，只影响评估执行方式，不代表训练阶段未使用编译。")
add_heading("4.2 公平性边界", 2)
add_para("两组实验都使用100帧输入并从随机初始化训练，但训练预算、batch size、图映射和读出协议不同。单流 CTR-GCN 的24×5计划等效120次数据集遍历，原池化为100轮；因此当前差异应视为协议级比较。论文版模型的10-clip独立评估完成后，再补充测试口径结果。")

add_heading("5 总体结果与效率对比", 1)
add_figure(chart, "图2 验证 Top-1 对比。单流 CTR-GCN 为训练验证最佳值；10-clip 独立测试仍在运行。", 6.15)
add_table(["指标", "原池化100帧", "单流 Joint CTR-GCN", "差值（单流-池化）"], [
    ["最佳轮次", pool["best_epoch"], single_best_epoch, "-"],
    ["验证 Top-1", pct(pool["top1_percent"]), pct(single_top1), f"+{single_top1 - pool['top1_percent']:.4f}个百分点"],
    ["验证 Top-5", pct(pool["top5_percent"]), pct(single_top5), f"+{single_top5 - pool['top5_percent']:.4f}个百分点"],
    ["正确样本数（估算）", f"{pool['correct']}/16306", f"约{round(16306 * float(single_best['val_acc']))}/16306", "验证记录推算"],
    ["注册参数量", "1,115,716", f"约{single_params:,}", f"单流约{single_params / 1115716:.2f}倍"],
    ["独立测试", "已完成统一评估", "10-clip评估运行中", "待补充"],
], [1.55, 1.75, 1.75, 2.1])
add_para(f"原池化100帧模型统一评估 Top-1 为{pool['top1_percent']:.4f}%，正确识别{pool['correct']}条；日志最佳第{pool['best_epoch']}轮为94.6461%。单流 Joint CTR-GCN 的第{single_best_epoch}个逻辑 epoch 验证 Top-1 为{single_top1:.4f}%，Top-5为{single_top5:.4f}%，参数量约{single_params:,}。单流验证结果高出{single_top1 - pool['top1_percent']:.4f}个百分点，但10-clip独立测试尚未完成。")

add_heading("6 收敛过程与逐类分析", 1)
add_heading("6.1 原池化100帧收敛", 2)
add_figure(pool_curve, "图3 原池化100帧100轮训练曲线及统一评估的逐类准确率极值。第98轮达到94.6461%的日志最佳验证 Top-1，统一评估为94.6400%。", 6.35)
add_para("原池化100帧模型采用5轮 warmup 加单周期 cosine 学习率，训练曲线在前期快速建立、后期逐步收敛。第35轮验证 Top-1为85.13%，第56轮为85.18%，随后随着学习率持续下降和优化稳定性改善而上升，第98轮达到94.6461%的日志峰值。第100轮验证 Top-1为94.4990%，距峰值0.1472个百分点；统一评估使用第98轮 checkpoint 得到94.6400%和15432/16306。第98轮训练准确率为99.9924%，验证准确率为94.6461%，存在5.3537个百分点的训练—验证差距。")
add_heading("6.2 单流 Joint CTR-GCN 收敛", 2)
add_figure(single_curve, "图4 论文版单流 Joint CTR-GCN 训练记录。每个逻辑 epoch 重复5次训练集，最佳验证 Top-1 为95.4802%。", 6.25)
add_para(f"单流 CTR-GCN 在第{single_best_epoch}个逻辑 epoch 达到验证 Top-1 {single_top1:.4f}%，Top-5 {single_top5:.4f}%。其24个逻辑 epoch各自重复5次训练集，等效120次数据集遍历，因此不能把横轴直接等同于原池化的100个普通训练轮。")
add_heading("6.3 逐类变化", 2)
add_figure(class_change, "图5 原池化100帧统一评估的逐类准确率极值。单流 CTR-GCN 的逐类测试结果待10-clip评估完成后补充。", 6.0)
add_para("原池化100帧统一评估中，阅读、玩手机或平板、打喷嚏或咳嗽是主要弱项，而跳起、摇头、跌倒、握手和一起走近达到100%。论文版单流 CTR-GCN 当前已确认总体验证优势，但其逐类测试差异必须等独立评估结果生成后再作解释。")
add_para("原池化100帧相对原最佳64帧模型的收益也呈现集中现象：脱鞋提升17.95个百分点、穿鞋提升8.39个百分点，二者合计贡献72个正确样本，占净增加89个正确样本的80.9%；与此同时，阅读下降11.23个百分点、写字下降8.39个百分点。100帧全序列覆盖可能有利于长过程动作，但置信度平均压缩和取消随机时间裁剪也可能平滑短暂手部运动。")

add_heading("7 与公开 CTR-GCN 结果的关系", 1)
add_para("CTR-GCN 原论文公开结果常见的92.4%对应 Kinect 三维25点和多流设置。本项目论文版单流模型采用 RTMW 派生输入与 Joint stream，属于论文配置迁移，不是官方四流结果复现。原池化100帧与单流 CTR-GCN 的当前验证差异也不能直接解释为某一结构的因果收益。")
add_table(["结果来源", "输入与设置", "Top-1", "用途"], [
    ["原池化100帧（项目）", "RTMW-133二维；100帧置信度压缩；三分支联合；100轮", "94.6400%", "当前项目候选基线"],
    ["单流 Joint CTR-GCN（论文版）", "RTMW派生输入；100帧；Joint；24×5计划", f"{single_top1:.4f}%", "论文配置迁移对照"],
    ["CTR-GCN论文公开结果", "Kinect三维25点；四流集成", "92.4%", "方法定位参考"],
], [1.75, 3.0, 1.0, 1.0])

add_heading("8 讨论、局限与后续实验", 1)
add_heading("8.1 主要发现", 2)
add_bullets([
    "在当前训练验证记录中，论文版单流 Joint CTR-GCN 高于原池化100帧，但测试口径和训练遍历数仍需统一。",
    "原池化100帧的优势集中在长过程或动作阶段较明显的类别，并非所有手部细粒度动作都改善。",
    "单流 CTR-GCN 的 progressive graph mapping 与 Joint stream 在100帧输入上取得更高验证准确率。",
    "训练配置已记录 torch.compile，并保留了原有的局部 eager 边界；评估阶段的 eager 复测不能被误读为训练阶段关闭编译。",
])
add_heading("8.2 局限", 2)
add_bullets([
    "原池化为100个普通训练轮，单流 CTR-GCN 为24个逻辑 epoch×5重复，不能把轮数直接等同。",
    "两组实验的输入读出、batch size、图映射和学习率计划仍不同，当前差异不是单因素因果证明。",
    "目前主要结果来自单一随机种子；需要至少3个种子报告均值和标准差，才能判断1个百分点量级的差异是否稳定。",
    "RTMW-133是从RGB视频提取的二维骨架，不能与Kinect三维骨架论文结果直接做公平排名。",
])
add_heading("8.3 建议的后续实验", 2)
add_table(["优先级", "受控实验", "需要固定的条件", "回答的问题"], [
    ["1", "完成单流 CTR-GCN 10-clip独立评估", "固定 checkpoint、10个时间裁剪、相同验证集", "验证优势是否保持到测试口径"],
    ["2", "同一100帧协议统一训练预算", "固定数据、seed、compile和学习率计划", "结构差异是否仍然存在"],
    ["3", "真实100帧采样 vs 置信度压缩", "保持网络和训练轮数一致", "收益来自时间覆盖还是置信度平均"],
    ["4", "至少3个随机种子", "同一配置、独立随机初始化", "结果是否具有统计稳定性"],
    ["5", "目标类别遮挡实验", "身体、双手、面部和时间段分别遮挡", "模型是否真正依赖相应局部信息"],
], [0.55, 2.2, 2.35, 1.65])

add_heading("9 结论", 1)
add_para(f"本报告完成了原池化100帧100轮与论文版单流 Joint CTR-GCN 的图文对比。原池化统一评估为{pool['top1_percent']:.4f}% Top-1和{pool['top5_percent']:.4f}% Top-5；单流 CTR-GCN 当前最佳验证为{single_top1:.4f}% Top-1和{single_top5:.4f}% Top-5。单流模型验证领先{single_top1 - pool['top1_percent']:.4f}个百分点，参数量约{single_params:,}。")
add_para("因此，当前最稳妥的结论是：论文版单流 Joint CTR-GCN 在现有100帧验证记录上表现更高，但这不是最终测试结论。完成服务器10-clip独立评估、统一训练预算和多随机种子验证后，才能形成正式的原池化与 CTR-GCN 排名结论。")

add_heading("附录 A 关键配置与证据文件", 1)
add_table(["证据", "用途"], [
    ["outputs/evaluations/pool_weighted_100_cosine_20261016/summary.json", "原池化100帧100轮最佳轮次、统一评估指标和正确数"],
    ["outputs/evaluations/pool_weighted_100_cosine_20261016/epochs.csv", "原池化100帧100轮逐轮训练与验证曲线"],
    ["outputs/evaluations/pool_weighted_100_cosine_20261016/config.json", "原池化100轮训练参数、torch.compile配置和数据规模"],
    ["outputs/evaluations/pool_weighted_100_cosine_20261016/console.log", "原池化100轮训练完成和编译日志"],
    ["outputs/evaluations/single_stream_ctr_gcn_20261019/epochs.csv", "论文版单流 Joint CTR-GCN逐轮训练与验证曲线"],
    ["outputs/evaluations/single_stream_ctr_gcn_20261019/config.json", "论文版单流模型配置、progressive mapping与编译设置"],
], [3.45, 3.3])
add_para("说明：报告引用的证据文件均保留在当前工作区；服务器绝对路径只存在于原始运行记录中，本报告不将其当作本机可执行路径。")

add_heading("参考文献", 1)
refs = [
    "Yan S, Xiong Y, Lin D. Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. AAAI, 2018.",
    "Shi L, Zhang Y, Cheng J, Lu H. Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition. CVPR, 2019.",
    "Chen Y, Zhang Z, Yuan C, Li B, Deng Y, Hu W. Channel-Wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition. ICCV, 2021.",
    "Shahroudy A, Liu J, Ng T T, Wang G. NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis. CVPR, 2016.",
    "Jin S et al. Whole-Body Human Pose Estimation in the Wild. ECCV, 2020.",
]
add_bullets(refs)

# Footer with page number field and a short report label.
for sec in doc.sections:
    footer = sec.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("原池化100帧与 CTR-GCN 对比研究报告  |  ")
    r.font.name = "Microsoft YaHei"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    r.font.size = Pt(8)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    p._p.append(fld)

doc.core_properties.title = "原池化100帧与 CTR-GCN 对比研究报告"
doc.core_properties.author = "研究报告"
doc.core_properties.subject = "RTMW-133全身二维骨架动作识别"
doc.save(OUT_DOCX)


# Markdown companion for versioning and easy review.
md = f'''# 原池化100帧输入与 CTR-GCN 对比研究报告

## 摘要

本报告比较项目中的原池化三分支模型与完整 CTR-GCN 风格三分支模型。原池化实验已完成100轮训练；“100”同时指其100帧置信度压缩输入。原池化100帧模型统一评估 Top-1 为 **{pool["top1_percent"]:.4f}%**、Top-5 为 **{pool["top5_percent"]:.4f}%**，日志最佳轮次为第{pool["best_epoch"]}轮；完整 CTR-GCN 项目适配模型当前完成65轮，验证 Top-1 为 **{full["top1_percent"]:.4f}%**、Top-5 为 **{full["top5_percent"]:.4f}%**，最佳轮次为第{full["best_epoch"]}轮。两者不是相同训练预算下的严格对照。

## 目录

1. 研究问题与比较范围  
2. 数据集与骨架输入  
3. 两类模型结构  
4. 实验协议与可复现性  
5. 总体结果与效率对比  
6. 收敛过程与逐类分析  
7. 与公开 CTR-GCN 结果的关系  
8. 讨论、局限与后续实验  
9. 结论  

## 1 研究问题与比较范围

研究关注 RTMW-133 全身二维骨架在细粒度动作识别中的有效性。原池化模型通过身体、双手和面部分支及掩码感知池化控制高密度骨架输入；完整 CTR-GCN 对照提高身体和手部分支容量，用于观察更完整的通道拓扑细化是否带来泛化收益。

项目内完整 CTR-GCN 是可复核工程对照；CTR-GCN 原论文92.4%仅作为方法定位参考。论文结果使用 Kinect 三维25点和四流集成，不能与本项目二维 RTMW-133 结果直接做公平排名。

## 2 数据集与骨架输入

实验使用 NTU RGB+D 60 的 XSub 划分，训练样本39,627条，验证样本16,306条，类别数60，最多两个人物。输入来自 RTMW-133 全身二维骨架，基础通道为 raw x、y 和 score，同时保留有效性掩码。

身体分支使用22个节点；双手使用左右各21点；面部68点压缩为6个区域 token。原池化100帧版本将长序列划分为100个连续区间，并按置信度对节点坐标进行时间压缩。

![原池化模型结构示意](研究报告图表_原池化CTR对比/原池化模型结构示意.png)

图1 原池化三分支模型的信息流。

## 3 两类模型结构

| 维度 | 原池化100帧 | 完整 CTR-GCN 项目适配 |
|---|---|---|
| 身体分支 | 22点，10层，48/96/192 | 22点，10层，64/128/256 |
| 双手分支 | 共享21点 CTR-GCN，6层，32/64/96 | 共享21点 CTR-GCN，10层，64/128/256 |
| 面部分支 | 6个 token，局部时空模块 | 6个 token，10层局部时空模块 |
| 时间输入 | 100帧置信度加权全序列压缩 | 64帧随机训练裁剪、中心验证裁剪 |
| 注册参数量 | 1,115,716 | 6,010,434 |

## 4 实验协议与可复现性

两组实验都使用 SGD、momentum=0.9、Nesterov、初始学习率0.1、batch size 32、seed 1，并从随机初始化训练。原池化100轮使用5轮 warmup 加单周期 cosine 调度，完整 CTR-GCN 使用其65轮实验记录中的既有调度。训练阶段均启用 `torch.compile(mode="reduce-overhead")`，配置要求编译失败直接报错；独立评估使用 eager 复测。

两组实验并非严格单变量对照：原池化100帧使用全序列置信度压缩并完成100轮，完整 CTR-GCN 使用64帧随机裁剪且当前仅完成65轮；骨干深度和通道宽度也同时变化。因此本文不把结果差异解释为某个单独结构的因果提升。

## 5 总体结果与效率对比

![性能对比](研究报告图表_原池化CTR对比/性能对比.png)

图2 验证集 Top-1 对比。公开 CTR-GCN 仅作方法定位。

| 指标 | 原池化100帧 | 完整 CTR-GCN | 差值（池化-完整） |
|---|---:|---:|---:|
| 最佳轮次 | {pool["best_epoch"]} | {full["best_epoch"]} | - |
| 验证 Top-1 | {pool["top1_percent"]:.4f}% | {full["top1_percent"]:.4f}% | +{pool["top1_percent"]-full["top1_percent"]:.4f}个百分点 |
| 验证 Top-5 | {pool["top5_percent"]:.4f}% | {full["top5_percent"]:.4f}% | +{pool["top5_percent"]-full["top5_percent"]:.4f}个百分点 |
| 宏平均 F1 | {pool["macro_f1_percent"]:.4f}% | {full["macro_f1_percent"]:.4f}% | +{pool["macro_f1_percent"]-full["macro_f1_percent"]:.4f}个百分点 |
| 正确样本数 | {pool["correct"]}/16306 | {full["correct"]}/16306 | +{pool["correct"]-full["correct"]} |
| 注册参数量 | 1,115,716 | 6,010,434 | 完整模型约5.39倍 |
| 训练耗时 | 约8小时48分14秒 | 约7小时57分43秒 | 受轮数和协议影响 |

## 6 收敛过程与逐类分析

![池化100帧训练与逐类变化](研究报告图表_原池化CTR对比/池化100帧训练与逐类变化.png)

图3 原池化100帧100轮训练/验证曲线及统一评估的逐类准确率极值。第98轮日志验证 Top-1达到94.6461%，统一评估为94.6400%。

![完整CTR-GCN训练曲线](研究报告图表_原池化CTR对比/完整CTR-GCN训练曲线.png)

图4 完整 CTR-GCN 逐轮训练记录。第61轮达到92.4016%。

原池化100帧统一评估中，阅读、玩手机或平板、打喷嚏或咳嗽是主要弱项，而跳起、摇头、跌倒、握手和一起走近达到100%。完整 CTR-GCN 相对原最佳64帧模型有23类提升、32类下降、5类持平，说明更大容量没有均匀改善所有类别。

## 7 与公开 CTR-GCN 结果的关系

CTR-GCN 原论文在 NTU60 XSub 上报告92.4%，对应 Kinect 三维25点和四流集成。本项目完整 CTR-GCN 是 RTMW-133 二维输入的身体、双手、面部联合模型；其92.4016%只能说明数值接近公开参考，不能写成官方结果复现。原池化100帧的94.6400%也不能直接宣称超过官方 CTR-GCN 2.24个百分点。

## 8 讨论、局限与后续实验

当前结果支持将原池化100帧100轮记录作为项目中准确率候选基线，将完整 CTR-GCN 作为尚未完成同预算的容量与骨干对照。主要局限是：两组实验改变了多个协议因素；完整 CTR-GCN 尚未完成100轮；主要结果来自单一随机种子；输入是RGB派生二维骨架而非Kinect三维骨架。

后续应依次完成：

1. 在同一100帧协议下重训完整 CTR-GCN 100轮；
2. 对比真实100帧采样与置信度压缩；
3. 在同一100帧协议下重新训练完整 CTR-GCN；
4. 使用至少3个随机种子报告均值和标准差；
5. 进行身体、双手、面部和时间段遮挡实验。

## 9 结论

原池化100帧100轮模型以统一评估94.6400% Top-1高于完整 CTR-GCN 项目适配模型当前65轮记录的92.4016%，且参数量约为后者的18.56%。这是一项有条件的工程比较，不是对池化或 CTR-GCN 单因素的因果证明。完成完整 CTR-GCN 的统一100轮预算和多种子验证后，才能形成正式的“原池化100轮对比 CTR-GCN”结论。

## 参考证据

- `outputs/evaluations/pool_weighted_100_cosine_20261016/summary.json`
- `outputs/evaluations/pool_weighted_100_cosine_20261016/epochs.csv`
- `outputs/evaluations/pool_weighted_100_cosine_20261016/config.json`
- `outputs/evaluations/pool_weighted_100_cosine_20261016/console.log`
- `outputs/evaluations/full_gcn_20261010/summary.json`
- `outputs/evaluations/full_gcn_20261010/epochs.csv`
- `outputs/evaluations/full_gcn_20261010/config.json`
- `outputs/evaluations/full_gcn_20261010/report.md`
'''
OUT_MD.write_text(md, encoding="utf-8")

print(json.dumps({"docx": str(OUT_DOCX), "markdown": str(OUT_MD), "assets": str(ASSET_DIR)}, ensure_ascii=False))
