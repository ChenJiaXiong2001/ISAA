import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

base = Path('/home/xiong/Desktop/ISAA/experiments/face_pair_raw_xy_20261104')
root = base / 'code'
sys.path.insert(0, str(root))
from tools.run_face_pair_fusion_suite import build_candidates, choose

out = base / 'fusion_suite_100_20261104'
cache = json.loads((base / 'fusion_cache_100/report.json').read_text())
summary = json.loads((out / 'summary.json').read_text())
selection = json.loads((out / 'selection_before_val.json').read_text())
cal = list(csv.DictReader((out / 'calibration_results.csv').open(encoding='utf-8-sig')))
val = list(csv.DictReader((out / 'validation_results.csv').open(encoding='utf-8-sig')))
assert len(cal) == len(val) == 112
assert summary['selection_unchanged_after_val']
assert hashlib.sha256((out / 'selection_before_val.json').read_bytes()).hexdigest() == summary['selection_frozen_before_val_sha256']
checks = {}
readouts = {}
for name in ('best', 'last'):
    meta = cache['checkpoints'][name]
    assert hashlib.sha256(Path(meta['path']).read_bytes()).hexdigest() == meta['sha256']
    assert meta['strict_state_load'] and meta['all_state_tensors_unchanged_during_inference']
    assert meta['finite_saved_model_and_optimizer']
    z = np.load(out / name / 'predictions.npz')
    preds, labels, cms, ids = (z[k] for k in ('predictions', 'labels', 'confusion_matrices', 'ids'))
    assert preds.shape == (56, 16306) and labels.shape == (16306,) and cms.shape == (56, 60, 60)
    assert len(set(ids)) == 56
    selected_cal = [r for r in cal if r['checkpoint'] == name and r['family'] != 'single_head']
    winner = min(selected_cal, key=lambda r: (-int(r['correct']), float(r['nll']), r['id']))
    assert winner['id'] == selection['checkpoints'][name]['calibration_selection']['id']
    for row in [r for r in val if r['checkpoint'] == name]:
        i = list(ids).index(row['id'])
        cm = np.bincount(labels * 60 + preds[i], minlength=3600).reshape(60, 60)
        assert np.array_equal(cm, cms[i])
        right = int((preds[i] == labels).sum())
        assert right == int(row['correct']) == int(cm.trace())
        assert abs(100 * right / len(labels) - float(row['top1_percent'])) < 1e-10
        for metric in ('nll', 'brier', 'ece15_percent', 'top5_percent', 'macro_f1_percent'):
            assert math.isfinite(float(row[metric]))
    fit = summary['checkpoints'][name]['temperature_fits']
    configs, weights = build_candidates(meta, [fit['hf']['temperature'], fit['bf']['temperature']])
    chosen_id = winner['id']
    observed_id = summary['checkpoints'][name]['highest_observed_val']['id']
    readouts[name] = {}
    for label, candidate in [('calibration_selected', chosen_id), ('highest_observed_val', observed_id)]:
        idx = next(i for i, c in enumerate(configs) if c['id'] == candidate)
        readouts[name][label] = dict(checkpoint=meta['path'], checkpoint_sha256=meta['sha256'],
                                    **configs[idx], HF_class_prior=weights[idx].tolist())
    checks[name] = dict(epoch=meta['epoch'], checkpoint_sha256=meta['sha256'], candidates=56,
                       samples=16306, confusion_prediction_correct_counts_match=True,
                       checkpoint_unchanged=True, state_unchanged_during_inference=True,
                       finite_model_optimizer_logits_metrics=True,
                       calibration_selected=chosen_id)
assert cache['args']['compile'] is True and cache['compile']['required'] is True
assert cache['args']['model_variant'] == 'best-face-pair-raw-xy'
assert summary['compile']['required'] is True and summary['compile']['fullgraph'] is True
audit = dict(status='passed', candidate_count=112, checks=checks,
             calibration_selection_frozen_before_val=True, backbone_and_fusion_compiled=True)
(out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
(out / 'fusion_readouts.json').write_text(json.dumps(readouts, indent=2) + '\n')
sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in [root/'tools/cache_face_pair_logits.py', root/'tools/check_face_pair_checkpoint.py',
                     root/'tools/run_face_pair_fusion_suite.py', root/'tools/face_pair_fusion_sweep.py',
                     root/'isaa/models/face_pair_fusion.py', root/'isaa/face_pair_training.py']}
(out / 'source_manifest.json').write_text(json.dumps(sources, indent=2) + '\n')
index = {(r['checkpoint'], r['id']): r for r in val}
def percent(name, candidate):
    return f"{float(index[name, candidate]['top1_percent']):.4f}%"
lines = ['# raw-XY 100 轮：HF/BF 融合测试', '',
         '日期：2026-11-04（Asia/Tokyo）。best.pt：第 85 轮；last.pt：第 100 轮。', '',
         '两份 checkpoint 各测试 56 种配置，共 112 项；每项使用完整的 16,306 条验证样本。',
         '内部校准集为训练受试者中留出的 3,191 条样本。校准集选参结果在读取验证缓存前已保存并冻结。', '',
         '| 融合方式 | best / 第85轮 | last / 第100轮 |',
         '|---|---:|---:|']
for title, candidate in [('HF 单头','raw/HF'), ('BF 单头','raw/BF'),
                         ('原硬路由','raw/hard'), ('等权概率平均','raw/static_gamma_0'),
                         ('按类准确率软加权 γ=1','raw/static_gamma_1'),
                         ('温度校准 + 软加权 γ=1','calibrated/static_gamma_1'),
                         ('温度校准 + 软加权 γ=2','calibrated/static_gamma_2'),
                         ('温度校准 + 软加权 γ=4','calibrated/static_gamma_4')]:
    lines.append(f'| {title} | {percent("best",candidate)} | {percent("last",candidate)} |')
lines += ['', '## 动态融合扫描最高值', '',
          '| 置信度方式 | best | last |', '|---|---:|---:|']
for mode, label in [('maxprob','最大概率'), ('margin','Top1–Top2 概率间隔'), ('normalized_entropy','归一化熵')]:
    values = [max(float(r['top1_percent']) for r in val if r['checkpoint']==n and r['family']=='dynamic' and r['confidence_mode']==mode) for n in ('best','last')]
    lines.append(f'| {label} | {values[0]:.4f}% | {values[1]:.4f}% |')
lines += ['', '## 选择结果', '']
for n in ('best','last'):
    ck = summary['checkpoints'][n]
    chosen = ck['calibration_selected_val']
    highest = ck['highest_observed_val']
    lines.append(f"- {n}：校准集预选 `{chosen['id']}`，验证 Top1 {chosen['top1_percent']:.4f}%；验证扫描最高 `{highest['id']}`，Top1 {highest['top1_percent']:.4f}%。")
chosen = summary['calibration_selected_validation_metrics']
lines += [f"- 跨 checkpoint 校准集预选：last / `{chosen['id']}`，验证 Top1 {chosen['top1_percent']:.4f}%，Top5 {chosen['top5_percent']:.4f}%。", '',
          '验证扫描最高是本轮比较中的观测结果；校准集预选结果来自读取验证结果前的独立选择。best.pt 的第 85 轮历史选择依据为训练过程中官方验证集硬路由准确率。', '',
          '## 融合定义与运行核验', '',
          '静态权重：w_HF,c = a_HF,c^γ / (a_HF,c^γ + a_BF,c^γ)，其中 a 为保存的内部校准集每类准确率；γ=0 为等权。γ 扫描 0、0.25、0.5、1、2、4、8。',
          '动态融合使用最大概率、概率间隔、归一化熵作为样本可靠度，β 扫描 0.5、1、2，结合 γ=0、1 先验。每种方式比较原始温度和校准温度。',
          '推理使用 torch.compile/Inductor。编译包含身体/手部编码、池化、拼接、双分类头、raw-XY 输入块及融合；面部 LocalSTBlock 第 1–3 块沿用既有 eager 边界。新融合计算 fullgraph 编译。',
          '9 项融合数值测试和 2 项模型分派/强制编译与随机初始化校验测试均通过。两份 checkpoint SHA256 与缓存一致，模型状态保持一致；112 项的预测、正确数和混淆矩阵已逐项复核。',
          'CUDA/TF32 重复推理中 best 的 HF 与训练记录相差 1 条样本；BF 和 hard 融合准确率与记录一致。', '',
          '完整结果：validation_results.csv；校准结果：calibration_results.csv；权重和温度：fusion_readouts.json；逐样本预测及全部混淆矩阵：best/、last/ 下 predictions.npz；核验：audit.json。', '',
          f'服务器目录：`{out}`', '']
(out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(audit))
