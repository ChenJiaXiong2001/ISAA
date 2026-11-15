"""Build the finished formal-training catalog from read-only scan inventories."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import tarfile


def research_notes(variant):
    if variant.startswith('SkeleT-'):
        return ['skelet_gcn.md']
    if variant.startswith('best-input-joint-selection-v5'):
        return ['input_joint_selection_v5.md']
    if variant == 'best-input-joint-selection':
        return ['input_joint_selection.md']
    mapping = {'best-action-hand-body': 'best_action_hand_body.md',
               'best-hand-body-proximity': 'best_hand_body_proximity.md',
               'best-hand-relation-fusion': 'best_hand_relation_100_cosine.md',
               'body-local-hand-ctr-wide-relative': 'best_weighted_100_cosine.md',
               'body-local-hand-ctr-wide-relative-full': 'full_gcn.md',
               'body-local-hand-ctr-wide-relative-no-pool': 'best_no_pool.md',
               'body-local-hand-ctr-wide-relative-same-channel-late-aug': 'best_weighted_100_same_channel_late_aug.md'}
    if variant.startswith('best-face-pair'):
        return ['face_pair_raw_xy.md' if 'raw-xy' in variant else 'face_pair_fusion.md',
                'recent_research_report_20261108.md']
    return [mapping[variant]] if variant in mapping else []


def build(inputs, destination, plan_path):
    groups = defaultdict(list)
    inventories = [json.loads(p.read_text(encoding='utf-8')) for p in inputs]
    for inventory in inventories:
        for run in inventory['completed']:
            if run['scope'] == 'full_schedule' and not run.get('max_samples'):
                groups[run['run_id']].append(run)
    destination.mkdir(parents=True, exist_ok=True)
    catalog, plan, artifact_rows = [], [], []
    for run_id, copies in groups.items():
        copies.sort(key=lambda r: (r['host'] == 'server', r['path'].startswith('outputs/'),
                                  '/evaluations/' not in r['path']), reverse=True)
        primary = copies[0]
        date = re.search(r'20\d{6}', primary['path'])
        label = re.sub(r'[^a-zA-Z0-9_-]', '_', primary['model_variant'])[:65]
        folder = f"{date.group() if date else 'undated'}_{label}_{run_id[:8]}"
        run_dir = destination / 'runs' / folder
        run_dir.mkdir(parents=True, exist_ok=True)
        sources = [{k: r[k] for k in ('host', 'path', 'absolute_path')} for r in copies]
        entry = {k: v for k, v in primary.items() if k != 'artifacts'}
        entry.update(folder=folder, sources=sources)
        catalog.append(entry)
        detail = [f"# {primary['model_variant']}", '',
                  f"运行编号：`{run_id}`。计划 {primary['planned_epochs']} 轮，逐轮记录到 {primary['recorded_epochs']} 轮。",
                  f"完成证据：`{primary['completion_evidence']}`；状态：`{primary['status']}`。", '',
                  f"数据划分：`{primary['split']}`；seed：`{primary['seed']}`；窗口：`{primary['window_size']}`；repeat：`{primary['repeat']}`。", '']
        acc = primary['best_logged_val_acc']
        if acc is not None:
            percent = acc * 100 if primary['accuracy_units'] == 'fraction' else acc
            detail += [f"训练日志最佳验证 Top-1：**{percent:.4f}%**，epoch {primary['best_logged_epoch']}。这是训练日志指标，不等同于统一独立评估。", '']
        detail += ['## 原始位置', '']
        for source in sources:
            detail.append(f"- {source['host']}：`{source['absolute_path']}`")
        notes = research_notes(primary['model_variant'])
        if notes:
            detail += ['', '## 相关研究日志', '']
            detail += [f'- [{name}](../../../../docs/{name})' for name in notes]
            detail += ['', '说明文档可能同时讨论短程筛选或失败方案；是否完成以本运行的状态和逐轮记录为准。']
        detail += ['', '## 已归集记录', '',
                   f"[配置、逐轮指标、状态、训练日志及源代码归档](../../records/{folder}/)", '',
                   '服务器上的 `original_run` 链接指向完整原始运行目录，包含权重和逐批日志。',
                   '大文件不重复复制；完整文件清单见 [artifacts.csv](../../artifacts.csv)。', '']
        (run_dir/'README.md').write_text('\n'.join(detail), encoding='utf-8')
        # Pick the genuine run folder first; add records from other verified
        # finished copies as evidence without overwriting same-name files.
        chosen = set()
        for index, run in enumerate(copies):
            for artifact in run['artifacts']:
                artifact_rows.append(dict(run_id=run_id, folder=folder, host=run['host'],
                                          **artifact))
                relative = Path(artifact['path']).relative_to(run['path'])
                if len(relative.parts) != 1:
                    continue
                suffix = relative.suffix.lower()
                if suffix not in ('.json', '.csv', '.log', '.txt', '.md', '.zip', '.png'):
                    continue
                if artifact['size'] > 32*1024*1024:
                    continue
                name = str(relative)
                target = f"{folder}/{name}" if name not in chosen else f"{folder}/evidence_{run['host']}_{index}/{name}"
                chosen.add(name)
                plan.append(dict(host=run['host'], source=artifact['path'], target=target,
                                 size=artifact['size'], mtime_ns=artifact['mtime_ns']))
    catalog.sort(key=lambda r: (r['updated_at'] or '', r['path']))
    (destination/'runs.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = ['run_id', 'folder', 'model_variant', 'planned_epochs', 'recorded_epochs',
              'status', 'split', 'seed', 'window_size', 'repeat', 'best_logged_val_acc',
              'best_logged_epoch', 'accuracy_units', 'host', 'path', 'epochs_sha256']
    with (destination/'runs.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(catalog)
    with (destination/'artifacts.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, ['run_id','folder','host','path','size','kind','mtime_ns'])
        writer.writeheader(); writer.writerows(artifact_rows)
    counts = Counter(r['planned_epochs'] for r in catalog)
    lines = ['# 已完成正式训练资料', '',
             f"已按完整逐轮记录核验并去重，收录 **{len(catalog)} 组**正式训练：" + '、'.join(f'{n} 轮 {c} 组' for n,c in sorted(counts.items())) + '。', '',
             '只收录计划至少 24 轮、未限制训练样本数、完成全部计划轮数的正式训练。',
             '24 轮的两组 SkeleT 实验每轮重复数据 5 次。短程筛选、smoke/dry-run、失败、中断、仍在运行及完成证据不足的记录不在本目录。',
             '原目录和全部实验均保留；本目录归集完成训练的记录，服务器通过原始目录链接访问完整权重和大日志。', '',
             '- [训练总表 CSV](runs.csv) / [结构化目录 JSON](runs.json)',
             '- [全部原始文件位置及大小](artifacts.csv)',
             '- [训练权重的位置、大小与 SHA256](checkpoints.csv)',
             '- [已归集的数据与日志](records/)', '',
             '每组运行的详情页同时关联已有研究说明。主表按结束时间排列，保留不同 seed 和历史模型各自的记录。', '',
             '表中指标来自训练 `epochs.csv`，用于定位历史记录，不能跨协议直接排名。',
             'canonical pooled baseline 的统一评估为 **94.6400%**，其训练日志值为 **94.6461%**；依据见 [canonical 配置](../../configs/canonical_baseline.json)。',
             'SkeleT-CTRGCN 两组记录来自 SkeleT-GCN 代码适配，不能标作原始 CTR-GCN。', '',
             '| 实验 / 模型 | 轮数 | seed | 日志最佳 Top-1 | 数据与日志 |',
             '|---|---:|---:|---:|---|']
    for r in catalog:
        acc = r['best_logged_val_acc']
        value = '未提取' if acc is None else f"{acc*(100 if r['accuracy_units']=='fraction' else 1):.4f}%"
        lines.append(f"| {r['model_variant']} | {r['recorded_epochs']}/{r['planned_epochs']} | {r['seed']} | {value} | [查看](runs/{r['folder']}/README.md) |")
    (destination/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(runs=len(catalog), selected_records=len(plan), record_bytes=sum(x['size'] for x in plan))))


def pack(root, host, plan_path, output):
    records = json.loads(plan_path.read_text(encoding='utf-8'))
    root = root.resolve()
    with tarfile.open(output, 'w:gz') as bundle:
        for item in records:
            if item['host'] != host:
                continue
            path = (root/item['source']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Source is outside project')
            stat = path.stat()
            if stat.st_size != item['size'] or stat.st_mtime_ns != item['mtime_ns']:
                raise RuntimeError('Record changed since scan: '+item['source'])
            bundle.add(path, arcname=item['target'], recursive=False)


def unpack(bundle_path, destination):
    destination = destination.resolve()
    manifest = []
    with tarfile.open(bundle_path, 'r:gz') as bundle:
        for entry in bundle.getmembers():
            path = (destination/entry.name).resolve()
            if not entry.isfile() or not path.is_relative_to(destination):
                raise ValueError('Unsafe archive member')
            data = bundle.extractfile(entry).read()
            if path.exists() and path.read_bytes() != data:
                raise RuntimeError('Refusing to replace a different archived record: '+entry.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            manifest.append(dict(path=entry.name, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('build');p.add_argument('--input', type=Path, nargs='+', required=True)
    p.add_argument('--destination', type=Path, required=True);p.add_argument('--plan', type=Path, required=True)
    p = sub.add_parser('pack');p.add_argument('--root', type=Path, required=True)
    p.add_argument('--host', required=True);p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'build':build(args.input, args.destination, args.plan)
    else:pack(args.root, args.host, args.plan, args.output)


if __name__ == '__main__':
    main()
