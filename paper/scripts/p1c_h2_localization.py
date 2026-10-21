#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H2 offline localization diagnostics: filename-stem proxy and graph expansion.

B0 ranks nodes by Jaccard(problem-statement stems, filename stems).
B1 interleaves the top 10 B0 seeds with their undirected one-hop calls neighbors.
B2 contains the top B0 seed and its undirected one-hop calls neighbors.
R1 is a fixed-seed random ranking; R2 ranks by filename-stem overlap count.
R3 places every coarse-gold node first, then appends the remaining B0 ranking.
The legacy key R3_oracle_1hop is retained for compatibility, but R3 performs
no graph traversal. Gold is the graph-node set matched to gold-changed files.

No query embedding encoder is available. B0 is a lexical proxy; no lower-bound
relationship to embedding retrieval, agent delivery, or repair success is claimed.
The norm sensitivity calculation reweights lexical scores by stored vector norms;
it does not measure normalization's effect on actual embedding-query retrieval.
Token costs use characters/4, without serialization, truncation, or a tokenizer.

Ties preserve NPZ member order; gold and adjacency sets use lexical node-ID order.
Outputs default to a fresh results/final_h2_localization_<run>.json. An explicit
--output must not exist. Historical research JSON is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

from run_support import fresh_output, prepare_output, run_metadata, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "paper", "scripts"))

from p1b_h1_feasibility import (  # noqa: E402
    GRAPHS, TASKS, diff_files, is_test_file, node_to_file,
)

EMB = os.path.join(ROOT, "data", "embeddings")
OUT = os.path.join(ROOT, "results", "h2_localization.json")
TOPK = (1, 3, 5, 10, 20, 50)

RANDOM_SEED = 20261016


# ---------------------------------------------------------------- gold 符号
def changed_lines_from_patch(patch_text):
    """与 p1_map_targets 同源：返回 {file: [新文件行号]}。"""
    files, cur, nl = {}, None, 0
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            p = line[4:].strip()
            p = p[2:] if p.startswith("b/") else p
            cur = None if p == "/dev/null" else p
            if cur:
                files.setdefault(cur, set())
        elif line.startswith("@@"):
            try:
                seg = line.split("+", 1)[1].split("@@", 1)[0].strip()
                nl = int(seg.split(",")[0])
            except (IndexError, ValueError):
                nl = 0
        elif cur and line.startswith("+") and not line.startswith("+++"):
            files[cur].add(nl)
            nl += 1
        elif cur and line.startswith("-"):
            pass
        elif cur and line.startswith("\\"):
            pass
        elif cur:
            nl += 1
    return {k: sorted(v) for k, v in files.items()}


def enclosing_symbols(src, lines):
    """行号 → 最内层函数/类名（简化版映射器，只需符号名）。"""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            spans.append((node.name, start, node.end_lineno))
    hits = []
    for ln in lines:
        cands = [s for s in spans if s[1] <= ln <= s[2]]
        if cands:
            cands.sort(key=lambda s: s[2] - s[1])
            hits.append(cands[0][0])
    return hits


# ---------------------------------------------------------------- 指标
def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def metrics(ranked, gold, kmax=50):
    """ranked: 有序节点 id 列表；gold: set。"""
    g = set(gold)
    out = {}
    if not g:
        return None
    first = next((i for i, n in enumerate(ranked) if n in g), None)
    out["recall"] = {}
    for k in TOPK:
        out["recall"][str(k)] = round(
            len(g & set(ranked[:k])) / len(g), 4)
    out["mrr"] = round(1.0 / (first + 1), 4) if first is not None and first < kmax else 0.0
    out["ndcg@50"] = round(
        dcg([1.0 if n in g else 0.0 for n in ranked[:50]]) /
        dcg([1.0] * min(len(g), 50)), 4)
    return out


def tok_cost(texts):
    """字符数/4 近似 token（显式标注为近似值）。"""
    return sum(len(t) for t in texts) // 4


# ---------------------------------------------------------------- 主流程
def rank_scores(scores):
    """Descending scores, with NPZ archive row order as the tie breaker."""
    return np.argsort(-scores, kind="stable")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        out_path, _ = prepare_output(args.output or fresh_output(ROOT, "h2_localization"), ROOT)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    with open(TASKS, encoding="utf-8") as fh:
        tasks = [json.loads(line) for line in fh if line.strip()]
    rng = random.Random(RANDOM_SEED)

    per = []
    skipped = {"no_graph": 0, "no_gold": 0, "syntax": 0, "no_match": 0}

    for t in tasks:
        key = f"{t['repo'].split('/')[-1]}_{t['base_commit']}"
        gpath, epath = os.path.join(GRAPHS, key + ".json"), os.path.join(EMB, key + ".npz")
        if not (os.path.exists(gpath) and os.path.exists(epath)):
            skipped["no_graph"] += 1
            continue

        with open(gpath, encoding="utf-8") as fh:
            g = json.load(fh)
        with np.load(epath, allow_pickle=False) as z:
            node_ids = list(z.keys())
            V = np.stack([z[n] for n in node_ids]).astype(np.float64) if node_ids else None
        if not node_ids:
            skipped["no_graph"] += 1
            continue
        id2row = {n: i for i, n in enumerate(node_ids)}

        # --- gold 符号：从 patch diff 抽，需要真实源码 → 借助 nodes[].text 无法覆盖
        #     源码不在图里，故用「改动文件 → 图节点」作为 coarse gold（与 H1 同源），
        #     并额外报告文件级 recall。文件级真值无需解压快照，成本为零。
        changed = diff_files(t["patch"])
        node2file = {n["id"]: node_to_file(n["id"]) for n in g["nodes"]}
        file2nodes = defaultdict(set)
        for nid, f in node2file.items():
            if f:
                file2nodes[f].add(nid)

        gold_nodes = set()
        for c in changed:
            for suf, nids in file2nodes.items():
                if c == suf or c.endswith("/" + suf):
                    gold_nodes |= nids
        if not gold_nodes:
            skipped["no_gold"] += 1
            continue
        gold_files = {node2file[n] for n in gold_nodes if node2file.get(n)}

        # --- query：problem_statement（agent 唯一可见的信息）
        q = t["problem_statement"].lower()
        # 官方没给「任意文本 → 向量」的编码器（embedding 是仓库内符号的 code embedding），
        # 因此这里用**词干集合 Jaccard** 作为文本侧检索信号，并在结果里显式标注
        # 「这是词干代理，不是 embedding 检索的下界，也不是官方工具复现」。
        qstems = {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", q)
                  if len(w) > 3 and w not in {"this", "that", "with", "from", "when"}}


        node_text = {n["id"]: n.get("text", "") for n in g["nodes"]}

        # 邻接（无向，供 1 跳扩展）
        adj = defaultdict(set)
        for e in g.get("edges", []):
            if e.get("type") != "calls":
                continue
            s, tt = e.get("source"), e.get("target")
            if s in id2row and tt in id2row:
                adj[s].add(tt)
                adj[tt].add(s)

        def stem_of(nid):
            f = node2file.get(nid) or ""
            base = os.path.basename(f).replace(".py", "")
            return set(w for w in re.split(r"[_\.]", base.lower()) if len(w) > 3)

        # --- B0 检索信号：文件名词干 Jaccard 代理
        sims = np.array([
            len(qstems & stem_of(n)) / (len(qstems | stem_of(n)) or 1)
            for n in node_ids
        ])
        order = list(rank_scores(sims))
        b0_ranked = [node_ids[i] for i in order]

        # 对照：若不做 L2 归一化，直接用「相似度 × 自身范数」排序，
        # 量化官方 npz 未归一化会造成的排序偏移（研���计划 §0 事实 3 的落地风险）
        order_unnorm = list(rank_scores(sims * np.linalg.norm(V, axis=1)))

        # --- R1 随机同规模
        r1_ranked = list(node_ids)
        rng.shuffle(r1_ranked)

        # --- R2 纯文件名匹配
        r2_ranked = sorted(node_ids, key=lambda n: -len(qstems & stem_of(n)))

        # --- R3 gold-first oracle（不走邻接；保留旧JSON字段兼容）
        r3_ranked = sorted(gold_nodes) + [n for n in b0_ranked if n not in gold_nodes]

        # --- B1: b0 top-k 种子 + 1 跳
        b1_ranked, seen = [], set()
        for i in order[:10]:
            n = node_ids[i]
            if n not in seen:
                seen.add(n)
                b1_ranked.append(n)
            for m in sorted(adj.get(n, ())):
                if m not in seen:
                    seen.add(m)
                    b1_ranked.append(m)

        # --- B2: 仅 1 跳（以 b0 top-1 为种子的闭包）
        seed1 = b0_ranked[0]
        b2_ranked = [seed1] + [m for m in sorted(adj.get(seed1, ()))]

        row = {
            "instance_id": t["instance_id"],
            "repo": t["repo"],
            "n_nodes": len(node_ids),
            "n_gold_nodes": len(gold_nodes),
            "n_gold_files": len(gold_files),
            "B0": metrics(b0_ranked, gold_nodes),
            "B1": metrics(b1_ranked, gold_nodes),
            "B2": metrics(b2_ranked, gold_nodes),
            "R1_random": metrics(r1_ranked, gold_nodes),
            "R2_stem": metrics(r2_ranked, gold_nodes),
            "R3_oracle_1hop": metrics(r3_ranked, gold_nodes),
            "_order_shift_by_norm": round(
                sum(1 for a, b in zip(b0_ranked, [node_ids[i] for i in order_unnorm])
                    if a != b) / len(node_ids), 4),
            "_cost_topk_tokens": {
                str(k): tok_cost([node_text.get(n, "") for n in b0_ranked[:k]])
                for k in TOPK
            },
            "_cost_b2_tokens": tok_cost([node_text.get(n, "") for n in b2_ranked]),
        }
        # 文件级 recall（更贴近「agent 找对文件了吗」）
        for name, rk in (("B0", b0_ranked), ("B1", b1_ranked), ("B2", b2_ranked),
                         ("R1", r1_ranked), ("R2", r2_ranked), ("R3", r3_ranked)):
            files = {node2file[n] for n in rk[:20] if node2file.get(n)}
            hit = sum(1 for f in gold_files
                      if any(f == g or g.endswith("/" + f) for g in files))
            row.setdefault("file_recall@20", {})[name] = round(hit / len(gold_files), 4)
        per.append(row)

    def agg(path):
        vals = []
        for r in per:
            cur = r
            for p in path:
                if cur is None:
                    break
                cur = cur.get(p) if isinstance(cur, dict) else None
            if cur is not None:
                vals.append(cur)
        return vals

    summary = {}
    # metrics 键名 → file_recall 键名（R1 的 key 在 row 里存为 R1）
    FKEY = {"R1_random": "R1", "R2_stem": "R2", "R3_oracle_1hop": "R3",
            "B0": "B0", "B1": "B1", "B2": "B2"}
    for name in ("B0", "B1", "B2", "R1_random", "R2_stem", "R3_oracle_1hop"):
        entry = {}
        for k in TOPK:
            v = agg([name, "recall", str(k)])
            entry[f"recall@{k}"] = round(statistics.mean(v), 4) if v else None
            entry[f"recall@{k}_median"] = round(statistics.median(v), 4) if v else None
        for m in ("mrr", "ndcg@50"):
            v = agg([name, m])
            entry[m] = round(statistics.mean(v), 4) if v else None
        v = agg(["file_recall@20", FKEY[name]])
        entry["file_recall@20"] = round(statistics.mean(v), 4) if v else None
        summary[name] = entry

    shifts = agg(["_order_shift_by_norm"])
    costs = defaultdict(list)
    for r in per:
        for k, v in r["_cost_topk_tokens"].items():
            costs[k].append(v)
    b2c = agg(["_cost_b2_tokens"])

    report = {
        "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "n_tasks_evaluated": len(per),
        "skipped": skipped,
        "run_metadata": run_metadata(TASKS, __file__, [t["instance_id"] for t in tasks],
                                     numpy=np.__version__, random_seed=RANDOM_SEED,
                                     tie_break="stable NPZ member order", set_order="lexical node ID",
                                     workers=1),
        "method_definitions": {
            "B0": "filename-stem Jaccard lexical proxy; ties follow NPZ member order",
            "B1": "top-10 B0 seeds interleaved with sorted undirected calls neighbors",
            "B2": "top-1 B0 seed followed by sorted undirected calls neighbors",
            "R3_oracle_1hop": "legacy key: gold-first oracle ranking, NOT a one-hop traversal",
            "delivery": "not measured; no serialization or truncation is simulated",
        },
        "gold_definition": "gold patch 改动文件对应的图节点集合（文件级 coarse gold）",
        "caveats": [
            "官方 embedding 是仓库内符号的 code embedding，没有任意文本编码器；"
            "B0/B1/B2 的检索信号用「problem_statement 词干 vs 节点文件名词干的 Jaccard」"
            "作为独立词干代理；不是 embedding 检索下界，也不是 search_similar_code 的复现。",
            "gold 是文件级而非符号级；符号级需要解压快照取源码，成本不在本脚本内。",
            "token 数为 字符数/4 的显式近似，未使用真实分词器。",
        ],
        "summary": summary,
        "normalization": {
            "note": "Lexical-score times stored-vector norm sensitivity; not embedding-query normalization",
            "mean_rank_shift_if_unnormalized": round(statistics.mean(shifts), 4)
            if shifts else None,
        },
        "pareto_topk_tokens_mean": {
            k: round(statistics.mean(v)) for k, v in sorted(costs.items(), key=lambda x: int(x[0]))
        },
        "pareto_b2_tokens_mean": round(statistics.mean(b2c)) if b2c else None,
        "per_task": per,
    }
    write_json(out_path, report)

    print("=" * 86)
    print("H2 三档注入消融（离线，零容器）")
    print("=" * 86)
    print(f"评测任务 {len(per)}，跳过 {skipped}")
    print()
    hdr = f"{'方法':<18}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'R@20':>7}{'R@50':>7}{'MRR':>7}{'NDCG':>7}{'fileR@20':>10}"
    print(hdr)

    def f3(x):
        return f"{x:>7.3f}" if isinstance(x, (int, float)) else f"{'n/a':>7}"

    for name in ("R1_random", "R2_stem", "B0", "B1", "B2", "R3_oracle_1hop"):
        s = summary[name]
        print(f"{name:<18}"
              f"{f3(s['recall@1'])}{f3(s['recall@5'])}{f3(s['recall@10'])}"
              f"{f3(s['recall@20'])}{f3(s['recall@50'])}{f3(s['mrr'])}"
              f"{f3(s['ndcg@50'])}{f3(s['file_recall@20'])}")
    print()
    print("帕累托（注入 token 数均值，字符数/4 近似）:")
    print("  top-k 检索 :", report["pareto_topk_tokens_mean"])
    print(f"  B2 1跳闭包 : {report['pareto_b2_tokens_mean']} tokens")
    print()
    print("词干分数范数重加权敏感性:", report["normalization"])
    print()
    print("字符/4成本不包含实际序列化或截断；未测量agent信息送达率。")
    print(f"→ 写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
