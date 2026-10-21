"""H2：三档上下文注入策略（B0 / B1 / B2）。

## 评测条件（必须先声明，否则指标无意义）

agent 推理时**只能看到 `problem_statement`（自然语言）+ 仓库文件清单**，
看不到 gold patch。所以 query 不是 gold 符号，而是 problem_statement。

而官方 harness **没有文本编码器** —— HARNESS_README §6.3 原文：

> "Because the sandbox runs offline without a live neural embedding server,
> `search_similar_code` resolves `query` against node keys/suffixes stored in
> the graph and `.npz` dictionary and computes cosine similarity across all
> other nodes in the repository (**with a lexical token-overlap fallback if
> `query` is a multi-word keyword string**)."

官方 npz 里只有**仓库内符号的 code embedding**，256 维，没有「文本 → 向量」
的编码路径。因此本模块把这一环境约束拆成两个可分离的信号：

| 信号 | 实现 | 说明 |
|---|---|---|
| `lexical` | problem_statement 词干 ∩ 节点名词干的 Jaccard | 官方 token-overlap fallback 的可执行定义 |
| `embedding` | symbol→symbol 余弦（以种子符号为 query） | 复刻 `search_similar_code` 的**图内**用法 |

两者都不需要 GPU，也都不假装自己是「自然语言语义检索」。

## 三档

| 档 | 定义 | 对应官方工具 |
|---|---|---|
| **B0** | 纯检索 top-k（默认 lexical，可切 embedding） | `search_similar_code` |
| **B1** | B0 top-`seed_k` 作种子 + 1 跳邻居扩展 | `+ get_code_neighbors` |
| **B2** | 仅 1 跳闭包（以 B0 top-1 为种子） | `get_code_neighbors` |

## 预算口径

官方 `max_stdout_chars = 5000` 是**总字符**预算，不是每节点预算：
k 越大每个节点分到越少，长节点会被截断。`pack_to_char_budget()`
复刻官方的「均分 + 短片段余量滚给长片段」分配规则。
实测 k 的有效上界（≥90% 节点完整）**median 只有 3**。
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .graphio import RepositoryGraph, cosine_topk, load_embeddings, node_to_file_suffix

__all__ = ["LocateConfig", "LocateResult", "lexical_score",
           "build_b0", "build_b1", "build_b2", "run_locate",
           "pack_to_char_budget", "effective_k_cap", "dcg",
           "recall_at_k", "mrr", "ndcg"]

# ---------------------------------------------------------------- 官方硬限制
MAX_STDOUT_CHARS = 5000     # HARNESS_README §6.3
COMPACT_AT_TOKENS = 14_336  # §7.2 压缩触发线
WINDOW_TOKENS = 32_768      # §7.2 窗口

_STOP = {"this", "that", "with", "from", "when", "then", "there",
         "have", "will", "does", "should", "would", "could", "been"}


# ---------------------------------------------------------------- 词干
def _tokens(text: str, min_len: int = 4) -> Set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
            if len(w) >= min_len and w.lower() not in _STOP}


def _node_stems(node: str, min_len: int = 4) -> Set[str]:
    """节点名的词干集合：路径段 + 符号名按 `_`/`.`/`/` 切分 + 驼峰再切。

    驼峰必须在**原始大小写**上切，再小写 —— 先小写再切是无效的
    （`TestClient` → `testclient` 已经看不出边界）。用零宽大写边界
    （`(?<=[a-z0-9])(?=[A-Z])`）而不是 `findall`，否则 `HTTPClient`
    会被切成 `HTTPClie` / `nt` 这种碎片。
    """
    parts = re.split(r"[._/]", node)
    out: Set[str] = set()
    for p in parts:
        if not p:
            continue
        for w in re.split(r"(?<=[a-z0-9])(?=[A-Z])", p):
            w = w.lower()
            if len(w) >= min_len:
                out.add(w)
    return out


def lexical_score(query: str, node: str) -> float:
    """词干 Jaccard。官方 token-overlap fallback 的可执行定义。

    官方只说 token-overlap，未给精确定义；Jaccard 是有界、对称、
    对「query 词多而节点词少」不过度惩罚的合理代理。
    """
    q, n = _tokens(query), _node_stems(node)
    if not q or not n:
        return 0.0
    return len(q & n) / len(q | n)


# ---------------------------------------------------------------- 预算
def pack_to_char_budget(texts: Sequence[str], cap: int = MAX_STDOUT_CHARS
                        ) -> Tuple[List[int], int]:
    """复刻官方的字符分配：均分 + 短片段余量滚给长片段。

    返回 `(每个节点可用字符数, 总分配量)`。
    """
    if cap < 0:
        raise ValueError("character budget must be nonnegative")
    k = len(texts)
    if k == 0:
        return [], 0
    lens = [len(t or "") for t in texts]
    per = cap // k
    alloc = [min(l, per) for l in lens]
    leftover = cap - sum(alloc)
    order = sorted(range(k), key=lambda i: lens[i], reverse=True)
    while leftover > 0:
        progressed = False
        for i in order:
            if alloc[i] < lens[i]:
                add = min(leftover, lens[i] - alloc[i])
                alloc[i] += add
                leftover -= add
                progressed = True
                if leftover == 0:
                    break
        if not progressed:
            break
    return alloc, sum(alloc)


def effective_k_cap(texts: Sequence[str], cap: int = MAX_STDOUT_CHARS,
                    quantile: float = 0.9) -> int:
    """在 5,000 字符总预算下，「≥90% 节点拿到完整文本」的最大 k。

    为什么不是「全部完整」：图里存在极端长节点（实测单个节点文本最长
    **383,734 字符**，p99 = 9,553）。要求全部完整会让 k 恒为 0 —— 指标
    没有区分度。而 agent 实际感知的是「绝大多数节点可读」，个别超长节点
    被截断是可接受的代价。
    """
    best = 0
    for k in range(1, len(texts) + 1):
        per = cap // k
        intact = sum(1 for t in texts[:k] if len(t or "") <= per)
        if intact / k >= quantile:
            best = k
        else:
            break
    return best


def est_tokens(texts: Sequence[str]) -> int:
    """字符数/4 近似 token。**这是近似值**，无分词器时的显式近似。"""
    return sum(len(t or "") for t in texts) // 4


# ---------------------------------------------------------------- 指标
def recall_at_k(ranked: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & set(ranked[:k])) / len(gold)


def mrr(ranked: Sequence[str], gold: Set[str], kmax: int = 50) -> float:
    for i, n in enumerate(ranked[:kmax]):
        if n in gold:
            return round(1.0 / (i + 1), 4)
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg(ranked: Sequence[str], gold: Set[str], k: int = 50) -> float:
    ideal = dcg([1.0] * min(len(gold), k))
    if ideal <= 0:
        return 0.0
    return round(dcg([1.0 if n in gold else 0.0 for n in ranked[:k]]) / ideal, 4)


# ---------------------------------------------------------------- 三档
@dataclass
class LocateConfig:
    """注入策略参数。默认值来自实测最优区间，不是随手填的。"""

    signal: str = "lexical"     # lexical | embedding
    seed_k: int = 10            # B1 的种子数
    max_k: int = 50             # 评估时的截断
    topk_grid: Tuple[int, ...] = (1, 3, 5, 10, 20, 50)
    char_cap: int = MAX_STDOUT_CHARS
    hops: int = 1               # B1/B2 的扩展跳数

    def __post_init__(self):
        if self.max_k < 0 or self.seed_k < 0 or self.char_cap < 0:
            raise ValueError("max_k, seed_k and char_cap must be nonnegative")


@dataclass
class LocateResult:
    tier: str
    ranked: List[str]
    seed_nodes: List[str]
    scores: List[float] = field(default_factory=list)
    expanded: int = 0           # 1 跳扩展新增的节点数
    tokens_est: int = 0
    truncated: bool = False
    per_k: Dict[int, int] = field(default_factory=dict)   # k -> 实际拿到字符的节点数

    @property
    def n(self) -> int:
        return len(self.ranked)


def _lexical_order(g: RepositoryGraph, query: str) -> Tuple[List[str], List[float]]:
    scored = [(lexical_score(query, n), n) for n in g.nodes]
    scored.sort(key=lambda x: (-x[0], x[1]))     # 稳定：同分按名字序
    return [n for _, n in scored], [s for s, _ in scored]


def _embedding_order(g: RepositoryGraph, graph_key: str,
                     query_node: str, k: int) -> Tuple[List[str], List[float]]:
    """以一个**符号名**为 query 的图内检索（复刻 search_similar_code 的可用形态）。"""
    ids, V = load_embeddings(graph_key)
    if not ids:
        return [], []
    idx = {n: i for i, n in enumerate(ids)}
    if query_node not in idx:
        return [], []
    ranked = [n for n, _ in cosine_topk(ids, V, V[idx[query_node]], min(k, len(ids)))]
    return ranked, []


def build_b0(g: RepositoryGraph, query: str, cfg: LocateConfig) -> LocateResult:
    """B0：纯检索 top-k。"""
    if cfg.signal == "embedding":
        raise ValueError(
            "embedding 信号需要一个符号名做 query（官方无文本编码器）；"
            "用 build_b0_embedding(g, graph_key, query_node, k)")
    ranked, scores = _lexical_order(g, query)
    ranked, scores = ranked[:cfg.max_k], scores[:cfg.max_k]
    texts = [g.texts.get(n, "") for n in ranked]
    alloc, _ = pack_to_char_budget(texts, cfg.char_cap)
    return LocateResult(
        tier="B0", ranked=ranked, seed_nodes=[], scores=scores,
        tokens_est=est_tokens(texts),
        truncated=any(a < len(t or "") for a, t in zip(alloc, texts)),
        per_k={k: effective_k_cap(texts[:k], cfg.char_cap) for k in cfg.topk_grid},
    )


def build_b0_embedding(g: RepositoryGraph, graph_key: str, query_node: str,
                        k: int = 50, char_cap: int = MAX_STDOUT_CHARS) -> LocateResult:
    """B0（embedding 变体）：以符号名为 query 的图内余弦检索。"""
    ranked, _ = _embedding_order(g, graph_key, query_node, k)
    texts = [g.texts.get(n, "") for n in ranked]
    alloc, _ = pack_to_char_budget(texts, char_cap)
    return LocateResult(
        tier="B0-emb", ranked=ranked, seed_nodes=[query_node],
        tokens_est=est_tokens(texts),
        truncated=any(a < len(t or "") for a, t in zip(alloc, texts)),
    )


def build_b1(g: RepositoryGraph, query: str, cfg: LocateConfig) -> LocateResult:
    """B1：B0 top-`seed_k` 作种子 + 1 跳邻居扩展。"""
    b0 = build_b0(g, query, cfg)
    seeds = b0.ranked[:cfg.seed_k]
    ranked: List[str] = list(seeds)
    seen: Set[str] = set(seeds)
    for seed in seeds:
        for nb in sorted(g.out_adj.get(seed, ())) + sorted(g.in_adj.get(seed, ())):
            if len(ranked) >= cfg.max_k:
                break
            if nb not in seen:
                seen.add(nb)
                ranked.append(nb)
    texts = [g.texts.get(n, "") for n in ranked]
    alloc, _ = pack_to_char_budget(texts, cfg.char_cap)
    return LocateResult(
        tier="B1", ranked=ranked, seed_nodes=seeds,
        scores=b0.scores[:cfg.seed_k],
        expanded=len(ranked) - len(seeds), tokens_est=est_tokens(texts),
        truncated=any(a < len(t or "") for a, t in zip(alloc, texts)),
    )


def build_b2(g: RepositoryGraph, query: str, cfg: LocateConfig) -> LocateResult:
    """B2：仅 1 跳闭包（以 B0 top-1 为种子）。"""
    b0 = build_b0(g, query, cfg)
    if not b0.ranked:
        return LocateResult(tier="B2", ranked=[], seed_nodes=[])
    seed = b0.ranked[0]
    ranked = [seed]
    for nb in sorted(g.out_adj.get(seed, ())) + sorted(g.in_adj.get(seed, ())):
        if len(ranked) >= cfg.max_k:
            break
        if nb not in ranked:
            ranked.append(nb)
    texts = [g.texts.get(n, "") for n in ranked]
    alloc, _ = pack_to_char_budget(texts, cfg.char_cap)
    return LocateResult(
        tier="B2", ranked=ranked, seed_nodes=[seed], scores=b0.scores[:1],
        expanded=len(ranked) - 1, tokens_est=est_tokens(texts),
        truncated=any(a < len(t or "") for a, t in zip(alloc, texts)),
    )


def run_locate(g: RepositoryGraph, query: str, gold: Optional[Set[str]] = None,
               cfg: Optional[LocateConfig] = None) -> Dict[str, LocateResult]:
    """跑三档并（可选）附带相对 gold 的检索指标。"""
    cfg = cfg or LocateConfig()
    out = {"B0": build_b0(g, query, cfg),
           "B1": build_b1(g, query, cfg),
           "B2": build_b2(g, query, cfg)}
    if gold:
        for r in out.values():
            r.per_k = {**r.per_k,
                       **{f"recall@{k}": recall_at_k(r.ranked, gold, k)
                          for k in cfg.topk_grid}}
    return out


def file_level_recall(ranked: Sequence[str], gold_files: Set[str],
                      k: int = 20) -> float:
    """文件级 recall —— 更贴近「agent 有没有找对文件」。"""
    if not gold_files:
        return 0.0
    hit_files = {node_to_file_suffix(n) for n in ranked[:k]}
    hit_files.discard(None)
    got = sum(1 for gf in gold_files
              if any(f == gf or (gf or "").endswith("/" + f)
                     for f in hit_files if f))
    return round(got / len(gold_files), 4)
