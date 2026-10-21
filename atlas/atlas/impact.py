"""H1：静态测试影响分析 —— 从被改符号反推「哪些测试会受影响」。

## 形式化

给定 gold patch 改动的符号集合 `S`，影响集是**反向 k-hop 可达闭包**：

```
Impact_k(S) = { t ∈ Tests :  ∃ 路径 t →(calls)^k → s,  s ∈ S }
```

方向语义来自官方图的边定义（`source` 调用 `target`），所以
「改了 s 会影响 t」等价于「t 在调用链上游」，即沿 `in_adj` 反向走。

## 这个方法的硬边界（必须写进论文，而不是藏起来）

**测试节点零出边 → 影响集恒为空。** 反向可达只沿 `in_adj` 展开，而
零出边的测试节点在 `in_adj` 里没有任何反向邻居可走。

实测（129 任务）：

| 仓库 | 测试节点零出边占比 |
|---|---|
| fastapi | **81.0%** |
| rich / requests / httpx | ≤ 11.1% |

fastapi 的测试通过 `TestClient(app)` **间接**调用被测代码，官方图只
捕捉 AST 直接 `calls`，因此这条边根本不存在。容器实跑印证了这一点：
`fastapi_11194` 在 `syntax_break` 下全量套件抓到 **193** 个新失败，
纯静态选取只抓到 **29** 个——漏杀 164 个，全部落在这个缺口上。

所以本模块提供 `test_node_isolation_ratio()` 作为**适用性诊断量**：
调用方应当先看这个比率，再决定是否相信 `Impact_k` 的输出。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

from .graphio import RepositoryGraph, is_test_node, is_test_symbol, node_to_file_suffix

__all__ = ["ImpactResult", "reverse_khop", "impacted_tests",
           "impacted_test_files", "changed_files_to_nodes",
           "diagnose_applicability"]


# --------------------------------------------------------------------------
# 图遍历
# --------------------------------------------------------------------------
def reverse_khop(g: RepositoryGraph, seeds: Iterable[str],
                 k: Optional[int] = None) -> Set[str]:
    """反向 k-hop 可达闭包（k=None 表示不限跳数）。

    沿 `in_adj` 走：被调用方 → 调用方。返回 seeds 自身 + 所有可达点。
    """
    seen: Set[str] = set(seeds)
    frontier = set(seeds)
    hops = 0
    while frontier and (k is None or hops < k):
        nxt: Set[str] = set()
        for n in frontier:
            nxt.update(g.in_adj.get(n) or ())
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
        hops += 1
    return seen


def changed_files_to_nodes(g: RepositoryGraph,
                           changed: Sequence[str]) -> Dict[str, List[str]]:
    """改动文件 → 该文件对应的图节点。

    官方节点名只带**后缀**路径，无法唯一还原全路径（仓库里存在同名符号
    出现在多个包下的情况），所以这里按「文件路径是节点后缀的前缀」匹配，
    并把**多候选**显式记进 `ambiguous` —— 不猜。

    返回 `matched` 与 `ambiguous` 两个字典：
      matched[f]   -> 命中该文件的节点（空列表表示没命中）
      ambiguous[f] -> {节点后缀: [节点, ...]}，len > 1 说明该文件无法唯一定位
    """
    by_suffix: Dict[str, List[str]] = defaultdict(list)
    for nid in g.nodes:
        suf = node_to_file_suffix(nid)
        if suf:
            by_suffix[suf].append(nid)

    matched: Dict[str, List[str]] = {}
    ambiguous: Dict[str, Dict[str, List[str]]] = {}
    for path in changed:
        hits: List[str] = []
        amb: Dict[str, List[str]] = {}
        for suf, nids in by_suffix.items():
            if path == suf or path.endswith("/" + suf):
                hits.extend(nids)
                amb[suf] = nids
        matched[path] = sorted(hits)
        if len(amb) > 1:
            ambiguous[path] = amb
    return matched, ambiguous


# --------------------------------------------------------------------------
# 影响集
# --------------------------------------------------------------------------
@dataclass
class ImpactResult:
    """一次影响分析的完整输出。"""

    instance_id: str
    graph_key: str
    seeds: List[str]                       # 改动符号（种子）
    by_hop: Dict[int, List[str]]           # k -> 该跳新增的测试节点
    test_files: Dict[int, List[str]]       # k -> 测试文件后缀（去重、排序）
    matched_files: Dict[str, List[str]]    # 改动文件 -> 命中节点
    ambiguous_files: Dict[str, Dict[str, List[str]]]
    isolation_ratio: float                 # 适用性诊断量
    unresolved_files: List[str]            # 改了但图里找不到对应节点的文件

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    @property
    def n_tests_total(self) -> int:
        return len({t for v in self.by_hop.values() for t in v})

    def coverage_ratio(self, k: int) -> float:
        """k 跳影响集占该图全部测试节点的比例。"""
        allt = len({t for v in self.by_hop.values() for t in v})
        return 0.0 if not allt else round(len(self.by_hop.get(k, [])) / allt, 4)

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "graph_key": self.graph_key,
            "n_seeds": self.n_seeds,
            "isolation_ratio": self.isolation_ratio,
            "impacted_tests_1hop": len(self.by_hop.get(1, [])),
            "impacted_tests_2hop": len(self.by_hop.get(2, [])),
            "impacted_tests_3hop": len(self.by_hop.get(3, [])),
            "impacted_test_files_3hop": len(self.test_files.get(3, [])),
            "unresolved_files": self.unresolved_files,
            "ambiguous_files": sorted(self.ambiguous_files),
        }


def impacted_tests(g: RepositoryGraph, seeds: Iterable[str],
                   max_hop: int = 3) -> Dict[int, List[str]]:
    """逐跳记录「本跳新增」的测试节点。

    注意 `by_hop[k]` 是**增量**（第 k 跳新到达的），不是累计闭包 ——
    累计值 = 并集。这一点在写论文表格时最容易搞错，故显式分开展示。
    """
    seeds = set(seeds)
    by_hop: Dict[int, List[str]] = {}
    seen: Set[str] = set()
    frontier = set(seeds)
    for hop in range(1, max_hop + 1):
        nxt: Set[str] = set()
        for n in frontier:
            nxt.update(g.in_adj.get(n) or ())
        nxt -= seen
        nxt -= seeds
        if not nxt:
            break
        tests = sorted(t for t in nxt
                       if is_test_node(t) or is_test_symbol(t))
        by_hop[hop] = tests
        seen |= nxt
        frontier = nxt
    return by_hop


def impacted_test_files(g: RepositoryGraph, seeds: Iterable[str],
                        max_hop: int = 3) -> Dict[int, List[str]]:
    """影响集 → 测试文件后缀列表（逐跳增量，去重排序）。"""
    by_hop = impacted_tests(g, seeds, max_hop)
    out: Dict[int, List[str]] = {}
    for hop, nodes in by_hop.items():
        files = {node_to_file_suffix(n) for n in nodes}
        out[hop] = sorted(f for f in files if f)
    return out


# --------------------------------------------------------------------------
# 顶层入口
# --------------------------------------------------------------------------
def impact_of_files(g: RepositoryGraph, instance_id: str,
                    changed: Sequence[str], max_hop: int = 3) -> ImpactResult:
    """给定 gold patch 改动文件，算出 1..max_hop 的测试影响集。"""
    matched, ambiguous = changed_files_to_nodes(g, changed)
    seeds = sorted({n for v in matched.values() for n in v})
    unresolved = [f for f, v in matched.items() if not v]
    return ImpactResult(
        instance_id=instance_id,
        graph_key=g.key,
        seeds=seeds,
        by_hop=impacted_tests(g, seeds, max_hop),
        test_files=impacted_test_files(g, seeds, max_hop),
        matched_files=matched,
        ambiguous_files=ambiguous,
        isolation_ratio=g.isolation_ratio(),
        unresolved_files=sorted(unresolved),
    )


def diagnose_applicability(g: RepositoryGraph,
                           ratio_threshold: float = 0.30) -> Dict[str, object]:
    """给出这张图上「静态影响分析能不能用」的判定。

    阈值 0.30 是从实测分布定的：fastapi 81.0% 明确不可用，
    其余 ≤ 11.1% 可用，中间地带留给人工判断而非硬阈值。
    """
    iso = g.isolation_ratio()
    all_tests = g.test_nodes()
    return {
        "graph_key": g.key,
        "n_nodes": len(g),
        "n_edges": g.edge_count(),
        "n_test_nodes": len(all_tests),
        "test_node_isolation_ratio": iso,
        "orphan_node_ratio": g.orphan_ratio(),
        "max_wcc_ratio": g.wcc()[0],
        "h1_applicable": iso < ratio_threshold,
        "reason": (
            f"测试节点零出边占比 {iso:.1%}"
            + ("，超过阈值，反向可达闭包对多数测试恒为空" if iso >= ratio_threshold
               else "，低于阈值，反向可达闭包有效")
        ),
    }
