"""官方 AST 调用图 / 符号 embedding 的读写层。

## 数据格式（唯一事实来源：HARNESS_README §6.3 + 实测文件）

图：`data/graphs/<repo_short>_<base_commit>.json`

```json
{
  "directed": true, "multigraph": true, "graph": {...},
  "nodes": [{"id": "pkg.mod.Class", "name": "...", "text": "def f(): ..."}],
  "edges": [{"type": "calls", "source": "caller", "target": "callee", "key": 0}]
}
```

边类型**只有 `calls`** 一一种（实测 129 张图全部如此）。方向语义：
`source` 是调用方，`target` 是被调用方。

embedding：`data/embeddings/<repo_short>_<base_commit>.npz`

每个 key 就是一个 node id，value 是 shape `(256,)` 的 float32 向量。

## 三个必须显式处理的坑

1. **key 是 `<repo_short>_<base_commit>`，不是 `instance_id`。**
   同一个仓库的多个任务共享同一张图；用 instance_id 去 join 会 0 命中。
2. **官方 npz 未做 L2 归一化。** 实测单向量 L2 范数 p50 ≈ 3.91
   （std/mean 0.13–0.30），直接点积排序会混入范数偏置。`load_embeddings()`
   默认 `l2_normalize=True`，并可返回未归一化副本用于量化这个偏移。
3. **图节点名只保留路径后缀。** `fastapi.applications.FastAPI` 对应的
   文件路径无法从名字里唯一还原（仓库里同名符号可能出现在多个包下），
   所以本库**不猜全路径**，只提供 `node_to_file_suffix()` 供调用方做
   suffix 匹配，并由调用方显式处理歧义。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = [
    "graph_key_for", "load_graph", "load_embeddings",
    "RepositoryGraph", "node_to_file_suffix", "is_test_node", "is_test_symbol",
    "data_root", "iter_available_keys", "cosine_topk", "EDGE_TYPE_CALLS",
]

# 官方唯一边类型（实测确认）
EDGE_TYPE_CALLS = "calls"


# --------------------------------------------------------------------------
# key 推导
# --------------------------------------------------------------------------
def graph_key_for(task: dict) -> str:
    """从 task 记录推导 `<repo_short>_<base_commit>`。

    `repo` 形如 `fastapi/fastapi`，取最后一段作为 short。
    """
    repo = str(task.get("repo") or "")
    short = repo.rsplit("/", 1)[-1] if "/" in repo else repo
    return f"{short}_{task['base_commit']}"


def data_root(explicit: Optional[str] = None) -> str:
    """默认定位到仓库根的 `data/`。可传 `ATLAS_DATA` 覆盖。"""
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get("ATLAS_DATA")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))          # atlas/atlas/
    for cand in (os.path.join(here, "..", "..", "data"),
                 os.path.join(here, "..", "data"),
                 os.path.join(here, "data")):
        c = os.path.abspath(cand)
        if os.path.isdir(os.path.join(c, "graphs")):
            return c
    return os.path.abspath(os.path.join(here, "..", "..", "data"))


def _paths(key: str, root: Optional[str] = None) -> Tuple[str, str]:
    r = data_root(root)
    return (os.path.join(r, "graphs", key + ".json"),
            os.path.join(r, "embeddings", key + ".npz"))


def iter_available_keys(root: Optional[str] = None) -> List[str]:
    """列出本地同时具备图与 embedding 的所有 key。"""
    r = data_root(root)
    gdir = os.path.join(r, "graphs")
    if not os.path.isdir(gdir):
        return []
    keys = []
    for fn in os.listdir(gdir):
        if fn.endswith(".json"):
            k = fn[:-5]
            if os.path.isfile(os.path.join(r, "embeddings", k + ".npz")):
                keys.append(k)
    return sorted(keys)


# --------------------------------------------------------------------------
# 名字工具
# --------------------------------------------------------------------------
def node_to_file_suffix(node_name: str) -> Optional[str]:
    """`a.b.c.Symbol` -> `a/b/c.py`（仅后缀路径，前缀未知）。

    注意这是**后缀**而非全路径：`fastapi.applications.FastAPI` 会得到
    `fastapi/applications.py`，在真实的 `fastapi/applications/main.py`
    布局下并不精确。调用方必须做 suffix 匹配并处理多候选歧义。
    """
    head, _, _sym = node_name.rpartition(".")
    if not head:
        return None
    return head.replace(".", "/") + ".py"


def is_test_node(node_name: str, file_suffix: Optional[str] = None) -> bool:
    """pytest 收集语义下的「测试节点」判定。

    对齐 HARNESS_README §8.2 步骤 3 的 protected path 规则：
    `test_*.py`、`*_test.py`，或 `tests/` `test/` `testing/` 下的 `.py`。
    """
    path = file_suffix
    if path is None:
        path = node_to_file_suffix(node_name) or ""
    base = path.rsplit("/", 1)[-1]
    if base.startswith("test_") or base.endswith("_test.py"):
        return True
    segs = path.split("/")[:-1]
    return any(s in ("tests", "test", "testing") for s in segs)


def is_test_symbol(node_name: str) -> bool:
    """符号名本身像 pytest 用例（test* 前缀）。"""
    sym = node_name.rsplit(".", 1)[-1]
    return sym.startswith("test") and len(sym) > 4


# --------------------------------------------------------------------------
# 容器
# --------------------------------------------------------------------------
@dataclass
class RepositoryGraph:
    """一张图的物化视图。

    字段：
      key         `<repo_short>_<base_commit>`
      nodes       全部 node id（保序，与 embedding 行序一致）
      texts       node id -> 源码文本
      out_adj     调用方 -> 被调用方集合（边方向即语义方向）
      in_adj      被调用方 -> 调用方集合（反向，H1 影响分析用这个）
      nodes_meta  node id -> 原始 dict
    """

    key: str
    nodes: List[str] = field(default_factory=list)
    texts: Dict[str, str] = field(default_factory=dict)
    out_adj: Dict[str, Set[str]] = field(default_factory=dict)
    in_adj: Dict[str, Set[str]] = field(default_factory=dict)
    nodes_meta: Dict[str, dict] = field(default_factory=dict)

    # ---- 基本查询 ----
    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, node: str) -> bool:
        return node in self.texts

    def file_suffixes(self) -> Dict[str, Optional[str]]:
        return {n: node_to_file_suffix(n) for n in self.nodes}

    def test_nodes(self) -> List[str]:
        return [n for n in self.nodes
                if is_test_node(n) or is_test_symbol(n)]

    def isolation_ratio(self) -> float:
        """**测试节点零出边占比** —— H1 的适用性诊断量。

        零出边（既不调用别人也不被调用）的测试节点，反向可达闭包永远是
        它自己，任何静态影响分析都只能返回空集。
        实测 fastapi 81.0%，其他仓库 ≤ 11.1%（测试通过 `TestClient(app)`
        间接调用，官方图只捕捉 AST 直接 `calls`）。
        """
        tn = self.test_nodes()
        if not tn:
            return 0.0
        iso = sum(1 for n in tn
                  if not self.out_adj.get(n) and not self.in_adj.get(n))
        return round(iso / len(tn), 4)

    def edge_count(self) -> int:
        return sum(len(v) for v in self.out_adj.values())

    def wcc(self) -> Tuple[float, int]:
        """返回 (最大弱连通分量占比, 该分量节点数)。"""
        if not self.nodes:
            return 0.0, 0
        und: Dict[str, Set[str]] = {n: set() for n in self.nodes}
        for s, ts in self.out_adj.items():
            for t in ts:
                und.setdefault(s, set()).add(t)
                und.setdefault(t, set()).add(s)
        seen: Set[str] = set()
        best = 0
        for start in self.nodes:
            if start in seen:
                continue
            stack, comp = [start], 0
            seen.add(start)
            while stack:
                cur = stack.pop()
                comp += 1
                for nb in und.get(cur, ()):  # noqa: B007
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            best = max(best, comp)
        return round(best / len(self.nodes), 4), best

    def orphan_ratio(self) -> float:
        n = len(self.nodes)
        if not n:
            return 0.0
        iso = sum(1 for x in self.nodes
                  if not self.out_adj.get(x) and not self.in_adj.get(x))
        return round(iso / n, 4)


def load_graph(key: str, root: Optional[str] = None) -> RepositoryGraph:
    """读一张图并物化正/反向邻接表。文件不存在抛 FileNotFoundError。"""
    gpath, _ = _paths(key, root)
    with open(gpath, encoding="utf-8") as f:
        raw = json.load(f)

    g = RepositoryGraph(key=key)
    for n in raw.get("nodes", []):
        nid = n.get("id") or n.get("name")
        if not nid:
            continue
        g.nodes.append(nid)
        g.texts[nid] = n.get("text") or ""
        g.nodes_meta[nid] = n
    known = set(g.nodes)

    for e in raw.get("edges", []):
        if e.get("type") != EDGE_TYPE_CALLS:
            continue
        s, t = e.get("source"), e.get("target")
        if s in known and t in known and s != t:
            g.out_adj.setdefault(s, set()).add(t)
            g.in_adj.setdefault(t, set()).add(s)
    return g


def load_embeddings(key: str, root: Optional[str] = None,
                    l2_normalize: bool = True) -> Tuple[List[str], np.ndarray]:
    """读 embedding，返回 (node_ids, 矩阵)。

    `l2_normalize=True`（默认）—— 官方 npz **未归一化**，实测 L2 p50≈3.91，
    不归一化会让范数偏置污染余弦排序（实测平均 11.58% 的名次偏移）。
    需要复现官方行为时传 `False`。
    """
    _, epath = _paths(key, root)
    z = np.load(epath, allow_pickle=True)
    ids = list(z.keys())
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    V = np.stack([np.asarray(z[n], dtype=np.float32).ravel() for n in ids])
    if l2_normalize:
        nrm = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / np.maximum(nrm, 1e-12)
    return ids, V


def cosine_topk(ids: Sequence[str], V: np.ndarray, q: np.ndarray,
                k: int = 10) -> List[Tuple[str, float]]:
    """余弦 top-k。`q` 已是单条向量（内部会再归一化一次）。"""
    if V.size == 0 or k <= 0:
        return []
    qn = q / max(float(np.linalg.norm(q)), 1e-12)
    sims = V @ qn
    k = min(k, len(ids))
    idx = np.lexsort((np.asarray(ids, dtype=str), -sims))[:k]
    return [(ids[i], float(sims[i])) for i in idx]
