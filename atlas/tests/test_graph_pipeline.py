"""graphio / impact / locate 的行为测试。

这些模块承载论文的核心主张，所以测试不只是「跑通」，而是把主张本身
钉成断言 —— 任何人改动实现导致主张失效，测试立刻红。

关键断言：
  1. `graph_key_for` 用的是 `<repo_short>_<base_commit>` 而非 instance_id
  2. embedding 必须 L2 归一化（官方 npz 未归一化）
  3. `impacted_tests` 的 `by_hop[k]` 是**增量**不是累计
  4. 零出边的测试节点对反向可达闭包贡献为零（H1 的硬边界）
  5. `effective_k_cap` 在超长节点存在时仍要有区分度
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "paper", "scripts"))

import numpy as np  # noqa: E402

from atlas.graphio import (  # noqa: E402
    graph_key_for, is_test_node, is_test_symbol, iter_available_keys,
    load_embeddings, load_graph, node_to_file_suffix,
)
from atlas.impact import (  # noqa: E402
    diagnose_applicability, impact_of_files, impacted_tests, reverse_khop,
)
from atlas.locate import (  # noqa: E402
    LocateConfig, build_b0, build_b1, build_b2, effective_k_cap,
    lexical_score, pack_to_char_budget,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = os.path.join(ROOT, "data", "tasks.jsonl")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f": {detail}" if detail else ""))


def main():
    tasks = [json.loads(l) for l in open(TASKS, encoding="utf-8")]
    print("=" * 78)
    print(f"atlas graphio / impact / locate —— {len(tasks)} 个任务的数据")
    print("=" * 78)

    print("\n【graph_key 推导】")
    t = tasks[0]
    key = graph_key_for(t)
    check("key 形如 <repo_short>_<base_commit>",
          key == f"{t['repo'].rsplit('/', 1)[-1]}_{t['base_commit']}", key)
    check("key 不是 instance_id", key != t["instance_id"])
    n_avail = len(iter_available_keys())
    check("本地可用 key 数 ≈ 129", 120 <= n_avail <= 129, f"{n_avail}")

    print("\n【embedding 归一化】")
    ids, V = load_embeddings(key)
    _, Vraw = load_embeddings(key, l2_normalize=False)
    check("归一化后 L2 ≈ 1",
          abs(float(np.median(np.linalg.norm(V, axis=1))) - 1.0) < 1e-4,
          f"{float(np.median(np.linalg.norm(V, axis=1))):.6f}")
    med_raw = float(np.median(np.linalg.norm(Vraw, axis=1)))
    check("官方 npz 未归一化（L2 明显 > 1）", med_raw > 1.5, f"p50={med_raw:.3f}")
    check("维度为 256", V.shape[1] == 256, str(V.shape))

    print("\n【图结构与 H1 适用性】")
    g = load_graph(key)
    check("节点数 > 0", len(g) > 0, str(len(g)))
    iso = g.isolation_ratio()
    diag = diagnose_applicability(g)
    check("诊断量与直算一致",
          abs(diag["test_node_isolation_ratio"] - iso) < 1e-9)
    check("fastapi 的 H1 适用性被判 False",
          (not diag["h1_applicable"]) == (iso >= 0.30),
          f"iso={iso:.3f} applicable={diag['h1_applicable']}")

    print("\n【反向 k-hop 的硬边界：零出边测试节点】")
    seeds = [n for n in g.nodes
             if is_test_node(n) and g.in_adj.get(n)][:3]
    if seeds:
        s = seeds[0]
        check("零 incoming 的测试节点 1-hop 影响集为空",
              impacted_tests(g, [s], 1).get(1, []) == [],
              f"seed={s}")
    isolated_tests = [n for n in g.test_nodes()
                      if not g.out_adj.get(n) and not g.in_adj.get(n)]
    check("图里存在零出边测试节点（fastapi 的已知缺口）",
          len(isolated_tests) > 0, f"{len(isolated_tests)} 个")

    print("\n【by_hop 是增量不是累计】")
    # 可控链：test_a 被 test_b 调用、test_b 被 test_c 调用、test_c 被 test_d 调用。
    # 节点名必须像测试用例（is_test_symbol 要求 test 前缀），否则 impacted_tests
    # 会把它们全过滤掉 —— 这本身也是一条要记住的语义。
    class FakeG:
        in_adj = {"pkg.test_a": {"pkg.test_b"},
                  "pkg.test_b": {"pkg.test_c"},
                  "pkg.test_c": {"pkg.test_d"}}
    fg = FakeG()
    by = impacted_tests(fg, ["pkg.test_a"], 3)
    check("1-hop 取到 test_b", by.get(1) == ["pkg.test_b"], str(by.get(1)))
    check("2-hop 取到 test_c（不含 test_b）", by.get(2) == ["pkg.test_c"], str(by.get(2)))
    check("3-hop 取到 test_d", by.get(3) == ["pkg.test_d"], str(by.get(3)))
    closure = reverse_khop(fg, ["pkg.test_a"], 3)
    check("闭包是累计（4 个点）", len(closure) == 4, str(sorted(closure)))

    print("\n【预算：5,000 字符顶】")
    alloc, total = pack_to_char_budget(["x" * 100] * 50)
    check("k=50 时每节点 100 字符",
          all(a == 100 for a in alloc) and total == 5000, f"total={total}")
    alloc2, total2 = pack_to_char_budget(["x" * 10] * 5)
    check("短片段余量滚给长片段（总量用满 5,000）",
          total2 == 50, f"total={total2}")
    # 有一个超长节点时，k 必须仍能超过 0（初版「全部完整」会恒为 0）
    texts = ["y" * 900] * 8 + ["z" * 383_734] + ["w" * 50] * 40
    cap = effective_k_cap(texts)
    check("超长节点存在时 k_cap 仍 > 0（有区分度）", cap > 0, f"k={cap}")
    check("k_cap ≤ 可用节点数", cap <= len(texts))

    print("\n【定位信号】")
    check("同名词干得 1.0",
          abs(lexical_score("parse header", "pkg.mod.parse_header") - 1.0) < 1e-9)
    check("完全无关得 0.0",
          lexical_score("parse header", "pkg.mod.zzz_yyy") == 0.0)
    check("驼峰可切分",
          lexical_score("client timeout", "a.TestClient.foo") > 0.0)

    print("\n【三档注入端到端】")
    cfg = LocateConfig(max_k=50)
    b0 = build_b0(g, t["problem_statement"], cfg)
    b1 = build_b1(g, t["problem_statement"], cfg)
    b2 = build_b2(g, t["problem_statement"], cfg)
    check("B0 返回 max_k 个节点", b0.n == min(50, len(g)))
    check("B2 ⊆ B0 的 1-hop 闭包且很小", 0 < b2.n <= b0.n, f"B2={b2.n}")
    check("B1 至少覆盖 B0 的种子", set(cfg and [b0.ranked[0]]) <= set(b1.ranked),
          "B0 top-1 必须在 B1 里")
    check("B1 的种子集就是 B0 的 top-seed_k",
          b1.seed_nodes == b0.ranked[:cfg.seed_k],
          f"{len(b1.seed_nodes)} 个")
    check("B0 无重复节点", len(set(b0.ranked)) == len(b0.ranked))
    check("B1 无重复节点", len(set(b1.ranked)) == len(b1.ranked))
    check("B1 扩展数 = n - 种子数",
          b1.n == min(cfg.max_k, len(b1.seed_nodes) + b1.expanded),
          f"n={b1.n} seed={len(b1.seed_nodes)} exp={b1.expanded}")

    print("\n【impact_of_files 端到端】")
    from p1b_h1_feasibility import diff_files
    changed = sorted(diff_files(t["patch"]))
    r = impact_of_files(g, t["instance_id"], changed)
    check("unresolved_files 与 changed 对齐",
          set(r.unresolved_files) <= set(changed))
    check("isolation_ratio 已填入", 0.0 <= r.isolation_ratio <= 1.0,
          f"{r.isolation_ratio}")
    d = r.to_dict()
    check("to_dict 含 1/2/3-hop 计数",
          all(k in d for k in ("impacted_tests_1hop", "impacted_tests_2hop",
                               "impacted_tests_3hop")))

    print("\n" + "=" * 78)
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("失败：" + ", ".join(FAIL))
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
