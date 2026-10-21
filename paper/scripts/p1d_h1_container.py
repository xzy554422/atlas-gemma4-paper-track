#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1 Step 3：H1 静态测试影响选择的**容器实跑**决策一致率实验。

回答的问题（研究计划 §3 H1）：
  给定候选补丁 P，用静态图信号从全量测试 T 中选出子集 T'，
  T' 上的通过/拒绝判定与官方口径（target tests）是否一致？
  与全量套件口径是否一致？

设计要点：
  1. 生产式 `select_tests` 接口只需要候选补丁 P；本历史扰动实验为保证跨 variant
     可比性，在每个任务上预先用 gold 改动文件固定一次选择，随后复用该 T'。
     因此这些历史比较带有 gold-location 信息优势，不应当解释为候选补丁端到端结果。
  2. 参照系有两个（对齐研究计划 v2 §3 的双口径）：
       官方 target-test 口径 = test_patch 经 AST 映射出的节点（主口径）
       全量套件口径 = 仓库测试目录全跑（次口径，用于成本上界）
  3. 容器复用：每个任务只做 1 次沙箱准备（账A 20s 级），
     之后所有扰动用「容器内改文件 + git checkout 回滚」，不重建容器。
     这与官方「每任务双容器」不同，但只影响墙钟测量口径，
     不影响判定结果——已在校准输出里单列说明。
  4. 扰动族（6 类，覆盖 accept / reject / accept-preserving 三种）：
       gold          参考解，期望 accept
       noop          空补丁，期望 reject（bug 未修）
       revert        应用 gold 后回滚被改文件，期望 reject
       syntax_break  gold + 非法 token，期望 reject（collection error）
       comment_only  gold + 一行注释，期望 accept（判定应保持，不得误杀）
       wrong_file    gold + 往另一个库文件注入非法 token，期望 reject

输出：默认新建 results/final_p1d_h1_<run>.json 及独立 JUnit 目录。
      --output/--junit-dir 可显式指定；--resume 恢复新运行结果。
      baseline 按完整绝对证据检查；未知差分默认重试，--keep-unknown 可保留已完成未知状态。
      原始历史 JSON/JUnit 路径只读，汇总不再隐式写合并文件。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics as st
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from run_support import (fresh_output, prepare_junit_dir, prepare_output,
                         run_metadata, select_ids, write_json, read_junit_evidence, junit_complete,
                         prepare_receipt)

ROOT = str(Path(__file__).resolve().parents[2])
SNAP_DIR = os.path.join(ROOT, "data", "snapshots")
SETUP_PY = os.path.join(ROOT, "data", "sandbox", "setup.py")
WHEELS = os.path.join(ROOT, "data", "wheels_local")
MAPPER = os.path.join(ROOT, "paper", "scripts", "p1_map_targets.py")
GRAPHS = os.path.join(ROOT, "data", "graphs")
OUT = os.path.join(ROOT, "results", "p1d_h1_decision.json")
JUNIT_DIR = os.environ.get(
    "P1D_JUNIT_DIR", os.path.join(ROOT, "results", "p1d_junit"))
IMAGE = "swebench-sandbox:latest"

DOCKER_LIMITS = "--memory=4g --cpus=2 --network=none"
PYTEST_TIMEOUT = int(os.environ.get("P1D_PYTEST_TIMEOUT", "330"))
SETUP_TIMEOUT = 1800
WORKERS = int(os.environ.get("P1D_WORKERS", "1"))

# 与 p1_timing.py 保持一致的闭包依赖（本地近似官方 cached site-packages）
PREDEPS = {
    "encode/httpx": ["certifi", "httpcore", "h11", "anyio", "sniffio", "idna"],
    "psf/requests": ["urllib3", "charset-normalizer", "idna", "certifi"],
    "Textualize/rich": ["markdown-it-py", "mdurl", "pygments"],
    "fastapi/fastapi": [
        "starlette==0.48.0", "pydantic==2.13.4", "pydantic-core==2.46.4",
        "annotated-types", "typing-extensions", "typing-inspection",
        "anyio", "sniffio", "idna", "certifi",
        "httpx==0.28.1", "httpcore", "h11", "python-multipart",
    ],
}

VARIANTS = ["baseline", "gold", "noop", "revert", "syntax_break",
            "comment_only", "wrong_file"]
MODES = ["target", "selected", "full"]
PERTURB_VARIANTS = [v for v in VARIANTS if v != "baseline"]


def active_modes():
    """本次运行实际要跑的口径（`target_cont` 由 target 的 junit 结果动态决定）。

    `target_cont` 是 target 的**兜底口径**：官方 target 口径绝不加
    `--continue-on-collection-errors`（HARNESS_README §8.2 步骤 4 原文），
    所以一旦扰动把模块 import 打断（例如 syntax_break 打在
    `src/requests/models.py`），pytest 直接 `Interrupted: N error during
    collection` 退出码 4、**根本不写 junitxml**，于是差分判定失去
    bad 集合，任务被整个排除在分析之外。
    2026-10-16 实测 psf/requests 8 个任务 × 2 个破坏性扰动 × 3 个口径
    = 48 条记录全部栽在这里（占全量 129 任务的 12.4%）。
    target_cont = 同一批官方 target 节点 + 该参数，只用于在 junit 缺失时
    补一份可做集合差分的证据；官方 target 口径本身的数值永不被它覆盖。

    `P1D_TARGET_CONT`：
        auto（默认）只在 target 的 junit 缺失时补跑，正常任务零额外成本
        1          强制每个变体都补跑（用于回归验证兜底逻辑本身）
        0          完全关闭
    """
    m = ["target"]
    m += [x for x in enabled_modes()]
    return m


def target_cont_policy():
    v = os.environ.get("P1D_TARGET_CONT", "auto").lower()
    return v if v in ("auto", "0", "1") else "auto"


def enabled_modes():
    m = []
    if os.environ.get("P1D_SELECTED", "1") == "1":
        m.append("selected")
    if os.environ.get("P1D_FULL", "1") == "1":
        m.append("full")
    return m


# ------------------------------------------------------------------ 宿主工具
def sh(cmd, timeout=None, input_bytes=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, input=input_bytes)


def dm(c, args, timeout=None):
    return sh(f"docker exec {c} " + args, timeout=timeout)


def host_path(p):
    return os.path.abspath(p).replace("\\", "/")


def start_container(name):
    cmd = (f'docker run --rm -d --name {name} {DOCKER_LIMITS} '
           f'-v "{WHEELS}:/wheels:ro" {IMAGE} sleep 7200')
    r = sh(cmd)
    if r.returncode != 0:
        return None, r.stderr.strip()[:300]
    return name, None


def prep_sandbox(c, iid, repo):
    """账A：解压快照 + 预装依赖 + 官方 setup.py(fast-path) + eval_baseline。"""
    t0 = time.time()
    tgz = os.path.join(SNAP_DIR, f"{iid}.tgz")
    if not os.path.isfile(tgz):
        return None, f"快照缺失: {tgz}"
    if sh(f'docker cp "{tgz}" {c}:/tmp/snap.tgz').returncode != 0:
        return None, "docker cp 失败"
    r = dm(c, 'bash -c "cd /workspace && tar xzf /tmp/snap.tgz"', timeout=900)
    if r.returncode != 0:
        return None, f"tar 失败: {r.stderr.strip()[:200]}"
    dm(c, 'bash -c "git config --global --add safe.directory /workspace"')
    deps = PREDEPS.get(repo)
    dep_note = "no-pin"
    if deps:
        r = dm(c, 'bash -c "python3 -m pip install --no-index '
                  f'--find-links=/wheels --no-build-isolation -q {" ".join(deps)} '
                  f'2>&1 | tail -2"', timeout=900)
        dep_note = "ok" if r.returncode == 0 else f"pip rc={r.returncode}"
    if sh(f'docker cp "{SETUP_PY}" {c}:/swegemma_setup.py').returncode != 0:
        return None, "setup.py 拷贝失败"
    try:
        r = dm(c, f'bash -c "cd /workspace && SWE_PREBAKED=1 python '
                  f'/swegemma_setup.py {repo} 2>&1 | tail -8"',
               timeout=SETUP_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, f"setup.py 超时(>{SETUP_TIMEOUT}s)"
    dm(c, 'bash -c "cd /workspace && git add -A && '
          'git commit -q -m eval_baseline || true"', timeout=180)
    return round(time.time() - t0, 1), f"[deps {dep_note}] {r.stdout.strip()[-400:]}"


def apply_patch(c, patch_text):
    p = "/tmp/_p1d.patch"
    payload = patch_text if patch_text.endswith("\n") else patch_text + "\n"
    subprocess.run(f'docker exec -i {c} bash -c "cat > {p}"', shell=True,
                   input=payload.encode("utf-8"), capture_output=True)
    for flags in ("-p1", "-p1 --3way", "-p1 --ignore-whitespace", "-p0"):
        if dm(c, f"git apply {flags} {p}").returncode == 0:
            return True, flags
    for flags in ("-p1 --batch --forward", "-p0 --batch --forward"):
        if dm(c, f'bash -c "patch {flags} -r - < {p}"').returncode == 0:
            return True, f"patch {flags}"
    return False, "failed"


def reset_workspace(c):
    dm(c, 'bash -c "cd /workspace && git checkout -- . && git clean -fdq"',
       timeout=300)


# --------------------------------------------------------- 官方 target 节点
def map_targets_in_container(c, patch_text, fallback):
    if sh(f'docker cp "{MAPPER}" {c}:/tmp/p1_map_targets.py').returncode != 0:
        return fallback, "hunk-fallback"
    r = dm(c, "python3 /tmp/p1_map_targets.py /tmp/_p1d_test.patch", timeout=300)
    if r.returncode != 0 or not r.stdout.strip():
        return fallback, "hunk-fallback"
    try:
        data = json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return fallback, "hunk-fallback"
    nodes = [n for info in data.values() for n in info.get("nodes", [])]
    nodes = sorted(set(nodes))
    return (nodes or fallback), ("ast" if nodes else "hunk-fallback")


def extract_test_targets(test_patch):
    """p1_timing.extract_test_targets 的自包含副本（hunk 启发式兜底）。"""
    targets, cur_file, cur_func, cur_class = [], None, None, None

    def emit():
        if cur_file and cur_func and cur_func.startswith("test_"):
            node = (f"{cur_file}::{cur_class}::{cur_func}"
                    if cur_class else f"{cur_file}::{cur_func}")
            if node not in targets:
                targets.append(node)

    for line in test_patch.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            cur_file = None if path in ("/dev/null", "") else path
            cur_func = cur_class = None
        elif line.startswith("@@"):
            m = re.search(r"@@.*@@\s*(.*)$", line)
            ctx = m.group(1).strip() if m else ""
            cm = re.search(r"class\s+(\w+)", ctx)
            fm = re.search(r"(?:async\s+)?def\s+(\w+)", ctx)
            if cm:
                cur_class, cur_func = cm.group(1), None
            elif fm:
                cur_func = fm.group(1)
            emit()
        elif line.startswith("+") and not line.startswith("+++"):
            body = line[1:]
            m = re.match(r"^class\s+(\w+)", body)
            if m:
                cur_class, cur_func = m.group(1), None
                continue
            m = re.match(r"^(async\s+)?def\s+(test_\w+)", body)
            if m:
                cur_class, cur_func = None, m.group(2)
                emit()
                continue
            m = re.match(r"^\s+(?:async\s+)?def\s+(test_\w+)", body)
            if m:
                cur_func = m.group(1)
                emit()
    return targets


# ------------------------------------------------------------- T' 静态选取
def diff_files(diff_text):
    out = set()
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            p = line[4:].strip()
            if p.startswith("b/"):
                p = p[2:]
            if p != "/dev/null":
                out.add(p)
    return out


def node_to_file(nid):
    head, _, _ = nid.rpartition(".")
    return head.replace(".", "/") + ".py" if head else None


def is_test_file(path):
    base = os.path.basename(path)
    parts = path.split("/")
    if base.startswith("test_") or base.endswith("_test.py"):
        return True
    return any(seg in ("tests", "test", "testing") for seg in parts[:-1]) \
        and base.endswith(".py")


def select_tests(changed_files, problem_statement, graph_path):
    """H1 的静态选取：反向 ∞-hop ∪ 词干重叠（离线实验里表现最佳的 S4）。

    只用 agent 推理期可见的信息：候选补丁改了哪些文件 + problem_statement。
    不使用 test_patch / gold。
    """
    if not os.path.exists(graph_path):
        return [], "no-graph"
    with open(graph_path, encoding="utf-8") as fh:
        g = json.load(fh)
    rev = defaultdict(set)
    for e in g.get("edges", []):
        if e.get("type") == "calls" and e.get("source") and e.get("target"):
            rev[e["target"]].add(e["source"])  # 反向：被调用方 → 调用方
    node2file, file2nodes = {}, defaultdict(set)
    for n in g.get("nodes", []):
        nid = n.get("id") or n.get("name")
        f = node_to_file(nid)
        if f:
            node2file[nid] = f
            file2nodes[f].add(nid)

    all_tests = {s for s in file2nodes if is_test_file(s)}

    # 种子：改动文件 → 图节点
    seeds = set()
    for c in changed_files:
        for suf, nids in file2nodes.items():
            if c == suf or c.endswith("/" + suf):
                seeds |= nids

    # 反向 ∞-hop
    seen, frontier = set(seeds), set(seeds)
    while frontier:
        nxt = set()
        for n in frontier:
            nxt |= rev.get(n, set())
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    khop = {node2file[n] for n in seen if n in node2file}

    # 词干重叠
    STOP = {"test", "tests", "main", "src", "lib", "app", "utils", "core",
            "base", "common", "init", "conftest", "setup", "python"}

    def stems(name):
        b = re.sub(r"\.py$", "", os.path.basename(name))
        b = re.sub(r"^test[_\.]", "", b)
        return {w for w in re.split(r"[_\.]", b.lower())
                if len(w) > 3 and w not in STOP}

    hints = set()
    for c in changed_files:
        hints |= stems(c)
    for w in re.findall(r"[a-z][a-z0-9_]{3,}", problem_statement.lower()):
        if w not in {"this", "that", "with", "from", "when", "then", "there"}:
            hints.add(w)
    stem_hit = {f for f in all_tests if stems(f) & hints}

    sel = sorted({f for f in (khop | stem_hit) if is_test_file(f)})
    return sel, ("khop+stem" if sel else "empty")


# ------------------------------------------------------------------ pytest
def run_pytest(c, targets, junit, timeout=PYTEST_TIMEOUT, cont_on_error=False):
    """逐字复刻官方 hermetic pytest（官方命令见 HARNESS_README §8.2 步骤 4）。

    `cont_on_error=True` 只用于**全量套件口径**：官方 target 口径绝不加它。
    实测 rich_3454 在 pytest 9.1.1 下 `tests/test_pretty.py` 抛
    `ModuleNotFoundError: attr`（官方 wheels 不含 attrs，本地无法补齐），
    默认行为会 `Interrupted: 1 error during collection` 让 768 个已收集用例
    一个都不跑 → 全量口径被单一无关 collection error 归零。
    基线里另有 10 个 pygments 版本敏感用例天然失败（`test_syntax.py`），
    故全量口径必须配 `--continue-on-collection-errors` 并与 baseline 差分。
    """
    if not targets:
        return None, 0.0, "no targets", False
    tgt = " ".join(shlex.quote(t) for t in targets)
    extra = " --continue-on-collection-errors" if cont_on_error else ""
    # 超时必须由**容器内**的 timeout 保证。
    # 2026-10-16 实测：只靠宿主 subprocess timeout 时，宿主的 docker exec
    # 客户端被掐断，但容器里的 pytest 继续跑（State: R）。宿主随后卡在
    # `docker cp`（同一个 exec 会话），该分片就此僵死 ——
    # p1d_rich_4006 从 21:55 挂到 22:33（38 分钟），另外两个分片各自
    # 只完成 7–9 个任务。容器镜像里没有 pkill/ps（实测全部 MISSING），
    # 所以改用 bash 内建 job control + timeout(1) 收尾。
    inner = int(max(timeout - 15, 30))
    script = (
        "#!/bin/bash\nset -x\ncd /workspace\n"
        "export PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1\n"
        f"timeout -k 5 {inner} "
        f"python3 -s -m pytest {tgt} --junitxml={junit} "
        f"-p no:anyio -o timeout=0 -o python_classes='Test* *Test' -q{extra}\n"
        "rc=$?\n"
        "# timeout 只杀直接子进程；用 jobs 拿到后台 job 的 pid 再 kill，\n"
        "# 兜住 pytest 自己 fork 出来的 worker（容器内无 pkill/pgrep）。\n"
        "for j in $(jobs -p 2>/dev/null); do kill -9 \"$j\" 2>/dev/null; done\n"
        "exit $rc\n"
    )
    subprocess.run(f'docker exec -i {c} bash -c "cat > /tmp/rp.sh"',
                   shell=True, input=script.encode("utf-8"), capture_output=True)
    dm(c, "chmod +x /tmp/rp.sh")
    t0 = time.time()
    try:
        r = dm(c, "bash /tmp/rp.sh", timeout=timeout)
        rc, out = r.returncode, (r.stdout.strip() or r.stderr.strip())[-500:]
    except subprocess.TimeoutExpired as e:
        raw = e.stdout if isinstance(e.stdout, (bytes, str)) else ""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        rc, out = -1, (raw or "")[-300:]
        # 宿主 timeout 先到：容器内那次 timeout 应该已经收尾；这里只是
        # 保险起见再放一遍 kill，且不阻塞（exec 已断开会立即返回）。
        dm(c, 'bash -c "for j in $(jobs -p 2>/dev/null); do kill -9 $j; done"',
           timeout=30)
        return rc, round(time.time() - t0, 1), out, True
    return rc, round(time.time() - t0, 1), out, rc == 124 or rc == 137


def junit_stats(path):
    stats = read_junit_evidence(path)
    stats.pop("cases", None)
    return stats


def verdict(rc, stats, timed_out):
    """Local absolute diagnostic; official required-test acceptance is separate."""
    if timed_out or rc is None:
        return "timeout"
    if not junit_complete(stats):
        return "unknown"
    if stats.get("skipped", 0) == stats.get("tests", 0):
        return "unknown"
    if stats.get("skipped", 0) > 0:
        return "fail"
    if rc == 0 and stats["failures"] == 0 and stats["errors"] == 0:
        return "pass"
    return "fail"


def needs_target_cont(stats, policy=None):
    """Fallback on absent/incomplete XML, not an empty (successful) bad set."""
    policy = target_cont_policy() if policy is None else policy
    return policy == "1" or (policy == "auto" and not junit_complete(stats))


def apply_baseline_delta(rec, base_rec, mode):
    """把绝对判定转成「相对未打补丁基线」的判定。

    全量套件口径下，即使完全正确的补丁也会被判 fail —— 基线本身就有
    预存在的失败（实测 rich_3454：pytest 9.1.1 + pygments 2.21 下
    `test_syntax.py` 10 个用例天然失败、`test_pretty.py` 1 个 collection error）。
    官方 target 口径不受影响（target 是本次 test_patch 引入的节点，基线必全红）。

    差分口径：
        new_bad = 本次 bad 集合 - 基线 bad 集合
        引入回归 → regress（对应失败模式 F4）
        无新增失败 → clean（仅回归诊断，不等于 repaired/resolved）
        JUnit 缺失或超时 → unknown（默认可恢复重试）
    """
    cur = (rec.get(mode) or {}).get("junit") or {}
    base = ((base_rec or {}).get(mode) or {}).get("junit") or {}
    if "bad" not in cur or "bad" not in base:
        rec[mode]["delta_verdict"] = "unknown"
        rec[mode]["delta_reason"] = "missing_current_junit" if "bad" not in cur else "missing_baseline_junit"
        return
    if (rec[mode].get("timed_out") or (base_rec or {}).get(mode, {}).get("timed_out")):
        rec[mode]["delta_verdict"] = "unknown"
        rec[mode]["delta_reason"] = "timeout"
        return
    if not junit_complete(cur) or not junit_complete(base):
        rec[mode]["delta_verdict"] = "unknown"
        rec[mode]["delta_reason"] = "incomplete_or_unexecuted_junit"
        return
    new_bad = sorted(set(cur["bad"]) - set(base["bad"]))
    fixed = sorted(set(base["bad"]) - set(cur["bad"]))
    rec[mode]["delta_verdict"] = "regress" if new_bad else "clean"
    rec[mode]["new_failures"] = new_bad[:20]
    rec[mode]["n_new_failures"] = len(new_bad)
    rec[mode]["n_fixed"] = len(fixed)


# ------------------------------------------------------------------ 扰动
def perturb_python(c, path, mode):
    """在容器内对指定 .py 做扰动（不依赖 unified diff 生成）。

    关键设计：**扰动必须注入到 target 测试能测到的符号上**，否则 target 口径
    会判 clean 而扰动本意是 regress —— 造成「假一致」。
    2026-10-16 实测：rich_3454 的 `wrong_file`（往 tests/ 之外的随机 .py
    注入非法 token）官方 target 判 clean，因为 target 只跑
    `test_highlight_regex`，根本不 import 那个文件。
    → 统一改为**在 gold patch 改动的同一文件、同一符号体内**做扰动：
       syntax_break  在被改函数的最后一个语句后插入非法 token
       comment_only  在被改函数的最后一个语句后插入一行注释（应保持判定）
       delete_branch 删除被改函数体的前 1/8 缩进行（逻辑取反，应 regress）
    """
    script = f'''
import ast, io, os, sys
p = {path!r}
mode = {mode!r}
if not os.path.exists(p):
    print("MISSING"); sys.exit(3)
src = io.open(p, encoding="utf-8").read()
try:
    tree = ast.parse(src)
except SyntaxError:
    print("SYNTAX_ERR_BEFORE"); sys.exit(5)

# 找被改函数体：test_patch 的 AST 映射已知 target 节点，但这里要用「gold 改动的行」。
# 简化：取文件中最后一个有缩进体的函数/类，作为注入点（gold 必然改到它）。
best = None
for node in ast.walk(tree):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    body = [b for b in node.body if not isinstance(
        b, (ast.Expr, ast.Pass, ast.Return)) or not (
        isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant)
        and isinstance(b.value.value, str))]
    ind = [b for b in body if b.lineno > node.lineno]
    if ind and (best is None or node.end_lineno > best[0]):
        best = (node.end_lineno, node.name, sorted(b.lineno for b in ind))

if best is None:
    print("NO_TARGET"); sys.exit(4)
end, name, _ = best
lines = src.splitlines(True)
if end > len(lines):
    print("NO_TARGET"); sys.exit(4)

if mode == "syntax_break":
    lines.insert(end, "\\n@@ATLAS_BREAK@@\\n")
elif mode == "comment_only":
    lines.insert(end, "\\n# atlas benign comment\\n")
elif mode == "delete_branch":
    # 在被改函数体内删掉前 1/8 的缩进语句行（逻辑取反）
    tree2 = ast.parse(src)
    fn = None
    for node in ast.walk(tree2):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \\
                and node.end_lineno == end:
            fn = node
    if fn is None:
        print("NO_TARGET"); sys.exit(4)
    stmts = [s for s in fn.body if s.lineno > fn.lineno]
    if not stmts:
        print("NO_TARGET"); sys.exit(4)
    drop_stmts = stmts[: max(1, len(stmts) // 8)]
    drop_lines = set()
    for s in drop_stmts:
        for i in range(s.lineno, (s.end_lineno or s.lineno) + 1):
            drop_lines.add(i)
    lines = [l for i, l in enumerate(lines, 1) if i not in drop_lines]
else:
    print("BAD_MODE"); sys.exit(2)

io.open(p, "w", encoding="utf-8").write("".join(lines))
print("OK")
'''
    subprocess.run(f'docker exec -i {c} bash -c "cat > /tmp/perturb.py"',
                   shell=True, input=script.encode("utf-8"), capture_output=True)
    r = dm(c, "python3 /tmp/perturb.py", timeout=180)
    return r.returncode == 0 and "OK" in r.stdout


def wrong_file_target(c, changed_files):
    """在**与 gold 改动文件同一目录**的另一个 .py 里注入（跨文件误改）。

    不能取全局随机 .py —— 那样改动落在 target 测试的 import 闭包之外，
    官方 target 口径根本感知不到，扰动语义失效（见 perturb_python 注释）。
    优先改同目录的兄弟模块；若 gold 只动了 1 个文件且无兄弟，退回
    「同目录 + 名字含 test 但不是 target」也无效，故取同目录任意 .py。
    """
    dirs = {os.path.dirname(p) for p in changed_files if p}
    for d in sorted(dirs):
        r = dm(c, f'bash -c "cd /workspace && git ls-files {shlex.quote(d)}/*.py 2>/dev/null"',
               timeout=120)
        cands = [x.strip() for x in r.stdout.splitlines()
                 if x.strip().endswith(".py") and x.strip() not in changed_files]
        if cands:
            return cands[0]
    r = dm(c, 'bash -c "cd /workspace && git ls-files 2>/dev/null | head -400"',
           timeout=120)
    for p in [x.strip() for x in r.stdout.splitlines() if x.strip().endswith(".py")]:
        if p not in changed_files and os.path.dirname(p) in dirs:
            return p
    return None


# ------------------------------------------------------------------ 单任务
def run_task(iid, task, variants=VARIANTS, do_full=True, do_selected=True):
    res = {"instance_id": iid, "repo": task["repo"],
           "started": datetime.now().isoformat(timespec="seconds"),
           "runs": {}}
    name = f"p1d_{iid}"
    c, err = start_container(name)
    if c is None:
        res["error"] = f"容器启动失败: {err}"
        return res
    try:
        sec, note = prep_sandbox(c, iid, task["repo"])
        res["account_A_setup_sec"] = sec
        if sec is None:
            res["error"] = note
            return res
        res["setup_tail"] = note

        # 官方 target 节点：先应用 test_patch 做 AST 映射
        ok, how = apply_patch(c, task["test_patch"])
        res["test_patch_applied"] = ok
        res["test_patch_how"] = how
        p_tp = "/tmp/_p1d_test.patch"
        payload = task["test_patch"] if task["test_patch"].endswith("\n") \
            else task["test_patch"] + "\n"
        subprocess.run(f'docker exec -i {c} bash -c "cat > {p_tp}"',
                       shell=True, input=payload.encode("utf-8"),
                       capture_output=True)
        targets, tsrc = map_targets_in_container(c, task["test_patch"],
                                                 extract_test_targets(task["test_patch"]))
        res["target_nodes"] = targets
        res["target_source"] = tsrc
        reset_workspace(c)

        # 历史实验固定一次 gold-derived T' 以便跨 variant 配对；这不是 candidate-only 选择。
        gold_files = sorted(diff_files(task["patch"]))
        res["gold_files"] = gold_files
        gkey = f"{task['repo'].split('/')[-1]}_{task['base_commit']}"
        gpath = os.path.join(GRAPHS, gkey + ".json")
        sel, how2 = select_tests(gold_files, task["problem_statement"], gpath)
        res["n_selected_raw"] = len(sel)
        sel, n_dropped = prune_missing(c, sel)
        res["n_selected_dropped_nonexistent"] = n_dropped
        res["selected_tests"] = sel
        res["selected_how"] = how2
        res["n_selected"] = len(sel)
        full_all = count_test_files(c)
        full, n_full_drop = prune_missing(c, full_all)
        res["full_test_files_raw"] = len(full_all)
        res["full_test_files"] = full
        res["n_full_dropped_nonexistent"] = n_full_drop

        for v in variants:
            rec = {}
            reset_workspace(c)
            # --- 施加扰动
            if v in ("noop", "baseline"):
                pass
            elif v == "revert":
                apply_patch(c, task["patch"])
                reset_workspace(c)   # 施加再回滚 = 等价 noop，但走完整 apply 路径
            else:
                ok_g, _ = apply_patch(c, task["patch"])
                rec["gold_applied"] = ok_g
                if v in ("syntax_break", "comment_only", "delete_branch") and gold_files:
                    rec["perturb_ok"] = perturb_python(c, gold_files[0], v)
                elif v == "wrong_file":
                    wf = wrong_file_target(c, gold_files)
                    rec["wrong_file"] = wf
                    if wf:
                        rec["perturb_ok"] = perturb_python(c, wf, "syntax_break")
            # test_patch 必须在扰动之后应用（官方 Phase 2 顺序）。
            # 全量口径下 test_patch 引入的用例会额外产生真实回归
            # （例如 rich_3454 修好 test_highlight_regex 后，
            #  test_markdown 反而开始失败），必须纳入差分。
            rec["tp_applied"] = apply_patch(c, task["test_patch"])[0]

            # 全量口径要收全 tests/ 目录（含 test_patch 新增文件），
            # 而不是 git ls-files 快照清单——test_patch 可能新增测试文件。
            if do_full:
                full_run = dm(c, 'bash -c "cd /workspace && find tests -name '
                                  "'test_*.py' -o -name '*_test.py' 2>/dev/null | sort'",
                              timeout=180)
                fr = [x.strip() for x in full_run.stdout.splitlines()
                      if x.strip().endswith(".py") and is_test_file(x.strip())]
                if len(fr) < len(full):
                    fr = sorted(set(full) | set(fr))
                res["full_test_files_runtime"] = fr
            else:
                fr = full

            modes = active_modes()
            queue = list(modes)
            while queue:
                m = queue.pop(0)
                tgt = {"target": targets, "target_cont": targets,
                       "selected": sel, "full": fr}[m]
                if not tgt:
                    rec[m] = {"verdict": "empty", "n_targets": 0}
                    continue
                j = f"/tmp/j_{v}_{m}.xml"
                # selected 与 full 必须用同一套参数，否则墙钟对比无意义。
                # 2026-10-16 实测：selected 口径缺 --continue-on-collection-errors 时，
                # 撞上 tests/test_pretty.py 的 collection error 会
                # `Interrupted: 1 error during collection` 整轮中止
                # （rich_3454 只跑了 1 个用例、1.2s，而 full 跑 803 个用例、4.9s），
                # 伪造出「T' 墙钟削减 76%」的假象。
                # target 口径保持官方原样（绝不加该参数）；
                # target_cont 是唯一例外（见 active_modes 的 docstring）。
                rc, sec_, tail, to = run_pytest(c, tgt, j,
                                               cont_on_error=(m != "target"))
                receipt = prepare_receipt(os.path.join(JUNIT_DIR, f"{iid}_{v}_{m}.xml"))
                copied = sh(f'docker cp {c}:{j} "{host_path(str(receipt))}"').returncode == 0 and receipt.is_file()
                st = junit_stats(str(receipt)) if copied else {"error": "missing JUnit receipt after docker cp", "evidence_complete": False}
                rec[m] = {
                    "verdict": verdict(rc, st, to),
                    "sec": sec_, "exit_code": rc, "timed_out": to,
                    "n_targets": len(tgt), "junit": st, "tail": tail[:250],
                }
                # 官方 target 口径被 collection error 打断（退出码 4 = pytest
                # `Interrupted: N error during collection`，junitxml 根本不落盘）时，
                # 差分判定拿不到 bad 集合，任务会被整个踢出分析。此时按
                # P1D_TARGET_CONT 策略补跑同一批节点 + --continue-on-collection-errors。
                # auto 策略下正常任务 rc != 4，零额外成本。
                want_cont = needs_target_cont(st)
                if m == "target" and want_cont and "target_cont" not in rec:
                    rec["target_junit_missing"] = not st.get("evidence_complete", False)
                    queue.append("target_cont")
            reset_workspace(c)
            res["runs"][v] = rec
        # 基线差分（放在循环外，需要 res["runs"]["baseline"] 已就绪）
        base_rec = res["runs"].get("baseline")
        for v, rec in res["runs"].items():
            if v == "baseline":
                continue
            for m in MODES + ["target_cont"]:
                if m in rec:
                    apply_baseline_delta(rec, base_rec, m)
    finally:
        sh(f"docker rm -f {name}")
    res["finished"] = datetime.now().isoformat(timespec="seconds")
    return res


def count_test_files(c):
    """全量套件口径的测试文件清单（对齐 pytest 收集语义）。

    注意：不能用 `git ls-files '*.py' | grep tests/` —— Windows cmd.exe 不把单引号
    当引用符，整条命令退化成 `git ls-files *.py`，再叠一层 grep 后结果为空
    （2026-10-16 实测 rich_3454 返回 0 个文件）。
    改为让容器内 python 走 git ls-files 全量输出再筛，避开所有引号层。
    """
    script = r'''
import subprocess, sys, re
out = subprocess.run(["git", "-C", "/workspace", "ls-files"],
                     capture_output=True, text=True).stdout.splitlines()
keep = []
for p in out:
    if not p.endswith(".py"):
        continue
    base = p.rsplit("/", 1)[-1]
    segs = p.split("/")[:-1]
    if (base.startswith("test_") or base.endswith("_test.py")
            or any(s in ("tests", "test", "testing") for s in segs)):
        keep.append(p)
print("\n".join(keep))
'''
    subprocess.run(f'docker exec -i {c} bash -c "cat > /tmp/ls_tests.py"',
                   shell=True, input=script.encode("utf-8"), capture_output=True)
    r = dm(c, "python3 /tmp/ls_tests.py", timeout=180)
    return [x.strip() for x in r.stdout.splitlines()
            if x.strip().endswith(".py") and is_test_file(x.strip())]


def prune_missing(c, paths, probe=12):
    """过滤掉快照里不存在的路径。

    图节点名只保留路径后缀，反推回全路径会有猜测成分；同时仓库里存在
    形如 `tests/test_console/BrokenRenderable.py` 的**数据文件**，
    直接传给 pytest 会 `file or directory not found` 并中断整轮收集。
    实测 rich_3454：图给出的 85 个路径里有 `tests/test_console/*.py`（数据文件），
    必须先裁掉。
    """
    if not paths:
        return [], 0
    script = (
        "import os,sys\n"
        f"paths = {paths!r}\n"
        "keep=[p for p in paths if os.path.isfile(os.path.join('/workspace',p))]\n"
        "sys.stdout.write('\\n'.join(keep))\n"
    )
    subprocess.run(f'docker exec -i {c} bash -c "cat > /tmp/prune.py"',
                   shell=True, input=script.encode("utf-8"), capture_output=True)
    r = dm(c, "python3 /tmp/prune.py", timeout=180)
    keep = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    return keep, len(paths) - len(keep)


# ------------------------------------------------------------------ 汇总
def absolute_complete(record):
    """A measured pass/fail needs verified XML plus exit status."""
    return bool(record.get("verdict") in ("pass", "fail")
                and isinstance(record.get("exit_code"), int)
                and not record.get("timed_out")
                and junit_complete(record.get("junit") or {}))


def task_complete(result, variants, modes=None, keep_unknown=False):
    """Baseline has no delta by definition; unknowns are retryable by default."""
    runs = result.get("runs") or {}
    for variant in variants:
        for mode in modes or active_modes():
            record = (runs.get(variant) or {}).get(mode) or {}
            if variant == "baseline":
                if not absolute_complete(record):
                    return False
            elif record.get("delta_verdict") in ("clean", "regress"):
                if not absolute_complete(record):
                    return False
            elif not (keep_unknown and result.get("finished") and
                      record.get("delta_verdict") == "unknown" and record.get("verdict")):
                return False
    return bool(variants)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--all", "--all-tasks", dest="all_tasks", action="store_true")
    parser.add_argument("--task-manifest", "--insts-from", dest="task_manifest", type=Path)
    parser.add_argument("--output", type=Path, default=os.environ.get("P1D_OUT"))
    parser.add_argument("--junit-dir", type=Path, default=os.environ.get("P1D_JUNIT_DIR"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-unknown", action="store_true", help="With resume, retain terminal unknown deltas instead of retrying")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    task_path = Path(ROOT) / "data" / "tasks.jsonl"
    tasks = {d["instance_id"]: d for d in
             (json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    try:
        iids = select_ids(tasks, args.ids, all_tasks=args.all_tasks, manifest=args.task_manifest)
        shard = os.environ.get("P1D_SHARD")
        if shard:
            k, m = (int(x) for x in shard.split("/"))
            if not 0 <= k < m:
                raise ValueError("P1D_SHARD must use zero-based k/m with 0 <= k < m")
            iids = [iid for i, iid in enumerate(iids) if i % m == k]
        if args.keep_unknown and not args.resume:
            raise ValueError("--keep-unknown requires --resume")
        if args.resume and args.output is None:
            raise ValueError("--resume requires explicit --output (or P1D_OUT)")
        if os.environ.get("P1D_SKIP_DONE") == "1" and not args.resume:
            raise ValueError("P1D_SKIP_DONE now requires --resume and an explicit new-run --output")
        out_path, results = prepare_output(args.output or fresh_output(ROOT, "p1d_h1"), ROOT, args.resume)
        junit_dir = args.junit_dir or out_path.with_suffix("").with_name(out_path.stem + "_junit")
        variants = os.environ.get("P1D_VARIANTS", ",".join(VARIANTS)).split(",")
        if not variants or set(variants) - set(VARIANTS + ["delete_branch"]):
            raise ValueError("P1D_VARIANTS contains an unsupported variant")
        if args.resume:
            before = len(iids)
            iids = [iid for iid in iids if not task_complete(results.get(iid, {}), variants,
                                                           keep_unknown=args.keep_unknown)]
            print(f"[resume] skipped {before - len(iids)} completed tasks; {len(iids)} remain")
        if args.dry_run:
            print(json.dumps({"task_ids": iids, "n_tasks": len(iids), "output": str(out_path),
                              "junit_dir": str(junit_dir), "workers": 1}, ensure_ascii=False))
            return 0
        global JUNIT_DIR
        JUNIT_DIR = str(prepare_junit_dir(junit_dir, ROOT, args.resume))
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    do_full = os.environ.get("P1D_FULL", "1") == "1"
    do_sel = os.environ.get("P1D_SELECTED", "1") == "1"
    metadata_path = out_path.with_suffix(".run.json")
    metadata = run_metadata(
        task_path, __file__, iids, workers=1, image=IMAGE, docker_limits=DOCKER_LIMITS,
        variants=variants, modes=active_modes(), target_cont=target_cont_policy(),
        resume=args.resume, keep_unknown=args.keep_unknown, junit_dir=JUNIT_DIR)
    if args.resume and metadata_path.is_file():
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("task_source_sha256", "script_sha256", "image", "docker_limits", "variants", "modes"):
            if prior.get(key) != metadata.get(key):
                raise ValueError(f"resume configuration mismatch for {key}")
        metadata = prior
        metadata.setdefault("resume_attempts", []).append(run_metadata(
            task_path, __file__, iids, workers=1, image=IMAGE, docker_limits=DOCKER_LIMITS,
            variants=variants, modes=active_modes(), target_cont=target_cont_policy(),
            resume=True, keep_unknown=args.keep_unknown, junit_dir=JUNIT_DIR))
    write_json(metadata_path, metadata)
    for iid in iids:
        if iid not in tasks:
            print(f"SKIP {iid}  <-- 不在 data/tasks.jsonl 中", file=sys.stderr,
                  flush=True)
            continue
        print(f"=== P1D {iid} ({tasks[iid]['repo']}) ===", flush=True)
        try:
            res = run_task(iid, tasks[iid], variants, do_full, do_sel)
        except Exception as e:
            res = {"instance_id": iid, "fatal": f"{type(e).__name__}: {e}"}
        results[iid] = res
        write_json(out_path, results)
        for v, r in (res.get("runs") or {}).items():
            if "fatal" in res or "error" in res:
                break
            row = " ".join(
                f"{m}={r.get(m, {}).get('verdict', '?')}"
                f"({r.get(m, {}).get('sec', '?')}s)"
                for m in active_modes() if m in r)
            print(f"    {v:<14} {row}", flush=True)
        if res.get("fatal") or res.get("error"):
            print(f"    !! {res.get('fatal') or res['error']}", flush=True)

    summarize(results)
    print(f"→ 结果 {out_path}")
    return 0


def summarize(results, _recurse=True):
    """Read-only descriptive summary; legacy _recurse argument is ignored.

    Source selection/merging and authoritative paired estimates belong to the
    final audit. This display never writes historical merged result artifacts.
    """
    ok = {k: v for k, v in results.items() if v.get("runs")}
    if not ok:
        print("\n(无可用结果)")
        return
    print("\n" + "=" * 78)
    print(f"P1D 汇总（{len(ok)} 任务）")
    print("=" * 78)
    print("【绝对判定】= 本地runner的exit/JUnit状态；不代表官方resolved验证")
    print(f"{'扰动':<14}{'官方target':>12}{'T静态选取':>12}{'全量套件':>12}{'T/target一致':>13}")
    agree = defaultdict(lambda: [0, 0])
    tsec, fsec, nsel, nfull = [], [], [], []
    for v in VARIANTS:
        rows = [r["runs"].get(v) for r in ok.values() if r["runs"].get(v)]
        rows = [r for r in rows if r and "target" in r]
        if not rows:
            continue

        def cnt(m):
            return sum(1 for r in rows
                       if r.get(m, {}).get("verdict") == "pass")

        n = len(rows)
        vt, vs, vf = cnt("target"), cnt("selected"), cnt("full")
        same = sum(1 for r in rows
                   if r.get("selected", {}).get("verdict")
                   == r.get("target", {}).get("verdict"))
        pairs = [r for r in rows if r.get("target", {}).get("verdict") in ("pass", "fail")
                 and r.get("selected", {}).get("verdict") in ("pass", "fail")]
        if v != "baseline":
            agree[v][0] += sum(r["selected"]["verdict"] == r["target"]["verdict"] for r in pairs)
            agree[v][1] += len(pairs)
        tsec += [r["selected"]["sec"] for r in rows
                 if isinstance(r.get("selected", {}).get("sec"), (int, float))]
        fsec += [r["full"]["sec"] for r in rows
                 if isinstance(r.get("full", {}).get("sec"), (int, float))]
        nsel += [r["selected"]["n_targets"] for r in rows if "selected" in r]
        nfull += [r["full"]["n_targets"] for r in rows if "full" in r]
        print(f"{v:<14}{f'{vt}/{n} pass':>12}{f'{vs}/{n} pass':>12}"
              f"{f'{vf}/{n} pass':>12}{f'{same}/{n}':>13}")
    print()
    tot_a = sum(a for a, _ in agree.values())
    tot_b = sum(b for _, b in agree.values())
    if tot_b:
        print(f"绝对判定一致率（T' vs 官方 target，剔除 baseline）: "
              f"{tot_a}/{tot_b} = {tot_a/tot_b*100:.1f}%")

    print()
    print("【差分判定】= 相对未打补丁基线的回归诊断；unknown不计为一致或通过")
    print(f"{'扰动':<14}{'期望':<10}{'target':>12}{'T静态选取':>12}{'全量套件':>12}"
          f"{'T/target一致':>13}")
    EXP = {"gold": "clean", "noop": "clean", "revert": "clean",
           "syntax_break": "regress", "comment_only": "clean",
           "wrong_file": "regress"}
    agree2 = defaultdict(lambda: [0, 0])
    for v in VARIANTS:
        rows = [r["runs"].get(v) for r in ok.values() if r["runs"].get(v)]
        rows = [r for r in rows if r and
                r.get("target", {}).get("delta_verdict") in ("clean", "regress")]
        if not rows:
            continue

        def dcnt(m):
            return sum(1 for r in rows
                       if r.get(m, {}).get("delta_verdict") == "clean")

        n = len(rows)
        dt, ds, df = dcnt("target"), dcnt("selected"), dcnt("full")
        pairs = [r for r in rows if r.get("selected", {}).get("delta_verdict") in ("clean", "regress")]
        same = sum(r["selected"]["delta_verdict"] == r["target"]["delta_verdict"] for r in pairs)
        agree2[v][0] += same
        agree2[v][1] += len(pairs)
        print(f"{v:<14}{EXP.get(v, '?'):<10}{f'{dt}/{n} clean':>12}"
              f"{f'{ds}/{n} clean':>12}{f'{df}/{n} clean':>12}{f'{same}/{n}':>13}")
    print()
    tot_a = sum(a for a, _ in agree2.values())
    tot_b = sum(b for _, b in agree2.values())
    if tot_b:
        print(f"差分判定一致率（T' vs 官方 target）: {tot_a}/{tot_b} = {tot_a/tot_b*100:.1f}%")
        # 误杀：期望 accept 却判 regress
        fa = [r for r in ok.values() for v, rec in r["runs"].items()
              if EXP.get(v) == "clean"
              and rec.get("selected", {}).get("delta_verdict") == "regress"]
        print(f"误杀（期望 clean 但 T' 判 regress）: {len(fa)} 例"
              + (f" — {fa[0]['instance_id']}" if fa else ""))
    print()
    if tsec and fsec:
        print(f"pytest 墙钟 median：T' {st.median(tsec):.1f}s / "
              f"全量 {st.median(fsec):.1f}s / "
              f"削减 {100*(1-st.median(tsec)/st.median(fsec)):.0f}%")
    if nsel and nfull:
        print(f"目标数 median：T' {st.median(nsel):.0f} / "
              f"全量 {st.median(nfull):.0f} / "
              f"削减 {100*(1-st.median(nsel)/st.median(nfull)):.0f}%")
    # Summaries are read-only. Audited provenance merging is a separate operation.


if __name__ == "__main__":
    sys.exit(main())
