#!/usr/bin/env python3
"""P1 双账计时 v2：官方口径沙箱重建 + F2P 基线 + Phase 2 验证计时。

三笔账（详见 01-research-plan.md v2 §6）：
  账A  沙箱准备（Container A：官方快照解压 + setup.py + eval_baseline 提交）
       —— 不计入 agent 12h 预算
  账B  agent loop 时间
       —— 计入预算；本脚本不测，留待 H1 主实验
  账C  官方 Phase 2 验证（全新 Container B：解压 + setup + gold patch
       + test_patch + hermetic pytest；pytest 本身受 300s 硬顶）
       —— agent 无感

与 v1 的关键差异（实验保真度修正）：
  1. 仓库快照改为在 Linux 容器内从官方 .tgz 解压，而非 bind-mount
     Windows 解压副本。原因：tar 解压到 Windows 丢失 git 符号链接
     （requests_7205 的 tests/certs/*/ca 为 mode 120000），会导致
     TLS 测试假失败，污染 F2P/P2P 判定；且 Windows 侧 core.autocrlf=true
     会造成行尾噪声。容器内解压逐字节复刻官方基线。
  2. F2P/P2P 的 pytest 目标节点从 test_patch 提取（官方 §8.2 口径：
     "test functions extracted from test_patch"），而非跑整个测试文件，
     避免超时度量失真。
  3. pytest 命令逐字复刻官方 hermetic 调用（PYTHONSAFEPATH /
     PYTHONNOUSERSITE / python3 -s / -p no:anyio / -o timeout=0 /
     junitxml），判定标准对齐官方 resolved 条件。
  4. 账C 使用全新 Container B（官方 Phase 2 即全新容器），含其完整
     沙箱准备时间；另单列 pytest 净时长（受 300s 硬顶约束的那一段）。

用法:
    python p1_timing.py httpx_3672 [fastapi_11194 ...]
    python p1_timing.py --all          # 4 个 P1 快照全跑

输出: 默认新建 results/final_p1_timing_<run>.json 及独立 JUnit 目录。
      --all 兼容原4任务；--all-tasks 选择全部129任务。
      --task-manifest 接受换行ID清单或JSON数组。
      --resume 只恢复显式指定的新运行输出；历史原始结果只读。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from run_support import (fresh_output, prepare_junit_dir, prepare_output,
                         run_metadata, select_ids, write_json, read_junit_evidence, junit_complete,
                         prepare_receipt)

ROOT = str(Path(__file__).resolve().parents[2])
WS = os.path.join(ROOT, "results", "p1_workspace")
SNAP_DIR = os.path.join(ROOT, "data", "snapshots")
SETUP_PY = os.path.join(ROOT, "data", "sandbox", "setup.py")
# 官方 wheels/ 有两个实测问题（2026-10-15）：
#  1) 4 个 cp312 轮子在 cp313 沙箱不可装，却被 setup.py 的"选最高版本"
#     去重逻辑选中并毒化整批 pip 安装（如 charset_normalizer-3.5.1-cp312
#     导致 requests 的 urllib3 也没装上）；
#  2) 缺 typing-inspection / h11 / python-multipart 等传递依赖。
# wheels_local/ = cp313 兼容过滤集 + 从 PyPI 补入的缺包
# （构建脚本见 results/tmp/build_wheels_local.py、fetch_missing_wheels.py）。
WHEELS = os.path.join(ROOT, "data", "wheels_local")
MAPPER = os.path.join(ROOT, "paper", "scripts", "p1_map_targets.py")
OUT = os.path.join(ROOT, "results", "p1_timing.json")
JUNIT_DIR = os.path.join(ROOT, "results", "p1_junit")
IMAGE = "swebench-sandbox:latest"

# 官方硬约束（HARNESS_README）：4 GiB / 2 vCPU / 断网
DOCKER_LIMITS = "--memory=4g --cpus=2 --network=none"
# 官方 Phase 2 pytest 硬顶 300s；留 30s 余量便于观察越顶行为
PYTEST_TIMEOUT = 330
SETUP_TIMEOUT = 1800

P1_TASKS = ["httpx_3672", "requests_7205", "rich_3454", "fastapi_11194"]

# 每个仓库目标测试的闭包依赖。官方 harness 用 "cached test site-packages"
# 流式注入预构建测试环境（HARNESS_README §4），公开数据集不含该缓存；
# 本地用 /wheels 近似补齐，预装后官方 setup.py 照常运行。
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


def sh(cmd, timeout=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout)


def dm(c, args, timeout=None):
    return sh(f"docker exec {c} " + args, timeout=timeout)


def start_container(name):
    """按官方约束起容器；/wheels 只读挂载（官方 Dockerfile.public 口径）。"""
    cmd = (f"docker run --rm -d --name {name} {DOCKER_LIMITS} "
           f'-v "{WHEELS}:/wheels:ro" {IMAGE} sleep 3600')
    r = sh(cmd)
    if r.returncode != 0:
        return None, r.stderr.strip()[:300]
    return name, None


def container_alive(c):
    """Windows 下 shell=True 走 cmd.exe：单引号不是引用符，必须用双引号。"""
    r = sh(f'docker inspect -f "{{{{.State.Status}}}} {{{{.State.ExitCode}}}}" {c}')
    if r.returncode != 0:
        return False, "gone"
    parts = r.stdout.strip().split()
    return parts[0] == "running", (parts[1] if len(parts) > 1 else "?")


def prep_sandbox(c, iid, repo):
    """账A/账C 共用的沙箱准备：解压官方快照 + 预装依赖 + setup.py + eval_baseline。

    返回 (秒数, 依赖秒数, setup 输出尾部, 错误信息或 None)。
    """
    t0 = time.time()
    tgz = os.path.join(SNAP_DIR, f"{iid}.tgz")
    if not os.path.isfile(tgz):
        return None, None, "", f"快照缺失: {tgz}"

    r = sh(f'docker cp "{tgz}" {c}:/tmp/snap.tgz')
    if r.returncode != 0:
        return None, None, "", f"docker cp 失败: {r.stderr.strip()[:200]}"

    r = dm(c, 'bash -c "cd /workspace && tar xzf /tmp/snap.tgz"', timeout=600)
    if r.returncode != 0:
        return None, None, "", f"tar 解压失败: {r.stderr.strip()[:200]}"

    # 快照 tar 保留原构建机 uid（786478），容器内以 root 运行会触发
    # git safe.directory 保护；仅加 git 配置，不动仓库内容
    dm(c, 'bash -c "git config --global --add safe.directory /workspace"')

    # 预装闭包依赖（近似官方 "Stream cached test site-packages" 步骤）。
    # 不用官方 setup.py 的慢路径：它把所有解析出的仓库依赖塞进一条 pip
    # 命令，单个不兼容/缺失轮子即毒化整批（实测 charset_normalizer-cp312、
    # flask→blinker、email-validator→dnspython 均导致整批失败）。
    deps = PREDEPS.get(repo)
    t_deps = time.time()
    dep_note = "no-pin"
    if deps:
        pkgs = " ".join(deps)
        r = dm(c, f'bash -c "python3 -m pip install --no-index '
                  f'--find-links=/wheels --no-build-isolation -q {pkgs} '
                  f'2>&1 | tail -2"', timeout=900)
        dep_note = "ok" if r.returncode == 0 else f"pip rc={r.returncode}"
    deps_sec = round(time.time() - t_deps, 1)

    r = sh(f'docker cp "{SETUP_PY}" {c}:/swegemma_setup.py')
    if r.returncode != 0:
        return None, None, "", f"setup.py 拷贝失败: {r.stderr.strip()[:200]}"

    # SWE_PREBAKED=1 走官方 fast-path：editable install + workspace_paths.pth
    # + pytest.ini/conftest.py（跳过易碎的慢路径批量安装）
    try:
        r = dm(c, f'bash -c "cd /workspace && SWE_PREBAKED=1 python '
                  f'/swegemma_setup.py {repo} 2>&1 | tail -12"',
               timeout=SETUP_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, None, "", f"setup.py 超时(>{SETUP_TIMEOUT}s)"
    out = f"[deps {deps_sec}s {dep_note}] " + r.stdout.strip()[-800:]

    # 官方 §8.2 步骤 1：创建 eval_baseline 提交作为 diff 基线
    dm(c, 'bash -c "cd /workspace && git add -A && '
          'git commit -q -m eval_baseline || true"', timeout=120)

    return round(time.time() - t0, 1), deps_sec, out, None


def apply_patch(c, patch_text):
    """4 级回退 apply（对齐官方 apply_patch_in_container 的 pass 1-4）。

    注意 1：stdin 写入必须用 bytes——Windows 下 subprocess(text=True) 会把
    \\n 翻译成 \\r\\n，导致 git apply 失败、GNU patch 把 \\r 掺进源码。
    注意 2（2026-10-15）：官方 test_patch 末行不带换行符时 git apply 报
    "corrupt patch at line N"（fastapi_11194 实测）。写盘前统一补尾换行。
    """
    p = "/tmp/_p1.patch"
    payload = patch_text if patch_text.endswith("\n") else patch_text + "\n"
    r = subprocess.run(
        f'docker exec -i {c} bash -c "cat > {p}"',
        shell=True, input=payload.encode("utf-8"), capture_output=True,
    )
    if r.returncode != 0:
        return False, "write failed"
    for flags in ("-p1", "-p1 --3way", "-p1 --ignore-whitespace",
                  "-p0", "-p1 --recount"):
        rr = dm(c, f"git apply {flags} {p}")
        if rr.returncode == 0:
            return True, flags
    # pass 4：GNU patch 回退（< 重定向必须在容器内 bash 中执行）
    for flags in ("-p1 --batch --forward", "-p0 --batch --forward"):
        rr = dm(c, f'bash -c "patch {flags} -r - < {p}"')
        if rr.returncode == 0:
            return True, f"patch {flags}"
    return False, "all passes failed"


def extract_test_targets(test_patch):
    """从 test_patch 提取 pytest 目标节点（hunk 头启发式，仅作兜底）。

    ⚠️ 该启发式会指错（2026-10-15 实测）：rich_3454 在模块级参数化列表里
    加行，hunk 头显示相邻的 test_wrong_type，真正消费者是
    test_highlight_regex。主路径改用 map_targets_in_container（AST 版）。
    """
    targets = []
    cur_file, cur_func, cur_class = None, None, None

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
            if path and path != "/dev/null":
                cur_file, cur_func, cur_class = path, None, None
        elif line.startswith("@@"):
            # hunk header 尾部是 git funcname 上下文（最近的 def/class 行）
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
                # 模块级 def：退出类上下文
                cur_class = None
                cur_func = m.group(2)
                emit()
                continue
            m = re.match(r"^\s+(?:async\s+)?def\s+(test_\w+)", body)
            if m:
                cur_func = m.group(1)
                emit()
    return targets


def map_targets_in_container(c, patch_text, fallback):
    """在应用了 test_patch 的容器里做 AST 目标映射（新主路径）。

    返回 (targets, source)：source ∈ {"ast", "ast+fallback", "hunk-fallback"}。
    AST 对每个文件给出精确节点；某文件映射失败（fallback=true）或没产出
    节点时，用 hunk 启发式的结果并集补齐。
    """
    r = sh(f'docker cp "{MAPPER}" {c}:/tmp/p1_map_targets.py')
    if r.returncode != 0:
        return fallback, "hunk-fallback"
    r = dm(c, "python3 /tmp/p1_map_targets.py /tmp/_p1.patch", timeout=300)
    if r.returncode != 0 or not r.stdout.strip():
        return fallback, "hunk-fallback"
    try:
        data = json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return fallback, "hunk-fallback"

    nodes, need_fallback = [], False
    for rel, info in data.items():
        if info.get("nodes"):
            nodes.extend(info["nodes"])
        else:
            need_fallback = True
    nodes = sorted(set(nodes))
    if nodes and not need_fallback:
        return nodes, "ast"
    if nodes and need_fallback:
        return sorted(set(nodes) | set(fallback)), "ast+fallback"
    return fallback, "hunk-fallback"


def run_hermetic_pytest(c, targets, junit_path):
    """逐字复刻官方 Phase 2 hermetic pytest 调用。

    命令写入容器内脚本再执行，规避 Windows 多层 shell 引号转义风险。
    """
    import shlex
    tgt = " ".join(shlex.quote(t) for t in targets)
    script = (
        "#!/bin/bash\n"
        "set -eux\n"
        "cd /workspace\n"
        "export PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1\n"
        f"python3 -s -m pytest {tgt} --junitxml={junit_path} "
        "-p no:anyio -o timeout=0 -o python_classes='Test* *Test' -q\n"
    )
    # bytes 写入：规避 Windows 文本模式 \n -> \r\n 翻译
    subprocess.run(
        f'docker exec -i {c} bash -c "cat > /tmp/run_pytest.sh"',
        shell=True, input=script.encode("utf-8"), capture_output=True,
    )
    dm(c, "chmod +x /tmp/run_pytest.sh")
    t0 = time.time()
    timed_out = False
    try:
        r = dm(c, "bash /tmp/run_pytest.sh", timeout=PYTEST_TIMEOUT)
        rc = r.returncode
        # set -x trace 走 stderr；pytest -q 摘要在 stdout，优先展示
        out = (r.stdout.strip() or r.stderr.strip())[-900:]
    except subprocess.TimeoutExpired as e:
        timed_out = True
        raw = e.stdout if isinstance(e.stdout, (bytes, str)) else ""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        rc, out = -1, (raw or "")[-400:]
        dm(c, 'bash -c "pkill -f pytest || true"')
    return rc, round(time.time() - t0, 1), out, timed_out


def junit_stats(xml_path):
    return read_junit_evidence(xml_path)


def timing_absolute_verdict(rc, stats, timed_out=False):
    """Absolute local diagnostic with skipped tests excluded from pass."""
    if timed_out or rc is None:
        return "timeout"
    if not junit_complete(stats):
        return "unknown"
    if stats.get("skipped", 0) == stats.get("tests", 0):
        return "unknown"
    if rc == 0 and stats.get("failures", 0) == 0 and stats.get("errors", 0) == 0 and stats.get("skipped", 0) == 0:
        return "pass"
    return "fail"


def node_passed(target, stats):
    """Match file/class/function and every observed parameter instance.

    A coarse parameterized function lacks the complete parameter inventory, so
    all observed passes are still unknown; an observed failure or skip is false.
    """
    if "cases" not in stats or stats.get("evidence_complete") is False:
        return None
    parts = target.split("::")
    if len(parts) < 2:
        return None
    path = parts[0].replace("\\", "/").removeprefix("./")
    module = path.removesuffix(".py").replace("/", ".")
    expected_classname = ".".join([module] + parts[1:-1])
    requested = parts[-1]
    exact_parameter = "[" in requested
    matches = []
    for case in stats["cases"]:
        name = case["name"]
        if (name != requested if exact_parameter else name.split("[", 1)[0] != requested):
            continue
        classname = case["classname"]
        if classname != expected_classname and not classname.endswith("." + expected_classname):
            continue
        case_file = case.get("file", "").replace("\\", "/")
        if case_file and case_file != path and not case_file.endswith("/" + path):
            continue
        matches.append(case)
    if not matches:
        return None
    if any(case["status"] != "passed" for case in matches):
        return False
    if not exact_parameter and any("[" in case["name"] for case in matches):
        return None
    return True


def reset_workspace(c):
    dm(c, 'bash -c "cd /workspace && git checkout -- . && git clean -fd"',
       timeout=300)


def host_path(p):
    """Windows 路径转 docker CLI 友好的正斜杠形式。"""
    return os.path.abspath(p).replace("\\", "/")


def run_task(iid, tasks):
    task = tasks[iid]
    res = {"instance_id": iid, "repo": task["repo"],
           "started": datetime.now().isoformat(timespec="seconds")}
    hunk_fallback = extract_test_targets(task["test_patch"])
    res["pytest_targets_hunk"] = hunk_fallback  # 留档对比

    # ============ 账A：Container A 沙箱准备 ============
    ca, err = start_container(f"p1a_{iid}")
    if ca is None:
        res["error"] = f"Container A 启动失败: {err}"
        return res
    sec, deps_sec, out, err = prep_sandbox(ca, iid, task["repo"])
    res["account_A_setup_sec"] = sec
    res["account_A_deps_sec"] = deps_sec
    res["setup_output_tail"] = out
    if err:
        alive, code = container_alive(ca)
        res["error"] = f"{err} (alive={alive}, exit={code})"
        sh(f"docker rm -f {ca}")
        return res

    # ============ F2P 基线：仅 test_patch，期望 FAIL ============
    # 先应用 test_patch，再在容器内做 AST 目标映射（对应用后的源码最准）
    ok, how = apply_patch(ca, task["test_patch"])
    res["f2p_patch_applied"] = ok
    res["f2p_apply_how"] = how
    targets, target_src = map_targets_in_container(ca, task["test_patch"],
                                                   hunk_fallback)
    res["pytest_targets"] = targets
    res["pytest_target_count"] = len(targets)
    res["target_source"] = target_src
    jpath = f"/tmp/junit_f2p_{iid}.xml"
    rc, psec, tail, to = run_hermetic_pytest(ca, targets, jpath)
    res["f2p_pytest_sec"] = psec
    res["f2p_pytest_timed_out"] = to
    res["f2p_pytest_tail"] = tail
    res["f2p_exit_code"] = rc
    f2p_receipt = prepare_receipt(os.path.join(JUNIT_DIR, iid + "_f2p.xml"))
    copied = sh(f'docker cp {ca}:{jpath} "{host_path(str(f2p_receipt))}"').returncode == 0 and f2p_receipt.is_file()
    stats = junit_stats(str(f2p_receipt)) if copied else {"error": "missing JUnit receipt after docker cp", "evidence_complete": False}
    res["f2p_junit"] = {k: v for k, v in stats.items() if k != "cases"}
    if "cases" in stats:
        res["f2p_node_pass"] = {t: node_passed(t, stats) for t in targets}
    res["f2p_verdict"] = {
        "expected_fail": (rc != 0) or (stats.get("failures", 0) > 0),
    }
    reset_workspace(ca)
    sh(f"docker rm -f {ca}")

    # ============ 账C：全新 Container B，官方 Phase 2 全流程 ============
    cb, err = start_container(f"p1b_{iid}")
    if cb is None:
        res["error_c"] = f"Container B 启动失败: {err}"
        return res
    t0 = time.time()
    sec, deps_sec, out, err = prep_sandbox(cb, iid, task["repo"])
    if err:
        alive, code = container_alive(cb)
        res["error_c"] = f"{err} (alive={alive}, exit={code})"
        sh(f"docker rm -f {cb}")
        return res
    res["account_C_prep_sec"] = sec
    res["account_C_deps_sec"] = deps_sec

    ok1, how1 = apply_patch(cb, task["patch"])
    ok2, how2 = apply_patch(cb, task["test_patch"])
    res["p2p_gold_applied"] = ok1
    res["p2p_gold_how"] = how1
    res["p2p_test_applied"] = ok2
    jpath = f"/tmp/junit_p2p_{iid}.xml"
    rc, psec, tail, to = run_hermetic_pytest(cb, targets, jpath)
    res["account_C_total_sec"] = round(time.time() - t0, 1)
    res["account_C_pytest_sec"] = psec
    res["p2p_pytest_timed_out"] = to
    res["p2p_exit_code"] = rc
    res["p2p_pytest_tail"] = tail
    p2p_receipt = prepare_receipt(os.path.join(JUNIT_DIR, iid + "_p2p.xml"))
    copied = sh(f'docker cp {cb}:{jpath} "{host_path(str(p2p_receipt))}"').returncode == 0 and p2p_receipt.is_file()
    stats = junit_stats(str(p2p_receipt)) if copied else {"error": "missing JUnit receipt after docker cp", "evidence_complete": False}
    res["p2p_junit"] = {k: v for k, v in stats.items() if k != "cases"}
    if "cases" in stats:
        res["p2p_node_pass"] = {t: node_passed(t, stats) for t in targets}
    # 官方 resolved 条件：exit 0 且 passed>0 且 failures==0 且 errors==0
    # 且所有必需节点显式通过
    nodes_ok = all(v for v in res.get("p2p_node_pass", {}).values()) \
        if res.get("p2p_node_pass") else None
    res["verdict"] = {
        "f2p_expected_fail": res["f2p_verdict"]["expected_fail"],
        "p2p_expected_pass": bool(
            rc == 0
            and junit_complete(stats)
            and stats.get("failures", 0) == 0
            and stats.get("errors", 0) == 0
            and nodes_ok
        ),
    }
    sh(f"docker rm -f {cb}")
    res["finished"] = datetime.now().isoformat(timespec="seconds")
    return res


def select_task_ids(tasks, explicit=(), *, all_tasks=False, legacy_all=False, manifest=None):
    return select_ids(tasks, explicit, all_tasks, legacy_all, manifest, P1_TASKS)


def timing_complete(record):
    """Only complete, executed XML measurements can be skipped on resume."""
    if not record.get("finished") or any(record.get(key) for key in ("fatal", "error", "error_c")):
        return False
    for phase in ("f2p", "p2p"):
        if (record.get(f"{phase}_pytest_timed_out") or
                not isinstance(record.get(f"{phase}_exit_code"), int) or
                not junit_complete(record.get(f"{phase}_junit") or {})):
            return False
    return all(isinstance(record.get("verdict", {}).get(key), bool)
               for key in ("f2p_expected_fail", "p2p_expected_pass"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--all", action="store_true", help="Legacy selection: only the four P1 calibration tasks")
    parser.add_argument("--all-tasks", action="store_true", help="Select all 129 data/tasks.jsonl records")
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--junit-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="Skip completed records and retry incomplete records in --output")
    parser.add_argument("--dry-run", action="store_true", help="Print selection/configuration without Docker or output writes")
    args = parser.parse_args(argv)
    task_path = Path(ROOT) / "data" / "tasks.jsonl"
    tasks = {d["instance_id"]: d for d in
             (json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    try:
        iids = select_task_ids(tasks, args.ids, all_tasks=args.all_tasks,
                               legacy_all=args.all, manifest=args.task_manifest)
        if args.resume and args.output is None:
            raise ValueError("--resume requires explicit --output")
        out_path, results = prepare_output(args.output or fresh_output(ROOT, "p1_timing"), ROOT, args.resume)
        junit_dir = args.junit_dir or out_path.with_suffix("").with_name(out_path.stem + "_junit")
        if args.all:
            print("[legacy --all] selects four calibration tasks; use --all-tasks for all 129", file=sys.stderr)
        if args.resume:
            iids = [iid for iid in iids if not timing_complete(results.get(iid, {}))]
        if args.dry_run:
            print(json.dumps({"task_ids": iids, "n_tasks": len(iids), "output": str(out_path),
                              "junit_dir": str(junit_dir), "workers": 1}, ensure_ascii=False))
            return 0
        global JUNIT_DIR
        JUNIT_DIR = str(prepare_junit_dir(junit_dir, ROOT, args.resume))
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    metadata = run_metadata(task_path, __file__, iids, workers=1,
                            docker_limits=DOCKER_LIMITS, image=IMAGE, resume=args.resume,
                            junit_dir=JUNIT_DIR,
                            required_coverage="AST target proxy; official required-test inventory unavailable")
    metadata_path = out_path.with_suffix(".run.json")
    if args.resume and metadata_path.is_file():
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("task_source_sha256", "script_sha256", "image", "docker_limits"):
            if prior.get(key) != metadata.get(key):
                raise ValueError(f"resume configuration mismatch for {key}")
        metadata = prior
        metadata.setdefault("resume_attempts", []).append(run_metadata(task_path, __file__, iids, workers=1,
            docker_limits=DOCKER_LIMITS, image=IMAGE, resume=True, junit_dir=JUNIT_DIR,
            required_coverage="AST target proxy; official required-test inventory unavailable"))
    write_json(metadata_path, metadata)
    for iid in iids:
        if iid not in tasks:
            print(f"SKIP {iid}: 不在 tasks.jsonl", file=sys.stderr)
            continue
        print(f"=== P1 {iid} ({tasks[iid]['repo']}) ===", flush=True)
        try:
            res = run_task(iid, tasks)
        except Exception as e:
            res = {"instance_id": iid, "fatal": f"{type(e).__name__}: {e}"}
        results[iid] = res
        write_json(out_path, results)
        v = res.get("verdict", {})
        print(f"    账A={res.get('account_A_setup_sec')}s "
              f"账C总={res.get('account_C_total_sec')}s "
              f"pytest={res.get('account_C_pytest_sec')}s "
              f"F2P失败={v.get('f2p_expected_fail')} "
              f"P2P通过={v.get('p2p_expected_pass')}", flush=True)
        if res.get("error") or res.get("error_c") or res.get("fatal"):
            print(f"    !! {res.get('error') or res.get('error_c') or res.get('fatal')}",
                  flush=True)

    print(f"\n写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
