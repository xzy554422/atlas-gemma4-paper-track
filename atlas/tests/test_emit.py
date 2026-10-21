"""emit.py 硬限制校验的反例测试。

每个用例都对着 HARNESS_README 里的一条硬限制。**这些测试必须全部抛
`ValidationError`** —— 官方编译器在提交阶段才发现问题，代价是 Kaggle
配额；提前在本地拦住才是这个模块存在的意义。

symlink 用例走注入的 `symlink_probe` 而不是真造链接：Windows 非管理员
会话下 `os.symlink` 会静默失败（实测：调用返回成功，但 `os.listdir`
为空、`os.path.islink` 为 False），造不出测试前提。
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.emit import (  # noqa: E402
    ALLOWED_MODEL_NAMES, DEFAULT_MODEL, EvalBudget, MAX_LORA_RANK,
    MAX_TOTAL_BYTES, ValidationError, emit_agent_yaml, emit_skill,
    emit_sub_agent, estimate_size, validate, write_submission, SubmissionLimits,
)

PASS, FAIL = [], []
_TEMP_ROOT = tempfile.TemporaryDirectory(prefix="atlas-emit-check-")


def _tmp_with_agent(model=DEFAULT_MODEL, extra_files=()):
    d = tempfile.mkdtemp(dir=_TEMP_ROOT.name)
    with open(os.path.join(d, "agent.yaml"), "w", encoding="utf-8") as f:
        f.write(f"name: a\nmodel: {model}\n")
    for rel, content in extra_files:
        full = os.path.join(d, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
    return d


def expect_raise(name, fn):
    try:
        fn()
    except ValidationError as e:
        PASS.append(name)
        print(f"  [PASS] {name}: {str(e)[:78]}")
        return
    except Exception as e:  # noqa: BLE001
        FAIL.append(name)
        print(f"  [FAIL] {name}: 抛了非 ValidationError -> {type(e).__name__}: {e}")
        return
    FAIL.append(name)
    print(f"  [FAIL] {name}: 竟然通过了校验")


def expect_ok(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        FAIL.append(name)
        print(f"  [FAIL] {name}: 误拒 -> {type(e).__name__}: {e}")
        return
    PASS.append(name)
    print(f"  [PASS] {name}: 正确放行")


def main():
    print("=" * 78)
    print("atlas.emit 硬限制校验")
    print("=" * 78)

    print("\n【放行基线】")
    expect_ok("正常 submission", lambda: validate(_tmp_with_agent()))

    print("\n【根配置唯一性 §2.2】")
    d = tempfile.mkdtemp(dir=_TEMP_ROOT.name)
    with open(os.path.join(d, "a.yaml"), "w", encoding="utf-8") as f:
        f.write("name: a\n")
    expect_raise("无根配置 → MissingRootConfigError", lambda: validate(d))
    shutil.rmtree(d, ignore_errors=True)

    expect_raise("多根配置 → MultipleRootConfigsError", lambda: validate(
        _tmp_with_agent(extra_files=[("agent.yml", "name: a\n")])))

    print("\n【symlink 禁令 §2.1】")
    # probe 命中目录项
    probe_dir = lambda p: os.path.basename(p) == "skills"  # noqa: E731
    expect_raise("symlink 目录", lambda: validate(
        _tmp_with_agent(extra_files=[("skills/x/SKILL.md", "---\nname: x\n---\n")]),
        symlink_probe=probe_dir))
    probe_file = lambda p: os.path.basename(p) == "evil"  # noqa: E731
    expect_raise("symlink 文件", lambda: validate(
        _tmp_with_agent(extra_files=[("evil", "x")]), symlink_probe=probe_file))

    print("\n【单基座模型规则 §3.2 / §3.3】")
    expect_raise("多基座模型", lambda: validate(
        _tmp_with_agent(extra_files=[("sub_agents/a.yaml",
                                      "model: gemma-4-31b-it-qat-w4a16-ct\n")])))
    expect_raise("模型不在 ALLOWED_MODEL_NAMES",
                 lambda: validate(_tmp_with_agent(model="llama-3-70b")))
    expect_raise("emit 阶段就拦住非法模型",
                 lambda: emit_agent_yaml(model="llama-3-70b"))

    print("\n【LoRA rank §3.4】")
    expect_raise("rank 256 > max_lora_rank 128", lambda: validate(
        _tmp_with_agent(extra_files=[
            ("adapters/l1/adapter_config.json", json.dumps({"r": 256}))])))
    expect_ok("rank 64 放行", lambda: validate(
        _tmp_with_agent(extra_files=[
            ("adapters/l2/adapter_config.json", json.dumps({"r": 64}))])))
    expect_ok(f"rank {MAX_LORA_RANK} 边界放行", lambda: validate(
        _tmp_with_agent(extra_files=[
            ("adapters/l3/adapter_config.json",
             json.dumps({"r": MAX_LORA_RANK}))])))

    print("\n【路径规则 §2.1】")
    from atlas.emit import _check_relpath
    for bad in ("../evil", "a/../../evil", "sub_agents/../../x"):
        expect_raise(f"路径穿越 {bad}",
                     lambda b=bad: _check_relpath(b, "skill"))
    for bad in ("/abs", "/etc/passwd", "C:\\Windows", "\\\\server\\share"):
        expect_raise(f"绝对路径 {bad}",
                     lambda b=bad: _check_relpath(b, "skill"))
    expect_raise("skill 里的 .. 分量",
                 lambda: emit_skill("..", "body"))
    expect_raise("skill 绝对路径", lambda: emit_skill("/abs", "body"))
    expect_raise("sub_agent 穿越", lambda: emit_sub_agent("../../evil"))

    print("\n【体积 §2.4】")
    big = _tmp_with_agent(extra_files=[("prompts/blob.txt", "")])
    # Exercise the same inclusive size boundary with a small configured limit;
    # truncate() does not guarantee a sparse file on every Windows filesystem.
    blob = os.path.join(big, "prompts", "blob.txt")
    with open(blob, "wb") as f:
        f.truncate(4096)
    total = sum(os.path.getsize(os.path.join(base, name))
                for base, _, names in os.walk(big) for name in names)
    expect_raise("解压后体积达到配置上限", lambda: validate(
        big, SubmissionLimits(max_total_size_bytes=total)))
    shutil.rmtree(big, ignore_errors=True)

    est = estimate_size([("npz", 540 * 1024 * 1024),
                         ("graph json", 1100 * 1024 * 1024)])
    print(f"  参考：129 图 + 129 npz 估算 {est['total_gib']} GiB，"
          f"是否塞得下 3 GiB = {est['fits_in_3gib']}")

    print("\n【端到端生成】")
    d = tempfile.mkdtemp(dir=_TEMP_ROOT.name)
    rep = write_submission(
        d,
        emit_agent_yaml(instructions="你是定位助手",
                        sub_agents=["sub_agents/locator.yaml"],
                        skills=["skills/atlas-locate"],
                        tools=["run_command", "read_file"]),
        skills={"atlas-locate": emit_skill(
            "atlas-locate", "先跑影响分析，再跑测试。", description="图引导定位")},
        sub_agents={"locator": emit_sub_agent("locator", instructions="定位改动点。")},
        eval_config=EvalBudget(max_tool_calls=60, max_time_minutes=30),
    )
    expect_ok("write_submission 自带 validate", lambda: rep["ok"] and None)
    print(f"  产物：{rep['written']} 文件 / {rep['total_gib']} GiB / "
          f"models={rep['models']} / skills={len(rep['skills'])}")
    shutil.rmtree(d, ignore_errors=True)

    print("\n" + "=" * 78)
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("失败用例：" + ", ".join(FAIL))
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
