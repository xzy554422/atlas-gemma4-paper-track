"""emit.py —— 生成官方声明式 submission（agent.yaml / sub_agents / skills）。

## 官方契约（全部来自 HARNESS_README §2 / §3，唯一事实来源）

提交物是**纯 YAML**，不提交 Python 入口：

```
submission/
├── agent.yaml            # 恰好一个根配置（多/零都报错）
├── eval_config.yaml      # 可选，evaluation.{timeout_seconds,max_tool_calls,...}
├── sub_agents/*.yaml
├── skills/<name>/SKILL.md
└── adapters/             # 可选 LoRA
```

## 本地结构预检（最终判据仍是官方编译器和运行时）

| 限制 | 出处 | 值 |
|---|---|---|
| 解压后总体积 | §2.4 | **< 3 GiB**（3,221,225,472 字节，含 `adapters/`） |
| LoRA rank | §3.4 | **≤ 128**（`max_lora_rank = max(detected_ranks) or 128`） |
| 基座模型 | §3.2 | 必须**唯一**且在 `ALLOWED_MODEL_NAMES` 内 |
| symlink | §2.1 | **禁止** |
| 路径穿越 | §2.1 | 绝对路径 / 空字节 / `..` 分量 / 越界 symlink 全部 `PathTraversalError` |
| 上下文 | §2.4 | 4x L4 窗口 **32,768 token** |
| 单次工具输出 | §6.3 | **5,000 字符** |

## 为什么 Atlas 天然满足这些限制

Atlas 不训练 LoRA（`adapters/` 可以整块不出现），不做 symlink，
所有资源都是仓库内相对路径 —— 体积上限的唯一真实风险是
「把 129 张图 + 129 个 npz 全塞进去」。实测 `npz` 单文件 ~4.2 MB
（4287 × 256 × float32），129 个 ≈ 540 MB，加上图 JSON ≈ 1.1 GB，
仍在 3 GiB 内，但已占三分之一；`estimate_size()` 把这件事量化出来，
让人在提交前就知道，而不是打包完才发现超限。
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SubmissionLimits", "MAX_TOTAL_BYTES", "MAX_LORA_RANK",
    "ALLOWED_MODEL_NAMES", "DEFAULT_MODEL", "MAX_CONTEXT_TOKENS",
    "MAX_STDOUT_CHARS", "ROOT_CONFIG_NAMES",
    "emit_agent_yaml", "emit_skill", "emit_sub_agent", "write_submission",
    "validate", "ValidationError", "estimate_size", "EvalBudget",
]

# §3.2 / §3.3
ALLOWED_MODEL_NAMES = (
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
)
DEFAULT_MODEL = "gemma-4-e4b-it-qat-w4a16-ct"     # 竞争默认 starter kit

# §2.4
MAX_TOTAL_BYTES = 3_221_225_472        # 3 GiB
MAX_LORA_RANK = 128
MAX_CONTEXT_TOKENS = 32_768
MAX_STDOUT_CHARS = 5_000

ROOT_CONFIG_NAMES = ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml")


class ValidationError(Exception):
    """本地结构预检失败；不替代官方编译器的完整 schema 校验。"""


@dataclass(frozen=True)
class SubmissionLimits:
    """官方 §2.4 `build_submission_limits()` 的镜像，便于测试。"""

    max_total_size_bytes: int = MAX_TOTAL_BYTES
    max_lora_rank: int = MAX_LORA_RANK
    max_context_tokens: int = MAX_CONTEXT_TOKENS
    max_stdout_chars: int = MAX_STDOUT_CHARS
    allowed_models: Tuple[str, ...] = ALLOWED_MODEL_NAMES


@dataclass
class EvalBudget:
    """`eval_config.yaml` 的 `evaluation:` 段（§7.1）。"""

    timeout_seconds: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_time_minutes: Optional[float] = None
    max_turns: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        d = {k: v for k, v in asdict(self).items() if v is not None}
        return d


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------
def _is_symlink(path: str, probe: Optional[Callable[[str], bool]] = None) -> bool:
    """symlink 判定。

    默认用 `os.lstat` + `S_ISLNK`，而不是 `os.path.islink`：后者在部分
    Windows 环境下对 junction 会误判为 False（官方 §2.1 禁的正是
    「解析到 submission root 之外」的链接，junction 也在内）。

    `probe` 是注入点：Windows 非管理员会话下 `os.symlink` 会**静默失败**
    （实测创建后 `os.listdir` 为空、`islink` False），无法在测试里造出
    真的 symlink，于是把判定逻辑抽成可替换的函数来测。
    """
    if probe is not None:
        return probe(path)
    try:
        info = os.lstat(path)
        return stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    except (OSError, AttributeError):
        return False


def _check_relpath(rel: str, what: str) -> None:
    """§2.1 的路径规则：相对路径、无空字节、无 `..` 分量。"""
    if not rel:
        raise ValidationError(f"{what}: 空路径")
    if "\x00" in rel:
        raise ValidationError(f"{what}: 含空字节")
    if os.path.isabs(rel) or rel.startswith("/") or rel.startswith("\\"):
        raise ValidationError(f"{what}: 绝对路径被禁止 ({rel})")
    parts = rel.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        raise ValidationError(f"{what}: 路径穿越被禁止 ({rel})")
    if re_drive(rel):
        raise ValidationError(f"{what}: 盘符路径被禁止 ({rel})")


def re_drive(rel: str) -> bool:
    return len(rel) >= 2 and rel[1] == ":"


def validate(root: str, limits: Optional[SubmissionLimits] = None,
             symlink_probe: Optional[Callable[[str], bool]] = None
             ) -> Dict[str, object]:
    """对一个 submission 目录做本地结构预检。

    返回体检报告；任一硬限制违反则抛 `ValidationError`。
    检查项：根配置唯一性 / 路径穿越 / symlink / 体积 / 模型声明 /
    LoRA rank。
    """
    lim = limits or SubmissionLimits()
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise ValidationError(f"submission 目录不存在: {root}")
    if _is_symlink(root, symlink_probe):
        raise ValidationError("submission root: symlink 被禁止")

    report: Dict[str, object] = {
        "root": root, "files": 0, "total_bytes": 0,
        "violations": [], "models": set(), "lora_ranks": [],
        "skills": [], "sub_agents": [],
    }

    # --- 1. 根配置唯一性（§2.2） ---
    roots_found = [n for n in ROOT_CONFIG_NAMES
                   if os.path.isfile(os.path.join(root, n))]
    if len(roots_found) == 0:
        raise ValidationError("MissingRootConfigError: 根目录没有任何根配置")
    if len(roots_found) > 1:
        raise ValidationError(f"MultipleRootConfigsError: {roots_found}")
    report["root_config"] = roots_found[0]

    # --- 2. 遍历全部条目：symlink / 体积 / 路径 ---
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for d in list(dirnames):
            full = os.path.join(dirpath, d)
            if _is_symlink(full, symlink_probe):
                raise ValidationError(f"symlink 被禁止: {os.path.relpath(full, root)}")
            _check_relpath(os.path.relpath(full, root).replace("\\", "/"), "目录")
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            _check_relpath(rel, "文件")
            if _is_symlink(full, symlink_probe):
                raise ValidationError(f"symlink 被禁止: {rel}")
            suffix = os.path.splitext(rel)[1].lower()
            if suffix not in {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}:
                raise ValidationError(f"不支持的文件扩展名: {rel}")
            if suffix == ".safetensors":
                parts = rel.split("/")
                if len(parts) != 3 or parts[0] != "adapters" or parts[2] != "adapter_model.safetensors":
                    raise ValidationError(f"safetensors 必须位于 adapters/<name>/adapter_model.safetensors: {rel}")
            report["files"] += 1
            try:
                report["total_bytes"] += os.path.getsize(full)
            except OSError as exc:
                raise ValidationError(f"无法读取文件大小: {rel}") from exc
            if suffix in {".yaml", ".yml"}:
                _scan_yaml(full, rel, report, root, symlink_probe)
            elif suffix == ".json":
                _scan_adapter_config(full, report)
            if rel.startswith("skills/") and rel.endswith("SKILL.md"):
                report["skills"].append(rel)
            if rel.startswith("sub_agents/"):
                report["sub_agents"].append(rel)

    # --- 3. 体积（§2.4） ---
    total = int(report["total_bytes"])
    report["total_bytes"] = total
    report["total_gib"] = round(total / 2**30, 4)
    report["size_headroom_gib"] = round(
        (lim.max_total_size_bytes - total) / 2**30, 4)
    if total >= lim.max_total_size_bytes:
        raise ValidationError(
            f"解压后体积 {total} 字节 ≥ 上限 {lim.max_total_size_bytes}（3 GiB）")

    # --- 4. 单基座模型规则（§3.2） ---
    models = report["models"]
    report["models"] = sorted(models)
    if not models:
        raise ValidationError("缺少 model 声明: submission 必须声明一种允许的基座模型")
    if len(models) > 1:
        raise ValidationError(f"单一基座模型规则违反: 声明了 {sorted(models)}")
    if models:
        m = next(iter(models))
        if m not in lim.allowed_models:
            raise ValidationError(f"模型不在 ALLOWED_MODEL_NAMES 内: {m}")

    # --- 5. LoRA rank（§3.4） ---
    ranks = sorted(set(report["lora_ranks"]))
    report["lora_ranks"] = ranks
    if ranks and max(ranks) > lim.max_lora_rank:
        raise ValidationError(
            f"max_lora_rank 越界: {max(ranks)} > {lim.max_lora_rank}")

    report["ok"] = True
    return report


def _scan_adapter_config(path: str, report: Dict[str, object]) -> None:
    """从 `adapter_config.json` 里抽 LoRA rank（§3.4）。"""
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"无效 JSON: {path}") from exc
    if not isinstance(cfg, dict):
        return
    for key in ("r", "lora_r", "rank", "lora_rank"):
        if isinstance(cfg.get(key), int):
            report["lora_ranks"].append(cfg[key])
            return


def _scan_yaml(path: str, rel: str, report: Dict[str, object], root: str,
               symlink_probe: Optional[Callable[[str], bool]] = None) -> None:
    """安全解析相对 include 和 agent 引用，收集递归 model 声明。"""
    try:
        import yaml
    except ImportError as exc:
        raise ValidationError("validate 需要 PyYAML；请安装 atlas-swe 的依赖") from exc
    # Retain visited objects so included mappings keep their source directory,
    # and Python cannot recycle their ids during recursive reference loading.
    seen = {}

    def resolve(base, relative):
        _check_relpath(relative, "include/config_path")
        target = os.path.abspath(os.path.join(os.path.dirname(base), relative.replace("\\", "/")))
        if os.path.commonpath([root, os.path.realpath(target)]) != root:
            raise ValidationError(f"include/config_path 越出 submission root: {relative}")
        current = target
        while current != root:
            if _is_symlink(current, symlink_probe):
                raise ValidationError(f"symlink 被禁止: {relative}")
            current = os.path.dirname(current)
        return target

    def load(current, include_stack=(), config_stack=()):
        if current in include_stack or current in config_stack:
            raise ValidationError(f"include/config_path cycle 循环: {current}")
        if len(include_stack) > 10:
            raise ValidationError("!include depth 深度超过 10")
        if len(config_stack) > 50:
            raise ValidationError("config_path depth 深度超过 50")
        class Loader(yaml.SafeLoader):
            pass

        def include(loader, node):
            target = resolve(current, loader.construct_scalar(node))
            suffix = os.path.splitext(target)[1].lower()
            if suffix in {".yaml", ".yml"}:
                return load(target, (*include_stack, current), config_stack)
            if suffix not in {".md", ".txt"}:
                raise ValidationError("!include 只支持 YAML、Markdown 或文本")
            with open(target, encoding="utf-8") as stream:
                return stream.read()

        Loader.add_constructor("!include", include)
        with open(current, encoding="utf-8") as stream:
            doc = yaml.load(stream, Loader=Loader)
        def walk(node):
            if isinstance(node, (dict, list)):
                if id(node) in seen:
                    return
                seen[id(node)] = node
            if isinstance(node, dict):
                if "model" in node:
                    m = node["model"]
                    if not isinstance(m, str) or not m.strip():
                        raise ValidationError(f"无效 model 声明: {current}")
                    for prefix in ("openai/", "google/", "hosted_vllm/", "custom/"):
                        if m.startswith(prefix):
                            m = m[len(prefix):]
                            break
                    report["models"].add(m)
                if "config_path" in node:
                    ref = node["config_path"]
                    if not isinstance(ref, str):
                        raise ValidationError("config_path 必须是相对路径字符串")
                    target = resolve(current, ref)
                    if os.path.splitext(target)[1].lower() not in {".yaml", ".yml"}:
                        raise ValidationError("config_path 必须指向 YAML")
                    load(target, include_stack, (*config_stack, current))
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(doc)
        return doc

    try:
        load(path)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValidationError(f"无法解析 YAML/include {rel}: {exc}") from exc


def estimate_size(files: Sequence[Tuple[str, int]]) -> Dict[str, object]:
    """在打包前预估体积（路径 → 字节数）。"""
    total = sum(sz for _, sz in files)
    lim = SubmissionLimits()
    return {
        "n_files": len(files),
        "total_bytes": total,
        "total_gib": round(total / 2**30, 4),
        "fits_in_3gib": total < lim.max_total_size_bytes,
        "headroom_gib": round((lim.max_total_size_bytes - total) / 2**30, 4),
    }


# --------------------------------------------------------------------------
# 生成
# --------------------------------------------------------------------------
def emit_agent_yaml(model: str = DEFAULT_MODEL,
                    instructions: Optional[str] = None,
                    sub_agents: Optional[Sequence[str]] = None,
                    skills: Optional[Sequence[str]] = None,
                    tools: Optional[Sequence[str]] = None,
                    agent_class: str = "LlmAgent",
                    agent_name: str = "atlas_agent") -> Dict[str, object]:
    """构造根 `agent.yaml` 的内容（返回 dict，调用方自行序列化）。

    单基座模型规则由 `validate()` 复核；这里先做一次早期检查，
    让错误在生成阶段而不是提交阶段暴露。
    """
    if model not in ALLOWED_MODEL_NAMES:
        raise ValidationError(f"model 不在 ALLOWED_MODEL_NAMES: {model}")
    for s in (skills or ()):
        _check_relpath(s, "skills 条目")
    for s in (sub_agents or ()):
        _check_relpath(s, "sub_agents 条目")
    doc: Dict[str, object] = {
        "name": agent_name,
        "model": model,
        "agent_class": agent_class,
    }
    if instructions:
        doc["instruction"] = instructions
    if tools:
        doc["tools"] = list(tools)
    if sub_agents:
        doc["sub_agents"] = [{"config_path": p} for p in sub_agents]
    if skills:
        doc["skills"] = list(skills)
    return doc


def emit_sub_agent(name: str, model: str = DEFAULT_MODEL,
                   instructions: str = "",
                   tools: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """构造一个 `sub_agents/<name>.yaml`。"""
    _check_relpath(f"sub_agents/{name}.yaml", "sub_agent 文件名")
    return emit_agent_yaml(model=model, instructions=instructions,
                           tools=tools, agent_name=name)


def emit_skill(name: str, body: str,
               description: str = "") -> str:
    """构造 `skills/<name>/SKILL.md`。

    技能内容是纯 Markdown，**不含可执行 Python**（§2.1 声明式安全模型：
    编译器不执行任何竞争者 Python，也不 `importlib`）。

    `name` 必须是**单个目录名**（官方 SKILL.md 的 `name:` 字段就是这么用的），
    所以先单独校验它 —— 否则 `name="/abs"` 拼出来的
    `skills//abs/SKILL.md` 仍是相对路径，会绕过 `_check_relpath`。
    """
    if os.path.isabs(name) or name.startswith(("/", "\\")) \
            or re_drive(name) or ".." in str(name).replace("\\", "/").split("/") \
            or "/" in name or "\\" in name or name in ("", "."):
        raise ValidationError(f"skill 名必须是单个相对目录名: {name!r}")
    _check_relpath(f"skills/{name}/SKILL.md", "skill 路径")
    front = ["---", f"name: {name}"]
    if description:
        front.append(f"description: {description}")
    front.append("---")
    return "\n".join(front) + "\n\n" + body.strip() + "\n"


def write_submission(root: str, agent_yaml: Dict[str, object],
                     skills: Optional[Dict[str, str]] = None,
                     sub_agents: Optional[Dict[str, Dict[str, object]]] = None,
                     eval_config: Optional[EvalBudget] = None,
                     dry_run: bool = False) -> Dict[str, object]:
    """把生成结果写到 `root`，然后立刻跑一次 `validate()`。"""
    files: Dict[str, str] = {"agent.yaml": _to_yaml(agent_yaml)}
    for nm, md in (skills or {}).items():
        files[f"skills/{nm}/SKILL.md"] = md
    for nm, doc in (sub_agents or {}).items():
        files[f"sub_agents/{nm}.yaml"] = _to_yaml(doc)
    if eval_config and eval_config.to_dict():
        files["eval_config.yaml"] = _to_yaml({"evaluation": eval_config.to_dict()})

    if dry_run:
        return {"written": 0, "planned": sorted(files)}

    os.makedirs(root, exist_ok=True)
    for rel, content in files.items():
        _check_relpath(rel, "待写文件")
        full = os.path.join(root, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    report = validate(root)
    report["written"] = len(files)
    return report


def _to_yaml(doc: Dict[str, object]) -> str:
    """极简 YAML 序列化（避免对 pyyaml 的硬依赖）。"""
    try:
        import yaml
        return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
    except ImportError:
        lines = []

        def emit(node, indent=0):
            pad = "  " * indent
            if isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"{pad}{k}:")
                        emit(v, indent + 1)
                    else:
                        lines.append(f"{pad}{k}: {v}")
            elif isinstance(node, list):
                for it in node:
                    if isinstance(it, (dict, list)):
                        lines.append(f"{pad}-")
                        emit(it, indent + 1)
                    else:
                        lines.append(f"{pad}- {it}")
        emit(doc)
        return "\n".join(lines) + "\n"
