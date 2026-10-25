"""Bounded single-task agent smoke test (v5) for the Gemma 4 Developer Agent corpus.

Runs one task end-to-end on Kaggle GPU with the paper-track vLLM configuration
(source-level SM 7.5 tile guard + FP16 + tensor parallelism 1). The loop implements a
reduced version of the documented swegemma harness protocol (HARNESS_README.md
sections 5-6, 8.1): read_file / edit_file / write_file / get_status / submit_patch,
protected-file stripping, automatic unsubmitted-diff recovery. Execution tools
(run_command) and code-graph tools are intentionally not declared, and no official
ADK stack is used; this is a minimal re-implementation for a bounded data point.

Outputs (under --output-dir):
  agent_patch.diff   stripped unified diff vs the baseline commit (may be empty)
  trace.jsonl        one JSON object per turn
  summary.json       budgets, token totals, timing, termination reason

This establishes a bounded agent data point only; it is not a competition score.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

MODEL_NAME = 'gemma-4-e4b-it-qat-w4a16-ct'
BACKSLASH = chr(92)

SYSTEM_INSTRUCTION = (
    "You are a software engineering agent fixing one GitHub issue in a repository mounted at "
    "/workspace (the current working directory). Available tools: read_file(filepath, "
    "start_line=None, end_line=None), edit_file(filepath, old_string, new_string), "
    "write_file(filepath, content), get_status(), submit_patch(). Respond with exactly one "
    "tool call per turn: your reply may contain short reasoning, but its LAST line must be "
    "the tool call in exactly this format and nothing else on that line:\n"
    'TOOL_CALL: {"name": "read_file", "arguments": {"filepath": "src/module.py"}}\n'
    "Rules: keep reasoning under 3 sentences per turn. Never re-read a file or line "
    "range you have already viewed; act on what you have seen. When you know the fix, "
    "emit the edit_file tool call immediately; never write code blocks, diffs, or "
    "patches as plain text - changes happen only through tool calls. Prefer small "
    "targeted edits via edit_file with an exact unique old_string. When the fix is "
    "complete, call submit_patch. Do not modify files under tests/ or files named "
    "test_*.py; they are discarded at verification time."
)

INITIAL_TEMPLATE = (
    "You are evaluating a software engineering task for repository {repo}.\n"
    "\n"
    "Problem Statement:\n"
    "{problem}\n"
    "\n"
    "## Task Budget (session terminates when any budget is exhausted)\n"
    "- Time allowance: {time_budget} minutes\n"
    "- Tool calls allowance: {tool_budget} calls\n"
    "\n"
    "## Execution Environment Rules\n"
    "- File view limit: 150 lines and 10000 characters per read_file call.\n"
    "- The environment is offline. Do not attempt to install packages.\n"
    "- No command execution tool is available in this smoke configuration; reason from "
    "file contents.\n"
    "\n"
    "## Symbol occurrence index (identifiers from the problem statement)\n"
    "{symbols}\n"
    "\n"
    "## Workspace Layout (top levels)\n"
    "{layout}\n"
    "\n"
    "Begin by locating the relevant code with read_file, then fix the issue and call "
    "submit_patch when done."
)

NUDGES = (
    "Please continue. Emit exactly one tool call as the LAST line (TOOL_CALL: {...}) - "
    "not a code block. If you already know the edit, emit edit_file now; otherwise call "
    "submit_patch to finish.",
    "You have not emitted a tool call. Either continue with a tool call now, or call "
    "submit_patch to finish.",
)

PROTECTED_RE = re.compile(
    r"(^|/)(test_[^/]*\.py|[^/]*_test\.py|conftest\.py|pytest\.ini|pyproject\.toml|"
    r"tox\.ini|setup\.cfg|\.pytest\.ini|sitecustomize\.py|usercustomize\.py)$"
    r"|(^|/)tests?/"
)


def is_protected(path: str) -> bool:
    return bool(PROTECTED_RE.search(path.replace(BACKSLASH, '/')))


# --------------------------------------------------------------------------
# vLLM tile guard: byte-identical to model_preflight v3/v4 (verified separately).
# --------------------------------------------------------------------------

def _candidate_module_paths():
    import glob as _glob
    paths = []
    try:
        import importlib.util
        spec = importlib.util.find_spec('vllm.v1.attention.ops.triton_unified_attention')
        if spec and spec.origin:
            paths.append(Path(spec.origin))
    except Exception:
        pass
    for pattern in ('/usr/local/lib/python*/dist-packages/vllm/v1/attention/ops/triton_unified_attention.py',
                    '/usr/local/lib/python*/site-packages/vllm/v1/attention/ops/triton_unified_attention.py'):
        paths.extend(Path(p) for p in _glob.glob(pattern))
    seen, unique = set(), []
    for path in paths:
        key = str(path)
        if key not in seen and path.is_file():
            seen.add(key)
            unique.append(path)
    return unique


ANCHOR = ('    # Default behavior\n'
          '    if is_prefill:\n'
          '        return 32\n')
GUARD = ('    # SM75_TILE_GUARD: Turing cannot allocate 3*tile*head*2 bytes of shared\n'
         '    # memory for 512-wide heads (vLLM issue #38918). Fall back to tile 16.\n'
         '    if head_size > 128:\n'
         '        return 16\n')


def apply_sm75_tile_patch(extra_paths=None) -> dict:
    """Rewrite the installed vLLM source so spawned engine processes inherit the guard."""
    info = {'source_files': [], 'capability_major': None, 'applied_modules': [], 'verify': None}
    try:
        import torch
        if not torch.cuda.is_available():
            info['capability_major'] = 'no_cuda'
            return info
        major = torch.cuda.get_device_capability()[0]
        info['capability_major'] = major
        if major >= 8:
            info['skipped'] = 'capability >= 8.0'
            return info
    except Exception as error:
        info['capability_major'] = 'error: ' + str(error)
        return info

    paths = list(extra_paths or []) + _candidate_module_paths()
    for path in paths:
        try:
            text = path.read_text(encoding='utf-8')
        except OSError as error:
            info['source_files'].append({'path': str(path), 'status': 'unreadable: ' + str(error)})
            continue
        if 'SM75_TILE_GUARD' in text:
            info['source_files'].append({'path': str(path), 'status': 'already_patched'})
            continue
        if ANCHOR not in text:
            info['source_files'].append({'path': str(path), 'status': 'anchor_not_found'})
            continue
        patched = text.replace(ANCHOR, GUARD + ANCHOR, 1)
        try:
            path.write_text(patched, encoding='utf-8')
        except OSError as error:
            info['source_files'].append({'path': str(path), 'status': 'unwritable: ' + str(error)})
            continue
        info['source_files'].append({
            'path': str(path), 'status': 'patched',
            'sha256_before': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'sha256_after': hashlib.sha256(patched.encode('utf-8')).hexdigest(),
        })

    # Also patch any already-imported copy so the parent process stays consistent.
    import importlib
    for name in ('vllm.v1.attention.ops.triton_unified_attention',
                 'vllm.attention.ops.triton_unified_attention',
                 'vllm.v1.attention.ops.triton_unified_attention_diffkv'):
        try:
            module = importlib.import_module(name)
            original = module._get_tile_size
        except Exception:
            continue
        def patched_fn(*args, _original=original, **kwargs):
            if args and isinstance(args[0], int) and args[0] > 128:
                return 16
            return _original(*args, **kwargs)
        module._get_tile_size = patched_fn
        info['applied_modules'].append(name)
    if info['applied_modules']:
        try:
            first = importlib.import_module(info['applied_modules'][0])
            info['verify'] = {'head512_decode': first._get_tile_size(512, 0, 2, False),
                              'head512_prefill': first._get_tile_size(512, 0, 2, True)}
        except Exception as error:
            info['verify'] = 'error: ' + str(error)
    return info


# --------------------------------------------------------------------------
# Workspace tools (reduced harness protocol)
# --------------------------------------------------------------------------

class ToolExecutor:
    """Implements the reduced tool set against the extracted workspace."""

    def __init__(self, workspace: Path, tool_budget: int, time_budget_s: float, started: float):
        self.workspace = workspace
        self.tool_calls_used = 0
        self.tool_budget = tool_budget
        self.time_budget_s = time_budget_s
        self.started = started
        self.patch_submitted = False
        self.submitted_patch = ''
        self.trace_file_tool_errors = 0
        self.files_viewed = {}
        self.last_git_stderr = ''

    # -- helpers ----------------------------------------------------------
    def resolve(self, filepath: str) -> Path:
        if not isinstance(filepath, str) or not filepath.strip():
            raise ValueError('filepath must be a non-empty string')
        rel = Path(filepath.replace(BACKSLASH, '/'))
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('filepath must be relative and stay inside the workspace')
        return self.workspace / rel

    def time_left(self) -> float:
        return self.time_budget_s - (time.monotonic() - self.started)

    @staticmethod
    def _ok(**fields) -> dict:
        return {'status': 'ok', **fields}

    @staticmethod
    def _err(error_type: str, message: str) -> dict:
        return {'status': 'error', 'error_type': error_type, 'error_message': message}

    def budgeted(self, count: bool = True):
        if count and self.tool_calls_used >= self.tool_budget:
            return self._err('BudgetExceeded', 'tool call budget exhausted')
        if self.time_left() <= 0:
            return self._err('BudgetExceeded', 'time budget exhausted')
        return None

    # -- tools ------------------------------------------------------------
    def read_file(self, filepath, start_line=None, end_line=None) -> dict:
        gate = self.budgeted()
        if gate:
            return gate
        self.tool_calls_used += 1
        try:
            path = self.resolve(filepath)
            text = path.read_text(encoding='utf-8', errors='replace')
        except Exception as error:
            self.trace_file_tool_errors += 1
            return self._err(type(error).__name__, str(error)[:300])
        lines = text.split('\n')
        start = int(start_line) if start_line else 1
        end = int(end_line) if end_line else min(start + 149, len(lines))
        start = max(1, start)
        end = max(start, min(end, start + 149, len(lines)))
        chunk = '\n'.join(lines[start - 1:end])[:10000]
        rel = str(path.relative_to(self.workspace))
        self.files_viewed.setdefault(rel, []).append([start, end])
        return self._ok(filepath=rel, content=chunk,
                        start_line=start, end_line=end, total_lines=len(lines),
                        is_truncated=(end < len(lines)) or len(chunk) == 10000)

    def edit_file(self, filepath, old_string, new_string, allow_multiple=False) -> dict:
        gate = self.budgeted()
        if gate:
            return gate
        self.tool_calls_used += 1
        try:
            if not isinstance(old_string, str) or old_string == '':
                raise ValueError('old_string must be a non-empty string')
            path = self.resolve(filepath)
            text = path.read_text(encoding='utf-8', errors='replace')
            if not text:
                raise ValueError('file is empty')
            norm_old = old_string.replace('\r\n', '\n')
            count = text.count(norm_old)
            if count == 0:
                raise ValueError('old_string not found in file')
            if count > 1 and not allow_multiple:
                raise ValueError(f'old_string matches {count} locations; provide a larger unique block')
            new_text = text.replace(norm_old, new_string.replace('\r\n', '\n'),
                                    -1 if allow_multiple else 1)
            path.write_text(new_text, encoding='utf-8')
            diff = ''.join(difflib.unified_diff(
                text.splitlines(True), new_text.splitlines(True),
                fromfile=f'a/{path.relative_to(self.workspace)}',
                tofile=f'b/{path.relative_to(self.workspace)}'))[:5000]
            return self._ok(filepath=str(path.relative_to(self.workspace)),
                            occurrences=count if allow_multiple else 1, diff=diff)
        except Exception as error:
            self.trace_file_tool_errors += 1
            return self._err(type(error).__name__, str(error)[:300])

    def write_file(self, filepath, content) -> dict:
        gate = self.budgeted()
        if gate:
            return gate
        self.tool_calls_used += 1
        try:
            if is_protected(str(filepath)):
                raise ValueError('protected test/config files cannot be written')
            path = self.resolve(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding='utf-8')
            return self._ok(filepath=str(path.relative_to(self.workspace)), size=path.stat().st_size)
        except Exception as error:
            self.trace_file_tool_errors += 1
            return self._err(type(error).__name__, str(error)[:300])

    def get_status(self) -> dict:
        return self._ok(
            tool_calls_used=self.tool_calls_used, patch_submitted=self.patch_submitted,
            tool_calls_remaining=max(0, self.tool_budget - self.tool_calls_used),
            time_seconds_remaining=round(self.time_left(), 1))

    def submit_patch(self) -> dict:
        patch = self.extract_patch()
        self.submitted_patch = patch
        self.patch_submitted = True
        return self._ok(patch_size=len(patch),
                        files_changed=patch.count('diff --git '))

    # -- patch extraction -------------------------------------------------
    def extract_patch(self) -> str:
        def run(args):
            return subprocess.run(args, cwd=self.workspace, capture_output=True, text=True)
        run(['git', 'add', '-A'])
        run(['git', 'add', '-N', '.'])
        proc = run(['git', 'diff', '--binary', 'SWEGEMMA_BASELINE'])
        self.last_git_stderr = (proc.stderr or '')[:400]
        return strip_protected_files(proc.stdout or '')

    def dispatch(self, name, arguments) -> dict:
        table = {
            'read_file': self.read_file, 'edit_file': self.edit_file,
            'write_file': self.write_file, 'get_status': self.get_status,
            'submit_patch': self.submit_patch,
        }
        if not isinstance(name, str) or name not in table:
            return self._err('UnknownTool', f'unknown tool {name!r}')
        arguments = arguments if isinstance(arguments, dict) else {}
        try:
            return table[name](**arguments)
        except TypeError as error:
            return self._err('BadArguments', str(error)[:300])


def strip_protected_files(diff_text: str) -> str:
    """File-level equivalent of the harness protected-file stripping."""
    if not diff_text:
        return ''
    sections = re.split(r'(?m)^diff --git ', diff_text)
    head, blocks = sections[0], sections[1:]
    kept = []
    for block in blocks:
        match = re.search(r'^\+\+\+ b/(\S+)', block, re.M)
        target = match.group(1) if match else ''
        if target == '/dev/null':
            match_del = re.search(r'^--- a/(\S+)', block, re.M)
            target = match_del.group(1) if match_del else ''
        if not is_protected(target):
            kept.append('diff --git ' + block)
    return head + ''.join(kept)


TOOL_CALL_RE = re.compile(r'TOOL_CALL:\s*(\{.*\})')

# Rough context budget in characters (~4 chars/token) before older turns are compacted.
# The 8,192-token window minus the 768-token completion allowance leaves ~7,400 tokens,
# so 24,000 characters keeps the templated prompt safely inside it.
CONTEXT_CHAR_LIMIT = 24000


def compact_messages(messages, note):
    """Keep system + initial user + a status note + the last two turns."""
    system, first = messages[0], messages[1]
    status = {'role': 'user', 'content': '[Context compaction] Earlier turns were omitted '
                                          'to fit the context window. Status: ' + note}
    return [system, first, status] + messages[-4:]


def context_chars(messages):
    return sum(len(m.get('content') or '') for m in messages)


def parse_tool_call(text: str):
    matches = TOOL_CALL_RE.findall(text or '')
    if not matches:
        return None
    raw = matches[-1].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {'name': None, 'arguments': {}, 'parse_error': raw[:200]}
    if not isinstance(payload, dict):
        return {'name': None, 'arguments': {}, 'parse_error': raw[:200]}
    return {'name': payload.get('name'), 'arguments': payload.get('arguments') or {}}


def workspace_layout(workspace: Path, limit: int = 60) -> str:
    skip = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.idea', '.vscode'}
    entries = []
    for path in sorted(workspace.rglob('*')):
        rel = path.relative_to(workspace)
        if any(part in skip or part.startswith('.') for part in rel.parts):
            continue
        if len(rel.parts) > 3:
            continue
        entries.append(('dir ' if path.is_dir() else 'file ') + rel.as_posix())
        if len(entries) >= limit:
            break
    return '\n'.join(entries) or '(empty)'


def symbol_index(workspace: Path, problem: str, limit: int = 14) -> str:
    """Lightweight stand-in for the harness code-intelligence tools: list lines that
    define or use identifiers mentioned in the problem statement. Definition lines take
    priority; docs/examples/benchmarks/tests trees are skipped."""
    tokens = [t for t in re.findall(r'[A-Za-z_][A-Za-z0-9_]{3,}', problem)
              if t.lower() not in {
                  'that', 'with', 'from', 'this', 'when', 'based', 'their', 'them',
                  'using', 'also', 'into', 'only', 'over', 'under', 'than', 'then',
                  'have', 'been', 'more', 'some', 'such', 'each', 'which', 'while',
                  'file', 'files', 'wrappers', 'wrapper', 'detection', 'error'}]
    seen, def_rows, use_rows = set(), [], []
    for path in sorted(workspace.rglob('*.py')):
        rel = path.relative_to(workspace).as_posix()
        parts = path.parts
        if (is_protected(rel) or '.git' in parts
                or parts[0] in {'docs', 'doc', 'examples', 'benchmarks', 'tests', 'test'}):
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            for token in tokens:
                if token not in line or (rel, lineno) in seen:
                    continue
                seen.add((rel, lineno))
                entry = f'{rel}:{lineno}: {stripped[:110]}'
                is_def = bool(re.match(
                    r'(?:async\s+)?def\s+' + re.escape(token) + r'\b|class\s+'
                    + re.escape(token) + r'\b', stripped))
                if is_def and len(def_rows) < limit:
                    def_rows.append(entry)
                elif not is_def and len(use_rows) < limit:
                    use_rows.append(entry)
                break
            if len(def_rows) >= limit and len(use_rows) >= limit:
                break
        if len(def_rows) >= limit and len(use_rows) >= limit:
            break
    lines = (def_rows + use_rows)[:limit]
    return chr(10).join(lines)


def extract_text(chat_result):
    if isinstance(chat_result, list):
        return chat_result[0].outputs[0]
    return chat_result.outputs[0]


def run_agent_loop(workspace: Path, task: dict, llm, sampling, budgets: dict,
                   output_dir: Path, log) -> dict:
    started = time.monotonic()
    executor = ToolExecutor(workspace, budgets['tool_calls'], budgets['time_minutes'] * 60, started)
    trace_path = output_dir / 'trace.jsonl'

    def record(turn, **fields):
        entry = {'turn': turn, 'elapsed_seconds': round(time.monotonic() - started, 2), **fields}
        with trace_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
        log(f'  turn {turn}: ' + json.dumps({k: entry[k] for k in fields if k in (
            'action', 'tool', 'prompt_tokens', 'output_tokens', 'note')}, ensure_ascii=False))
        return entry

    messages = [
        {'role': 'system', 'content': SYSTEM_INSTRUCTION},
        {'role': 'user', 'content': INITIAL_TEMPLATE.format(
            repo=task.get('repo', ''), problem=task.get('problem_statement', ''),
            time_budget=budgets['time_minutes'], tool_budget=budgets['tool_calls'],
            symbols=symbol_index(workspace, task.get('problem_statement', '')),
            layout=workspace_layout(workspace))},
    ]
    turn = 0
    consecutive_nudges = 0
    totals = {'prompt_tokens': 0, 'output_tokens': 0}
    termination = 'tool_budget'
    while time.monotonic() - started < budgets['time_minutes'] * 60:
        if executor.tool_calls_used >= budgets['tool_calls']:
            termination = 'tool_budget'
            break
        turn += 1
        if context_chars(messages) > CONTEXT_CHAR_LIMIT:
            viewed = '; '.join(f'{k} {v}' for k, v in executor.files_viewed.items()) or 'none'
            note = (f'{executor.tool_calls_used}/{budgets["tool_calls"]} tool calls used; '
                    f'patch_submitted={executor.patch_submitted}; already viewed: {viewed}. '
                    "Do not re-read these; proceed to edit_file or submit_patch.")
            before = context_chars(messages)
            messages = compact_messages(messages, note)
            record(turn, action='compact', note=f'{before} -> {context_chars(messages)} chars')
            turn += 1
        try:
            out = extract_text(llm.chat(messages, sampling, use_tqdm=False))
        except Exception as error:
            if 'context' in str(error).lower() and context_chars(messages) > 8000:
                messages = compact_messages(messages, 'context error recovery')
                record(turn, action='compact', note='context error recovery')
                continue
            record(turn, action='llm_error', note=str(error)[:300])
            termination = 'llm_error'
            break
        text = out.text or ''
        totals['prompt_tokens'] += len(getattr(out, 'prompt_token_ids', []) or [])
        totals['output_tokens'] += len(out.token_ids or [])
        call = parse_tool_call(text)
        if call is None:
            consecutive_nudges += 1
            record(turn, action='nudge', prompt_tokens=len(getattr(out, 'prompt_token_ids', []) or []),
                   output_tokens=len(out.token_ids or []), text_head=text[:400])
            if consecutive_nudges > len(NUDGES):
                termination = 'no_tool_call'
                break
            messages.append({'role': 'assistant', 'content': text[:800]})
            messages.append({'role': 'user', 'content': NUDGES[min(consecutive_nudges, len(NUDGES)) - 1]})
            continue
        consecutive_nudges = 0
        if call.get('name') is None:
            result = executor._err('BadToolCall', 'unparseable tool call: ' + str(call.get('parse_error')))
            action = 'parse_error'
        else:
            result = executor.dispatch(call['name'], call['arguments'])
            action = 'tool_call'
        record(turn, action=action, tool=call.get('name'),
               prompt_tokens=len(getattr(out, 'prompt_token_ids', []) or []),
               output_tokens=len(out.token_ids or []),
               text_head=text[:1500], tool_result_head=json.dumps(result, ensure_ascii=False)[:600])
        messages.append({'role': 'assistant', 'content': text[-800:]})
        if call.get('name') == 'submit_patch':
            termination = 'submit_patch'
            break
        messages.append({'role': 'user', 'content': 'TOOL_RESULT: ' + json.dumps(result, ensure_ascii=False)[:6000]})
    else:
        termination = 'time_budget'

    patch = executor.submitted_patch or executor.extract_patch()
    summary = {
        'schema_version': 5, 'model': MODEL_NAME,
        'task': task.get('instance_id'), 'repo': task.get('repo'),
        'python': platform.python_version(), 'platform': platform.platform(),
        'termination': termination, 'turns': turn,
        'tool_calls_used': executor.tool_calls_used,
        'submit_called': executor.patch_submitted,
        'patch_chars': len(patch), 'patch_files': patch.count('diff --git '),
        'file_tool_errors': executor.trace_file_tool_errors,
        'git_stderr': executor.last_git_stderr,
        'totals': totals,
        'elapsed_seconds': round(time.monotonic() - started, 2),
        'configuration_scope': 'single-task bounded agent smoke, not a competition score',
        'agent_status': 'ran' if turn else 'not_run',
    }
    (output_dir / 'agent_patch.diff').write_text(patch, encoding='utf-8')
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def prepare_workspace(snapshot_tgz: Path, workspace: Path) -> dict:
    if workspace.exists():
        import shutil
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    with tarfile.open(snapshot_tgz, 'r:gz') as tar:
        tar.extractall(workspace)
    inner = list(workspace.iterdir())
    root = inner[0] if len(inner) == 1 and inner[0].is_dir() else workspace
    if root != workspace:
        import shutil as _sh
        for child in list(root.iterdir()):
            _sh.move(str(child), str(workspace / child.name))
        root.rmdir()
    def run(args):
        return subprocess.run(args, cwd=workspace, capture_output=True, text=True)
    # Kaggle kernels trip git's dubious-ownership check for extracted snapshots;
    # allow the workspace globally or every later git command fails silently.
    subprocess.run(['git', 'config', '--global', '--add', 'safe.directory', '*'],
                   capture_output=True, text=True)
    run(['git', 'add', '-A'])
    commit = run(['git', '-c', 'user.email=smoke@local', '-c', 'user.name=smoke',
                  'commit', '-m', 'SWEGEMMA_BASELINE', '--allow-empty'])
    run(['git', 'branch', '-f', 'SWEGEMMA_BASELINE', 'HEAD'])
    head = run(['git', 'rev-parse', 'HEAD'])
    check = run(['git', 'rev-parse', 'SWEGEMMA_BASELINE'])
    head_sha = (head.stdout or '').strip()
    ref_sha = (check.stdout or '').strip()
    if not head_sha or head_sha != ref_sha:
        raise RuntimeError('baseline commit failed: '
                           + (commit.stderr or commit.stdout or 'empty')[:300])
    return {'baseline_commit': head_sha[:12], 'baseline_ref_ok': True,
            'commit_output': (commit.stdout or commit.stderr or '')[:200]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--task-json', type=Path)
    parser.add_argument('--tasks-jsonl', type=Path)
    parser.add_argument('--instance-id', type=str)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tool-budget', type=int, default=24)
    parser.add_argument('--time-minutes', type=float, default=28.0)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def log(message):
        print(message, flush=True)

    if args.task_json:
        task = json.loads(args.task_json.read_text(encoding='utf-8'))
    else:
        match = [json.loads(line) for line in args.tasks_jsonl.read_text(encoding='utf-8').splitlines()
                 if json.loads(line).get('instance_id') == args.instance_id]
        if not match:
            raise SystemExit(f'instance {args.instance_id} not found in {args.tasks_jsonl}')
        task = match[0]
    prep = prepare_workspace(args.snapshot, args.output_dir / 'workspace')
    log(f'workspace prepared: {prep}')
    result = {'schema_version': 5, 'task': task.get('instance_id'), 'workspace': prep,
              'python': platform.python_version(), 'platform': platform.platform(),
              'gpus': [], 'packages': {}, 'agent_status': 'not_run'}
    for package in ('torch', 'vllm', 'transformers'):
        try:
            result['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][package] = None
    try:
        import torch
        if torch.cuda.is_available():
            result['gpus'] = [{'name': torch.cuda.get_device_name(i),
                               'capability': list(torch.cuda.get_device_capability(i))}
                              for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    result['sm75_tile_patch'] = apply_sm75_tile_patch()
    (args.output_dir / 'summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    try:
        if not result['gpus']:
            raise RuntimeError('No CUDA GPU available.')
        from vllm import LLM, SamplingParams
        config = {'model': str(args.model_path), 'dtype': 'half',
                  'tensor_parallel_size': 1, 'max_model_len': 8192,
                  'gpu_memory_utilization': 0.80, 'enforce_eager': True,
                  'disable_log_stats': True,
                  'limit_mm_per_prompt': {'image': 0, 'video': 0, 'audio': 0}}
        result['runtime_config'] = config
        load_started = time.monotonic()
        llm = LLM(**config)
        result['load_seconds'] = round(time.monotonic() - load_started, 2)
        sampling = SamplingParams(temperature=0.0, max_tokens=1024)
        budgets = {'tool_calls': args.tool_budget, 'time_minutes': args.time_minutes}
        summary = run_agent_loop(args.output_dir / 'workspace', task, llm, sampling,
                                 budgets, args.output_dir, log)
        result.update(summary)
        result['total_elapsed_seconds'] = round(time.monotonic() - started, 2)
    except Exception as error:
        result['error_type'] = type(error).__name__
        result['error'] = str(error)[:1200]
    (args.output_dir / 'summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result.get(k) for k in (
        'task', 'termination', 'turns', 'tool_calls_used', 'patch_chars',
        'patch_files', 'totals', 'error_type')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
