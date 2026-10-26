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
import importlib.metadata
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

BACKSLASH = chr(92)

def system_instruction(allow_run_command: bool) -> str:
    tools = ("read_file(filepath, start_line=None, end_line=None), edit_file(filepath, "
             "old_string, new_string), write_file(filepath, content), get_status(), "
             "submit_patch()")
    if allow_run_command:
        tools = ("run_command(command), " + tools +
                 ". run_command executes a shell command in /workspace with a 300s timeout; "
                 "use it to run targeted pytest checks before submitting")
    return (
        "You are a software engineering agent fixing one GitHub issue in a repository mounted at "
        "/workspace (the current working directory). Available tools: " + tools + ". "
        "Respond with exactly one "
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
    "{execution_rule}\n"
    "\n"
    "## Symbol occurrence index (identifiers from the problem statement)\n"
    "{symbols}\n"
    "{guidance_section}\n"
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

    def __init__(self, workspace: Path, tool_budget: int, time_budget_s: float, started: float,
                 allow_run_command: bool = False):
        self.workspace = workspace
        self.allow_run_command = allow_run_command
        self.edits = []
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
            self.edits.append(str(path.relative_to(self.workspace)))
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

    def run_command(self, command) -> dict:
        gate = self.budgeted()
        if gate:
            return gate
        self.tool_calls_used += 1
        try:
            if not isinstance(command, str) or not command.strip():
                raise ValueError('command must be a non-empty string')
            timeout = max(5.0, min(300.0, self.time_left()))
            proc = subprocess.run(['bash', '-lc', command], cwd=self.workspace,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {'status': 'error', 'error_type': 'TimeoutExceeded',
                    'error_message': 'command exceeded the timeout without ending the session'}
        except Exception as error:
            return self._err(type(error).__name__, str(error)[:300])
        stdout = (proc.stdout or '')[:5000]
        stderr = (proc.stderr or '')[:5000]
        if proc.returncode == 0:
            return self._ok(stdout=stdout, stderr=stderr, exit_code=0)
        return {'status': 'error', 'error_type': 'CommandError',
                'error_message': f'exit code {proc.returncode}',
                'details': {'stdout': stdout, 'stderr': stderr, 'exit_code': proc.returncode}}

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
        if self.allow_run_command:
            table['run_command'] = self.run_command
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

MAX_PROMPT_SHRINKS = 32


def prompt_token_count(tokenizer, messages):
    """Count the actual chat template, including generation-prefix tokens."""
    return len(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True))


def fit_prompt(messages, tokenizer, prompt_budget, note, strategy='truncate'):
    """Bounded compaction; retain instructions and label any truncated content.

    Even the initial issue or a single tool result can exceed the window. Never
    rely on character/token ratios or retry a compaction that made no progress.
    """
    messages = [dict(message) for message in messages]
    for attempt in range(MAX_PROMPT_SHRINKS + 1):
        count = prompt_token_count(tokenizer, messages)
        if count <= prompt_budget:
            return messages, count
        if attempt == MAX_PROMPT_SHRINKS:
            break
        if len(messages) > 3:
            status = {'role': 'user', 'content': '[Context compaction] ' + note}
            if strategy == 'state':
                messages = messages[:2] + [status]
            else:
                # Drop whole assistant/result pairs, never duplicate the prefix.
                tail = messages[2:][-2:] if len(messages) > 4 else []
                messages = messages[:2] + [status] + tail
                if len(messages) > 3 and attempt:
                    messages = messages[:3]
            continue
        candidates = [i for i in range(1, len(messages))
                      if len(messages[i]['content']) > 64]
        if not candidates:
            break
        index = max(candidates, key=lambda i: len(messages[i]['content']))
        content = messages[index]['content']
        keep = len(content) // 4
        messages[index]['content'] = (content[:keep] + '\n[truncated to fit context]\n'
                                       + content[-keep:])
    raise ValueError('prompt cannot fit token budget without removing system instructions')


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
    """Return request and completion separately: prompt IDs belong to the request."""
    request = chat_result[0] if isinstance(chat_result, list) else chat_result
    return request, request.outputs[0]


def gold_files(task: dict) -> list:
    files = sorted(set(re.findall(r'^\+\+\+ b/(\S+)', task.get('patch', '') or '', re.M)))
    return [f for f in files if not is_protected(f)]


def b0_rank(workspace: Path, problem: str, limit: int = 8) -> list:
    """B0 lexical proxy: Jaccard overlap between issue tokens and path-stem tokens."""
    stop = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'not', 'are', 'but'}
    q_tokens = {t.lower() for t in re.findall(r'[A-Za-z_][A-Za-z0-9_]{2,}', problem)
                if t.lower() not in stop}
    scored = []
    for path in sorted(workspace.rglob('*.py')):
        rel = path.relative_to(workspace).as_posix()
        if '.git' in path.parts:
            continue
        stems = set()
        for part in re.split(r'[\/]', rel)[:-1] + [Path(rel).stem]:
            stems.update(t.lower() for t in re.findall(r'[A-Za-z_][A-Za-z0-9_]{2,}', part))
        union = q_tokens | stems
        score = len(q_tokens & stems) / len(union) if union else 0.0
        scored.append((score, rel))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [rel for score, rel in scored[:limit] if score > 0] or         [rel for _, rel in scored[:limit]]


def build_localization_section(workspace: Path, task: dict, mode: str) -> str:
    if mode == 'oracle':
        files = gold_files(task)
        if not files:
            return ''
        return ('## Localization hints (answer-informed reference; not an implementable '
                'retrieval method)\n'
                + chr(10).join(f'- {f}' for f in files) + chr(10))
    if mode == 'b0':
        ranked = b0_rank(workspace, task.get('problem_statement', ''))
        if not ranked:
            return ''
        return ('## Localization hints (lexical filename-overlap proxy B0)\n'
                + chr(10).join(f'- {f}' for f in ranked) + chr(10))
    return ''


def verification_guidance_section(task: dict, mode: str) -> str:
    if mode == 'none':
        return ''
    files = sorted(set(re.findall(r'^\+\+\+ b/(\S+)', task.get('test_patch', '') or '', re.M)))
    if not files:
        return ''
    if mode == 'full':
        return ('## Verification guidance (full)\n'
                'The repository test files are: ' + ', '.join(files) + '.\n')
    fns = []
    for line in (task.get('test_patch', '') or '').splitlines():
        m = re.match(r'\+\s*def (test_\w+)\(', line)
        if m:
            fns.append(m.group(1))
    if not fns:
        return ''
    return ('## Verification guidance (selected)\n'
            'The test patch adds these tests, which must pass after your fix: '
            + ', '.join(fns) + '.\n')



def run_agent_loop(workspace: Path, task: dict, llm, sampling, budgets: dict,
                   output_dir: Path, log, allow_run_command: bool = False) -> dict:
    started = time.monotonic()
    executor = ToolExecutor(workspace, budgets['tool_calls'], budgets['time_minutes'] * 60, started,
                            allow_run_command=allow_run_command)
    trace_path = output_dir / 'trace.jsonl'

    def record(turn, **fields):
        entry = {'turn': turn, 'elapsed_seconds': round(time.monotonic() - started, 2), **fields}
        with trace_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
        log(f'  turn {turn}: ' + json.dumps({k: entry[k] for k in fields if k in (
            'action', 'tool', 'prompt_tokens', 'output_tokens', 'note')}, ensure_ascii=False))
        return entry

    mode = budgets.get('localization_mode', 'index')
    localization_section = build_localization_section(workspace, task, mode)
    guidance_section = verification_guidance_section(
        task, budgets.get('verification_guidance', 'none'))
    messages = [
        {'role': 'system', 'content': system_instruction(allow_run_command)},
        {'role': 'user', 'content': INITIAL_TEMPLATE.format(
            repo=task.get('repo', ''), problem=task.get('problem_statement', ''),
            execution_rule=('- run_command is available for targeted offline checks.'
                            if allow_run_command else
                            '- No command execution tool is available; reason from file contents.'),
            time_budget=budgets['time_minutes'], tool_budget=budgets['tool_calls'],
            symbols=symbol_index(workspace, task.get('problem_statement', '')),
            localization_section=localization_section,
            guidance_section=guidance_section,
            layout=workspace_layout(workspace))},
    ]
    turn = 0
    consecutive_nudges = 0
    totals = {'prompt_tokens': 0, 'output_tokens': 0}
    tokenizer = llm.get_tokenizer()
    max_model_len = budgets.get('max_model_len', 8192)
    max_tokens = sampling.max_tokens
    prompt_budget = max_model_len - max_tokens
    if prompt_budget <= 0:
        raise ValueError('max_tokens must be smaller than max_model_len')
    model_path = budgets.get('model_path') or getattr(
        getattr(getattr(llm, 'llm_engine', None), 'model_config', None), 'model', None)
    settings = {'max_model_len': max_model_len, 'max_tokens': max_tokens,
                'temperature': getattr(sampling, 'temperature', None),
                'prompt_token_budget': prompt_budget, 'allow_run_command': allow_run_command,
                **budgets}
    finish_reason = None
    termination = 'tool_budget'
    while time.monotonic() - started < budgets['time_minutes'] * 60:
        if executor.tool_calls_used >= budgets['tool_calls']:
            termination = 'tool_budget'
            break
        turn += 1
        viewed = '; '.join(f'{k} {v[-1]}' for k, v in
                           list(executor.files_viewed.items())[-8:]) or 'none'
        note = (f'{executor.tool_calls_used} tool calls; edits: '
                + (', '.join(executor.edits[-8:]) or 'none') + '; viewed: ' + viewed)
        try:
            # One smaller retry for template/engine disagreement, never an
            # unbounded outer-loop context recovery.
            for attempt in range(2):
                messages, estimated_tokens = fit_prompt(
                    messages, tokenizer, prompt_budget if not attempt else
                    min(prompt_budget - 256, estimated_tokens - 256), note,
                    budgets.get('compaction_strategy', 'truncate'))
                try:
                    request, out = extract_text(llm.chat(messages, sampling, use_tqdm=False))
                    break
                except Exception as error:
                    if attempt or 'context' not in str(error).lower():
                        raise
                    record(turn, action='compact', note='bounded context error retry')
        except Exception as error:
            record(turn, action='llm_error', note=str(error)[:300])
            termination = 'llm_error'
            break
        text = out.text or ''
        prompt_tokens = len(getattr(request, 'prompt_token_ids', None) or [])
        output_tokens = len(out.token_ids or [])
        finish_reason = getattr(out, 'finish_reason', None)
        totals['prompt_tokens'] += prompt_tokens
        totals['output_tokens'] += output_tokens
        call = parse_tool_call(text)
        if call is None:
            consecutive_nudges += 1
            record(turn, action='nudge', prompt_tokens=prompt_tokens, finish_reason=finish_reason,
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
               prompt_tokens=prompt_tokens, finish_reason=finish_reason,
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
        'schema_version': 5, 'model': str(model_path) if model_path is not None else None,
        'settings': settings, 'finish_reason': finish_reason,
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



def run_batch(args) -> int:
    """Direction 1 driver: run every task in the sample, resumable per task."""
    sample = json.loads(args.batch_file.read_text(encoding='utf-8'))
    tasks_index = {}
    for line in args.tasks_jsonl.read_text(encoding='utf-8').splitlines():
        if line.strip():
            record = json.loads(line)
            tasks_index[record['instance_id']] = record
    snapshot_dir = args.snapshot_dir
    batch_summary_path = args.output_dir / 'batch_summary.json'
    batch = json.loads(batch_summary_path.read_text(encoding='utf-8')) if batch_summary_path.exists() else {'tasks': {}}
    for entry in sample:
        iid = entry['instance_id']
        task_dir = args.output_dir / iid
        done = batch['tasks'].get(iid, {})
        if done.get('termination') in ('submit_patch', 'tool_budget', 'time_budget', 'no_tool_call', 'llm_error'):
            print(f'[skip] {iid} already terminated: {done.get("termination")}', flush=True)
            continue
        snapshot = snapshot_dir / f'{iid}.tgz'
        if not snapshot.is_file():
            batch['tasks'][iid] = {'fatal': 'snapshot missing'}
            continue
        task_dir.mkdir(parents=True, exist_ok=True)
        single = ['--snapshot', str(snapshot), '--tasks-jsonl', str(args.tasks_jsonl),
                  '--instance-id', iid, '--model-path', str(args.model_path),
                  '--output-dir', str(task_dir),
                  '--tool-budget', str(args.tool_budget),
                  '--time-minutes', str(args.time_minutes),
                  '--localization-mode', args.localization_mode,
                  '--compaction-strategy', args.compaction_strategy,
                  '--verification-guidance', args.verification_guidance,
                  '--max-model-len', str(args.max_model_len)]
        if args.allow_run_command:
            single.append('--allow-run-command')
        print(f'=== task {iid} ({entry["repo"]}) ===', flush=True)
        rc = main(single)
        summary = {}
        summary_path = task_dir / 'summary.json'
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                summary = {'fatal': 'summary unreadable'}
        batch['tasks'][iid] = {k: summary.get(k) for k in (
            'termination', 'turns', 'tool_calls_used', 'submit_called', 'patch_chars',
            'patch_files', 'totals', 'elapsed_seconds', 'error_type', 'fatal')}
        batch_summary_path.write_text(json.dumps(batch, ensure_ascii=False, indent=1) + chr(10), encoding='utf-8')
    print(json.dumps(batch, ensure_ascii=False, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--task-json', type=Path)
    parser.add_argument('--tasks-jsonl', type=Path)
    parser.add_argument('--instance-id', type=str)
    parser.add_argument('--batch-file', type=Path,
                        help='JSON list of {instance_id, repo, ...}; runs each task into <output>/<iid>/')
    parser.add_argument('--snapshot-dir', type=Path)
    parser.add_argument('--localization-mode', type=str, default='index',
                        choices=['index', 'oracle', 'b0'])
    parser.add_argument('--compaction-strategy', type=str, default='truncate',
                        choices=['truncate', 'state'])
    parser.add_argument('--verification-guidance', type=str, default='none',
                        choices=['none', 'selected', 'full'])
    parser.add_argument('--allow-run-command', action='store_true')
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tool-budget', type=int, default=24)
    parser.add_argument('--time-minutes', type=float, default=28.0)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.batch_file:
        if not (args.tasks_jsonl and args.snapshot_dir and args.model_path):
            parser.error('batch mode requires --tasks-jsonl, --snapshot-dir, --model-path')
        return run_batch(args)
    if not args.snapshot:
        parser.error('single-task mode requires --snapshot')

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
              'model': str(args.model_path),
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
                  'tensor_parallel_size': 1,
                  'max_model_len': args.max_model_len,
                  'gpu_memory_utilization': 0.80, 'enforce_eager': True,
                  'disable_log_stats': True,
                  'limit_mm_per_prompt': {'image': 0, 'video': 0, 'audio': 0}}
        result['runtime_config'] = config
        load_started = time.monotonic()
        llm = LLM(**config)
        result['load_seconds'] = round(time.monotonic() - load_started, 2)
        sampling = SamplingParams(temperature=0.0, max_tokens=1024)
        budgets = {'tool_calls': args.tool_budget, 'time_minutes': args.time_minutes,
                   'localization_mode': args.localization_mode,
                   'compaction_strategy': args.compaction_strategy,
                   'verification_guidance': args.verification_guidance,
                   'max_model_len': args.max_model_len,
                   'model_path': str(args.model_path)}
        allow_run_command = args.allow_run_command
        summary = run_agent_loop(args.output_dir / 'workspace', task, llm, sampling,
                                 budgets, args.output_dir, log,
                                 allow_run_command=allow_run_command)
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
