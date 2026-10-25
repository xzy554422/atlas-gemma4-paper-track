"""Local stub test for the agent smoke runner: tools, parsing, patch extraction.

Runs the real agent_smoke_v5 logic against a fake LLM with a scripted conversation and
a real extraction of the requests_7502 snapshot. No GPU, no vLLM, no network.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SNAPSHOT = ROOT / 'data' / 'snapshots' / 'requests_7502.tgz'
TASKS = ROOT / 'data' / 'tasks.jsonl'


def load_module():
    spec = importlib.util.spec_from_file_location(
        'agent_smoke_v5', ROOT / 'paper' / 'scripts' / 'agent_smoke_v5.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeOutput:
    def __init__(self, text, prompt_tokens=100, output_tokens=20):
        self.text = text
        self.prompt_token_ids = list(range(prompt_tokens))
        self.token_ids = list(range(output_tokens))


class FakeResponse:
    def __init__(self, output):
        self.outputs = [output]


class FakeLLM:
    """Scripted responses; asserts the message list grows as expected."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat(self, messages, sampling, use_tqdm=False):
        self.calls += 1
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        assert 'TOOL_CALL:' in messages[0]['content']
        text = self.script.pop(0) if self.script else 'TOOL_CALL: {"name": "submit_patch"}'
        return [FakeResponse(FakeOutput(text))]


def main() -> int:
    module = load_module()
    assert module.patch_section_check() if hasattr(module, 'patch_section_check') else True

    # Parsing
    assert module.parse_tool_call('thinking...\nTOOL_CALL: {"name": "read_file", "arguments": {"filepath": "src/x.py"}}') == {
        'name': 'read_file', 'arguments': {'filepath': 'src/x.py'}}
    assert module.parse_tool_call('no call here') is None
    # No braces at all -> treated as "no tool call" (nudge path), like an unclosed tag.
    assert module.parse_tool_call('TOOL_CALL: not json') is None
    # Braces but invalid JSON -> structured parse error fed back to the model.
    bad = module.parse_tool_call('TOOL_CALL: {name: read_file}')
    assert bad['name'] is None and bad['parse_error']

    # Protected-file stripping
    diff = (
        'diff --git a/src/requests/models.py b/src/requests/models.py\n'
        '--- a/src/requests/models.py\n+++ b/src/requests/models.py\n@@ -1 +1 @@\n-a\n+b\n'
        'diff --git a/tests/test_requests.py b/tests/test_requests.py\n'
        '--- a/tests/test_requests.py\n+++ b/tests/test_requests.py\n@@ -1 +1 @@\n-x\n+y\n'
        'diff --git a/notes.md b/notes.md\n--- a/notes.md\n+++ /dev/null\n'
    )
    stripped = module.strip_protected_files(diff)
    assert 'models.py' in stripped and 'test_requests.py' not in stripped
    assert 'notes.md' in stripped  # deletions of non-protected files survive
    print('parsing + stripping OK')

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        ws = out / 'workspace'
        prep = module.prepare_workspace(SNAPSHOT, ws)
        assert prep['baseline_commit'], prep
        assert (ws / 'src' / 'requests' / 'models.py').is_file()
        assert not (ws / 'requests_7502').is_dir(), 'snapshot top-level dir not flattened'

        task = next(json.loads(line) for line in TASKS.open(encoding='utf-8')
                    if json.loads(line)['instance_id'] == 'requests_7502')

        # Scripted conversation: read -> bad call -> edit -> submit.
        script = [
            'Looking at models.py.\nTOOL_CALL: {"name": "read_file", "arguments": {"filepath": "src/requests/models.py", "start_line": 230, "end_line": 245}}',
            'TOOL_CALL: {"name": "edit_file", "arguments": {"filepath": "src/requests/models.py", "old_string": "THIS IS NOT PRESENT"}}',
            'TOOL_CALL: {"name": "edit_file", "arguments": {"filepath": "src/requests/models.py", "old_string": "elif isinstance(fp, _SupportsRead):  # defensive check for untyped callers", "new_string": "elif isinstance(fp, _SupportsRead) or hasattr(fp, \'read\'):"}}',
            'Done.\nTOOL_CALL: {"name": "submit_patch"}',
        ]
        llm = FakeLLM(script)
        budgets = {'tool_calls': 24, 'time_minutes': 5}
        summary = module.run_agent_loop(ws, task, llm, None, budgets, out, log=print)

        assert summary['termination'] == 'submit_patch', summary
        assert summary['submit_called'] is True
        # submit_patch never counts; a call with missing required arguments is rejected
        # at dispatch (TypeError) before budget gating, mirroring harness tool-error
        # recovery, so only read + good edit count here.
        assert summary['tool_calls_used'] == 2
        assert summary['turns'] == 4
        assert summary['totals']['output_tokens'] > 0

        patch = (out / 'agent_patch.diff').read_text(encoding='utf-8')
        assert 'models.py' in patch and 'hasattr' in patch
        assert 'diff --git' in patch

        trace_lines = (out / 'trace.jsonl').read_text(encoding='utf-8').strip().splitlines()
        entries = [json.loads(line) for line in trace_lines]
        # Turn 2's edit is rejected at dispatch for a missing new_string argument.
        assert 'missing 1 required positional argument' in entries[1]['tool_result_head']
        assert entries[1]['action'] == 'tool_call' and entries[1]['tool'] == 'edit_file'
        assert len(entries) == 4
        print('summary:', json.dumps({k: summary[k] for k in (
            'termination', 'turns', 'tool_calls_used', 'patch_chars')}, ensure_ascii=False))

        # Nudge path: a text-only turn then a submit.
        out2 = out / 'run2'
        out2.mkdir()
        prep2 = module.prepare_workspace(SNAPSHOT, out2 / 'workspace')
        script2 = ['I will start soon.', 'TOOL_CALL: {"name": "submit_patch"}']
        summary2 = module.run_agent_loop(out2 / 'workspace', task, FakeLLM(script2), None,
                                         budgets, out2, log=lambda *_: None)
        # One nudge turn for the text-only response, then the submit turn.
        assert summary2['turns'] == 2 and summary2['termination'] == 'submit_patch'
        trace2 = [json.loads(line) for line in (out2 / 'trace.jsonl').read_text().splitlines()]
        assert trace2[0]['action'] == 'nudge'

    print('AGENT SMOKE LOGIC OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
