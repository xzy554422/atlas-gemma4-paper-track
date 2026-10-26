"""Local batch-mode smoke for agent_smoke_v5 with a fake LLM (no GPU, no vLLm import)."""
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('agent_m', ROOT / 'paper/scripts/agent_smoke_v5.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Fake vllm module so the runner's late import succeeds.
fake_vllm = types.ModuleType('vllm')


class FakeSamplingParams:
    def __init__(self, max_tokens=1024, **kwargs):
        self.max_tokens = max_tokens
        self.kwargs = kwargs


fake_vllm.SamplingParams = FakeSamplingParams


class FakeLLMClass:
    def __init__(self, **config):
        self.config = config


fake_vllm.LLM = FakeLLMClass
sys.modules['vllm'] = fake_vllm

out = Path(tempfile.mkdtemp())
(out / 'sample.json').write_text(json.dumps(
    [{'repo': 'psf/requests', 'patch_chars': 604, 'patch_files': 1,
      'instance_id': 'requests_7502'}]))


class Out:
    def __init__(self, text):
        self.text = text
        self.prompt_token_ids = [1]
        self.token_ids = [1]


class Resp:
    def __init__(self, text):
        self.outputs = [Out(text)]


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kw):
        text = ' '.join(str(m.get('content', '')) for m in messages)
        return list(range(len(text.split()))) if tokenize else text


class FakeLLM(FakeLLMClass):
    def get_tokenizer(self):
        return FakeTokenizer()

    def chat(self, messages, sampling, use_tqdm=False):
        return [Resp('TOOL_CALL: {"name": "submit_patch"}')]


# The runner imports LLM from vllm at call time, so the override must live on the
# fake module, not on the runner namespace.
fake_vllm.LLM = FakeLLM
rc = m.main(['--batch-file', str(out / 'sample.json'),
             '--tasks-jsonl', str(ROOT / 'data/tasks.jsonl'),
             '--snapshot-dir', str(ROOT / 'data/snapshots'),
             '--model-path', 'unused', '--output-dir', str(out),
             '--tool-budget', '4', '--time-minutes', '2'])
batch = json.loads((out / 'batch_summary.json').read_text(encoding='utf-8'))
task = batch['tasks']['requests_7502']
print('rc:', rc, '| termination:', task.get('termination'),
      '| patch_files:', task.get('patch_files'))
assert task.get('termination') == 'submit_patch', task
print('BATCH MODE OK')
