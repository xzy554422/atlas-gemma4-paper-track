"""Generate model_agent_smoke_v5.ipynb: single-task bounded agent run on T4."""
import ast
import json
from pathlib import Path

V4 = Path('paper/notebooks/model_preflight_v4.ipynb')
RUNNER = Path('paper/scripts/agent_smoke_v5.py')
OUT = Path('paper/notebooks/model_agent_smoke_v5.ipynb')


def build_runner(probe: str) -> str:
    lines = [
        "from pathlib import Path",
        "import json, os, subprocess, sys, time",
        "WORK = Path('/kaggle/working')",
        "WORK.mkdir(exist_ok=True)",
        "START = time.monotonic()",
        "(WORK / 'agent_smoke_v5.py').write_text(" + json.dumps(probe) + ", encoding='utf-8')",
        "MODEL = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2')",
        "COMP = Path('/kaggle/input/competitions/gemma-4-developer-agent')",
        "SNAP = COMP / 'snapshots' / 'requests_7502.tgz'",
        "TASKS = COMP / 'tasks.jsonl'",
        "listing = {str(q): q.stat().st_size for q in Path('/kaggle/input').rglob('*') if q.is_file()}",
        "print('INPUT LISTING HEAD:', json.dumps(sorted(listing)[:8]))",
        "assert SNAP.exists() and TASKS.exists(), f'competition inputs missing; have {sorted(listing)[:20]}'",
        "env = os.environ.copy()",
        "env.update(CUDA_VISIBLE_DEVICES='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',",
        "           TRANSFORMERS_NO_TF='1', TRANSFORMERS_NO_FLAX='1',",
        "           VLLM_NO_USAGE_STATS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')",
        "if installation['status'] != 'passed':",
        "    result = {'agent_status': 'not_run', 'reason': 'runtime_installation_failed'}",
        "else:",
        "    with (WORK / 'agent.log').open('w') as log:",
        "        try:",
        "            process = subprocess.run(",
        "                [sys.executable, str(WORK / 'agent_smoke_v5.py'),",
        "                 '--snapshot', str(SNAP), '--tasks-jsonl', str(TASKS),",
        "                 '--instance-id', 'requests_7502',",
        "                 '--model-path', str(MODEL), '--output-dir', str(WORK)],",
        "                env=env, stdout=log, stderr=subprocess.STDOUT,",
        "                timeout=2700, start_new_session=True)",
        "            if (WORK / 'summary.json').exists():",
        "                result = json.loads((WORK / 'summary.json').read_text())",
        "            else:",
        "                result = {'agent_status': 'failed', 'exit_code': process.returncode,",
        "                          'reason': 'worker_exited_without_result'}",
        "        except subprocess.TimeoutExpired:",
        "            result = {'agent_status': 'timeout', 'limit_seconds': 2700}",
        "print((WORK / 'agent.log').read_text(errors='replace')[-4000:])",
        "result['notebook_elapsed_seconds'] = round(time.monotonic() - START, 3)",
        "(WORK / 'summary.json').write_text(json.dumps(result, indent=2))",
        "print(json.dumps(result, indent=2))",
    ]
    return chr(10).join(lines) + chr(10)


def main():
    nb = json.loads(V4.read_text(encoding='utf-8'))
    probe = RUNNER.read_text(encoding='utf-8')
    compile(probe, 'agent_smoke_v5.py', 'exec')
    assert '--tasks-jsonl' in probe, 'runner must support tasks-jsonl input'

    intro = chr(10).join([
        "# ATLAS: bounded single-task agent smoke (v5, requests_7502 on T4)",
        "",
        "v4 established the paper-track runtime configuration (source-level SM 7.5 tile",
        "guard, FP16, tensor parallelism 1) is compatible and numerically sound, and that",
        "templated prompting produces coherent output. v5 closes the remaining gap with one",
        "bounded end-to-end agent run:",
        "",
        "- Task: requests_7502 (single-file fix in src/requests/models.py; gold patch 604",
        "  chars; short, directive issue statement). Snapshot and task record come from the",
        "  official competition input mounted at /kaggle/input/competitions/.",
        "- Loop: a reduced implementation of the documented swegemma harness protocol:",
        "  read_file / edit_file / write_file / get_status / submit_patch, protected-file",
        "  stripping, unsubmitted-diff recovery, continuation nudges, and context compaction.",
        "  Execution tools (run_command) and code-graph tools are not declared; the official",
        "  ADK stack is not used.",
        "- Budgets: 24 tool calls, 28 wall-clock minutes, 768 output tokens per turn,",
        "  temperature 0, 8,192-token context.",
        "- Outputs: agent_patch.diff, trace.jsonl (per-turn tokens and tool results),",
        "  summary.json.",
        "",
        "Verification of the produced patch happens separately (Phase 2), outside this",
        "kernel. This run produces a bounded agent data point only; it is not a competition",
        "score.",
    ])
    nb['cells'][0]['source'] = intro

    runner = build_runner(probe)
    compile(runner, 'runner_v5.py', 'exec')
    nb['cells'][2]['source'] = [runner]

    outro = chr(10).join([
        "The JSON and logs in notebook outputs are the evidence. summary.json records the",
        "termination reason, turn and tool-call counts, token totals, and the stripped patch",
        "size; trace.jsonl carries the per-turn protocol. The patch is verified in a fresh",
        "sandbox separately: apply agent_patch.diff, then task.test_patch, then run the",
        "test-patch targets under pytest with JUnit XML. Resolution is not expected from a",
        "smoke run; the measured claims are loop integrity, budgets, tokens, and timing.",
    ])
    nb['cells'][3]['source'] = outro

    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')

    tree = ast.parse(''.join(nb['cells'][2]['source']))
    embedded = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'write_text' and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            embedded = node.args[0].value
            break
    if embedded is None:
        raise SystemExit('runner literal not found')
    compile(embedded, 'agent_smoke_v5_embedded.py', 'exec')
    assert embedded == probe, 'embedded runner drifted from source file'
    assert 'SWEGEMMA_BASELINE' in embedded and 'apply_sm75_tile_patch' in embedded
    print('v5 written:', OUT, '| runner chars:', len(embedded))


if __name__ == '__main__':
    main()
