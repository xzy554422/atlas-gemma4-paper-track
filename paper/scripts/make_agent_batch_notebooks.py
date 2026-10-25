"""Generate per-repo batch notebooks for Direction 1 (multi-task agent evaluation).

Produces model_agent_batch_<repo>.ipynb from the v4 skeleton: same install cell,
runner plus a repo-filtered task sample embedded, batch driver with resumable
per-task summaries.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / 'paper/notebooks/model_preflight_v4.ipynb'
RUNNER = ROOT / 'paper/scripts/agent_smoke_v5.py'
SAMPLE = ROOT / 'results/dep_cache/d1_task_sample.json'
OUT_DIR = ROOT / 'paper/notebooks'

BATCHES = {
    'rich': ('Textualize/rich', 'ATLAS agent batch rich'),
    'fastapi': ('fastapi/fastapi', 'ATLAS agent batch fastapi'),
    'requests': ('psf/requests', 'ATLAS agent batch requests'),
    'httpx': ('encode/httpx', 'ATLAS agent batch httpx'),
}


def build_runner(probe: str, repo: str, sample: list) -> str:
    sample_repo = [t for t in sample if t['repo'] == repo]
    lines = [
        "from pathlib import Path",
        "import json, os, subprocess, sys, time",
        "WORK = Path('/kaggle/working')",
        "WORK.mkdir(exist_ok=True)",
        "START = time.monotonic()",
        "(WORK / 'agent_smoke_v5.py').write_text(" + json.dumps(probe) + ", encoding='utf-8')",
        "SAMPLE = " + json.dumps(sample_repo),
        "(WORK / 'batch_sample.json').write_text(json.dumps(SAMPLE), encoding='utf-8')",
        "MODEL = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2')",
        "COMP = (lambda cands: next((p for p in cands if p.is_dir()), cands[-1]))([Path('/kaggle/input/competitions/gemma-4-developer-agent'), Path('/kaggle/input/gemma-4-developer-agent')])",
        "SNAPS = COMP / 'snapshots'",
        "TASKS = COMP / 'tasks.jsonl'",
        "assert SNAPS.is_dir() and TASKS.is_file(), 'competition inputs missing'",
        "print('batch size:', len(SAMPLE))",
        "env = os.environ.copy()",
        "env.update(CUDA_VISIBLE_DEVICES='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',",
        "           TRANSFORMERS_NO_TF='1', TRANSFORMERS_NO_FLAX='1',",
        "           VLLM_NO_USAGE_STATS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')",
        "if installation['status'] != 'passed':",
        "    result = {'agent_status': 'not_run', 'reason': 'runtime_installation_failed'}",
        "else:",
        "    with (WORK / 'batch.log').open('w') as log:",
        "        try:",
        "            process = subprocess.run(",
        "                [sys.executable, str(WORK / 'agent_smoke_v5.py'),",
        "                 '--batch-file', str(WORK / 'batch_sample.json'),",
        "                 '--tasks-jsonl', str(TASKS), '--snapshot-dir', str(SNAPS),",
        "                 '--model-path', str(MODEL), '--output-dir', str(WORK),",
        "                 '--tool-budget', '24', '--time-minutes', '20'],",
        "                env=env, stdout=log, stderr=subprocess.STDOUT,",
        "                timeout=14400, start_new_session=True)",
        "            print('runner exit:', process.returncode)",
        "        except subprocess.TimeoutExpired:",
        "            print('runner timeout; completed tasks are in batch_summary.json')",
        "    print((WORK / 'batch.log').read_text(errors='replace')[-3000:])",
        "summary = WORK / 'batch_summary.json'",
        "result = json.loads(summary.read_text()) if summary.exists() else {'tasks': {}}",
        "result['notebook_elapsed_seconds'] = round(time.monotonic() - START, 3)",
        "summary.write_text(json.dumps(result, indent=1))",
        "print(json.dumps(result, indent=1))",
    ]
    return chr(10).join(lines) + chr(10)


def main() -> None:
    nb_v4 = json.loads(V4.read_text(encoding='utf-8'))
    probe = RUNNER.read_text(encoding='utf-8')
    compile(probe, 'agent_smoke_v5.py', 'exec')
    assert '--batch-file' in probe, 'runner lacks batch mode'
    sample = json.loads(SAMPLE.read_text(encoding='utf-8'))

    for key, (repo, title) in BATCHES.items():
        subset = [t for t in sample if t['repo'] == repo]
        if not subset:
            print('skip', key, '(no tasks)')
            continue
        nb = json.loads(V4.read_text(encoding='utf-8'))  # fresh copy
        nb['cells'][0]['source'] = chr(10).join([
            f"# {title} (Direction 1)",
            "",
            f"Batch evaluation of the v7 smoke configuration on {len(subset)} {repo} tasks,",
            "stratified by gold-patch size (max 20k chars, max 8 files). Same runner and",
            "budgets as the smoke run: reduced harness protocol, 24 tool calls, 20 minutes,",
            "temperature 0, 8,192-token context. Per-task outputs land in <iid>/ with",
            "trace.jsonl, agent_patch.diff, and summary.json; batch_summary.json accumulates",
            "incrementally so an interrupted kernel keeps completed tasks. Verification of",
            "each non-empty patch runs separately on the local sandbox.",
        ])
        runner = build_runner(probe, repo, sample)
        compile(runner, f'runner_{key}.py', 'exec')
        nb['cells'][2]['source'] = [runner]
        nb['cells'][3]['source'] = (
            "batch_summary.json is the primary artifact: termination reason, turns, tool",
            "calls, tokens, and patch size per task. Fetching the output directory yields",
            "every task subdirectory for local Phase 2 verification.")
        out = OUT_DIR / f'model_agent_batch_{key}.ipynb'
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
        print('written', out.name, f'({len(subset)} tasks)')


if __name__ == '__main__':
    main()
