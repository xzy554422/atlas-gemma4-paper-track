"""Generate Direction 3/5/6 batch notebooks from the rich batch template.

d5_state : 10-task subset, --compaction-strategy state
d6_12b   : 10-task subset, 12B model (path discovered at runtime)
d3_sel   : 10-task subset, --allow-run-command --verification-guidance selected
d3_full  : 10-task subset, --allow-run-command --verification-guidance full
All d3 kernels also install the per-repo dependency superset so run_command pytest works.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'paper/notebooks/model_agent_batch_httpx.ipynb'
RUNNER = ROOT / 'paper/scripts/agent_smoke_v5.py'
SAMPLE = ROOT / 'results/dep_cache/d2_task_subset.json'
OUT_DIR = ROOT / 'paper/notebooks'

DEPS_SUPERSET = ['attrs', 'inline-snapshot', 'dirty-equals', 'orjson', 'sqlalchemy',
                 'pydantic-settings', 'a2wsgi', 'strawberry-graphql', 'flask', 'sqlmodel',
                 'pyyaml', 'email-validator', 'pyjwt', 'passlib', 'bcrypt', 'ujson',
                 'pytest-httpbin']


def make_runner(probe: str, base_runner: str, *, sample: list, model: str | None,
                extra_args: list, install_deps: bool) -> str:
    head = base_runner.split("(WORK / 'agent_smoke_v5.py').write_text(", 1)[0]
    tail = base_runner[base_runner.index(chr(10) + 'MODEL = Path('):]
    runner = (head + "(WORK / 'agent_smoke_v5.py').write_text("
              + json.dumps(probe) + ", encoding='utf-8')" + tail)
    out_lines = []
    for line in runner.splitlines():
        if line.startswith("(WORK / 'agent_smoke_v5.py').write_text("):
            out_lines.append(line)
            out_lines.append('SAMPLE = ' + json.dumps(sample))
            out_lines.append("(WORK / 'batch_sample.json').write_text("
                             "json.dumps(SAMPLE), encoding='utf-8')")
            continue
        if model and line.startswith('MODEL = Path('):
            out_lines.append('MODEL_CANDIDATES = ' + json.dumps(model))
            out_lines.append("MODEL = next((q for q in (Path(c) for c in "
                             "MODEL_CANDIDATES) if q.exists()), "
                             "Path(MODEL_CANDIDATES[-1]))")
            continue
        if install_deps and line.startswith("if installation['status'] != 'passed':"):
            out_lines.append('DEPS = ' + json.dumps(DEPS_SUPERSET))
            out_lines.append("PDEPS = subprocess.run(['python3', '-m', 'pip', 'install', "
                             "'--no-index', '--no-build-isolation', '-q', "
                             "'--find-links=/wheels', '--find-links=/wheels_extra'] + DEPS, "
                             "capture_output=True, text=True)")
            out_lines.append("print('deps install rc:', PDEPS.returncode, "
                             "(PDEPS.stderr or '')[-200:])")
        if line.strip() == "'--tool-budget', '24', '--time-minutes', '20'],":
            line = line[:-2] + ', ' + ', '.join(extra_args) + '],'
        out_lines.append(line)
    return chr(10).join(out_lines) + chr(10)


def main() -> None:
    probe = RUNNER.read_text(encoding='utf-8')
    compile(probe, 'agent_smoke_v5.py', 'exec')
    sample = json.loads(SAMPLE.read_text(encoding='utf-8'))
    base_nb = json.loads(BASE.read_text(encoding='utf-8'))
    base_runner = ''.join(base_nb['cells'][2]['source'])

    jobs = [
        ('d5_state', 'ATLAS d5 state compaction', None,
         ["'--compaction-strategy', 'state'"], False),
        ('d6_12b', 'ATLAS d6 12B model', [
            '/kaggle/input/models/google/gemma-4/other/gemma-4-12b-it-qat-w4a16-ct/2',
            '/kaggle/input/models/google/gemma-4/other/gemma-4-12b-it-qat-w4a16-ct/1',
        ], [], False),
        ('d3_sel', 'ATLAS d3 selected guidance', None,
         ["'--verification-guidance', 'selected'", "'--allow-run-command'"], True),
        ('d3_full', 'ATLAS d3 full guidance', None,
         ["'--verification-guidance', 'full'", "'--allow-run-command'"], True),
    ]
    for key, title, model, extra_args, install_deps in jobs:
        nb = json.loads(json.dumps(base_nb))
        nb['cells'][0]['source'] = chr(10).join([
            f"# {title} (batch, 10-task subset)",
            "",
            "Same loop and budgets as Direction 1; the changed variable is: "
            + {'d5_state': 'context compaction strategy (state vs truncate)',
               'd6_12b': 'model scale (12B vs E4B)',
               'd3_sel': 'verification guidance (selected targets, run_command enabled)',
               'd3_full': 'verification guidance (full file list, run_command enabled)'}[key]
            + ". Control arm data comes from Direction 1.",
            "",
            "run_command executes bash inside the kernel workspace (offline; repository and "
            "test dependencies pre-installed from the wheel mounts). Verification of all "
            "patches happens afterwards on the local sandbox.",
        ])
        runner = make_runner(probe, base_runner, sample=sample, model=model,
                             extra_args=extra_args, install_deps=install_deps)
        compile(runner, f'runner_{key}.py', 'exec')
        nb['cells'][2]['source'] = [runner]
        out = OUT_DIR / f'model_agent_{key}.ipynb'
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
        print('written', out.name)


if __name__ == '__main__':
    main()
