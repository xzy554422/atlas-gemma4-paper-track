"""Generate Direction 2 ablation notebooks (oracle / b0 arms) via AST span rewrite."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'paper/notebooks/model_agent_batch_httpx.ipynb'
RUNNER = ROOT / 'paper/scripts/agent_smoke_v5.py'
SAMPLE = ROOT / 'results/dep_cache/d2_task_subset.json'
OUT_DIR = ROOT / 'paper/notebooks'

MODES = {'oracle': ('yzx22442255/atlas-d2-oracle', 'ATLAS d2 oracle arm'),
         'b0': ('yzx22442255/atlas-d2-b0', 'ATLAS d2 b0 arm')}


def main() -> None:
    nb = json.loads(BASE.read_text(encoding='utf-8'))
    probe = RUNNER.read_text(encoding='utf-8')
    compile(probe, 'agent_smoke_v5.py', 'exec')
    sample = json.loads(SAMPLE.read_text(encoding='utf-8'))
    assert len(sample) == 10

    for mode, (kid, title) in MODES.items():
        nb2 = json.loads(json.dumps(nb))
        nb2['cells'][0]['source'] = chr(10).join([
            f"# {title} (Direction 2 localization ablation)",
            "",
            f"10-task subset (rich 3 / fastapi 3 / requests 3 / httpx 1), localization mode = {mode}.",
            "The index arm for the same tasks reuses the Direction 1 runs. Everything else is",
            "identical to the v7 smoke configuration: 24 tool calls, 20 minutes, temperature 0,",
            "8,192-token context. Three-arm verification afterwards on the local sandbox.",
            "Oracle hints are answer-informed references, not an implementable retrieval method.",
        ])

        runner = ''.join(nb2['cells'][2]['source'])
        head = runner.split("(WORK / 'agent_smoke_v5.py').write_text(", 1)[0]
        # The tail starts after the write_text call's closing ", encoding='utf-8')" —
        # found via the LAST occurrence before the MODEL = line.
        marker = chr(10) + "MODEL = Path("
        tail = runner[runner.index(marker):]

        # Per-line overrides inside the runner wrapper.
        tail_lines = []
        for line in tail.splitlines():
            if line.strip() == "'--tool-budget', '24', '--time-minutes', '20'],":
                line = (line.replace("],", "")
                        + ", '--localization-mode', '" + mode + "'],")
            tail_lines.append(line)
        tail = chr(10).join(tail_lines)

        runner2 = (head + "(WORK / 'agent_smoke_v5.py').write_text("
                   + json.dumps(probe) + ", encoding='utf-8')" + tail)

        # Whole-wrapper line overrides. The head/tail split drops everything between
        # the write_text call and MODEL (including SAMPLE), so re-insert SAMPLE
        # explicitly right after the write_text line.
        out_lines = []
        for line in runner2.splitlines():
            if line.startswith("(WORK / 'agent_smoke_v5.py').write_text("):
                out_lines.append(line)
                out_lines.append('SAMPLE = ' + json.dumps(sample))
                out_lines.append("(WORK / 'batch_sample.json').write_text("
                                 "json.dumps(SAMPLE), encoding='utf-8')")
                continue
            if line.strip() == "'--tool-budget', '24', '--time-minutes', '20'],":
                line = (line.replace("],", "")
                        + ", '--localization-mode', '" + mode + "'],")
            out_lines.append(line)
        runner2 = chr(10).join(out_lines) + chr(10)
        compile(runner2, f'runner_{mode}.py', 'exec')
        assert f"'--localization-mode', '{mode}'" in runner2
        assert json.dumps(sample) in runner2
        nb2['cells'][2]['source'] = [runner2]

        out = OUT_DIR / f'model_agent_d2_{mode}.ipynb'
        out.write_text(json.dumps(nb2, ensure_ascii=False, indent=1), encoding='utf-8')
        print('written', out.name)


if __name__ == '__main__':
    main()
