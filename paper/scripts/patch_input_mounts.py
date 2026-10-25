"""Make batch notebooks resilient to flat vs nested Kaggle input mounts.

Patches the COMP resolution in model_agent_batch_httpx.ipynb (cell 2 runner) and the
generator so competition inputs resolve from either /kaggle/input/competitions/<slug>
or /kaggle/input/<slug>.
"""
import json
from pathlib import Path

Q = chr(39)

OLD = "COMP = Path(" + Q + "/kaggle/input/competitions/gemma-4-developer-agent" + Q + ")"
NEW = ("COMP = (lambda cands: next((p for p in cands if p.is_dir()), cands[-1]))"
       "([Path(" + Q + "/kaggle/input/competitions/gemma-4-developer-agent" + Q
       + "), Path(" + Q + "/kaggle/input/gemma-4-developer-agent" + Q + ")])")

targets = [
    Path('results/final_kaggle/batch_httpx/model_agent_batch_httpx.ipynb'),
    Path('paper/notebooks/model_agent_batch_httpx.ipynb'),
    Path('paper/notebooks/model_agent_batch_rich.ipynb'),
    Path('paper/notebooks/model_agent_batch_fastapi.ipynb'),
    Path('paper/notebooks/model_agent_batch_requests.ipynb'),
    Path('paper/scripts/make_agent_batch_notebooks.py'),
]
for path in targets:
    if not path.is_file():
        continue
    text = path.read_text(encoding='utf-8')
    if OLD not in text:
        print('skip (no anchor):', path.name)
        continue
    if path.suffix == '.ipynb':
        nb = json.loads(text)
        src = ''.join(nb['cells'][2]['source'])
        if OLD not in src:
            print('skip (cell2 no anchor):', path.name)
            continue
        nb['cells'][2]['source'] = [src.replace(OLD, NEW, 1)]
        compile(src.replace(OLD, NEW, 1), path.name, 'exec')
        path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
    else:
        text = text.replace(OLD, NEW, 1)
        compile(text, path.name, 'exec')
        path.write_text(text, encoding='utf-8')
    print('patched:', path.name)
