"""Add the wheelhouse path fallback to the httpx batch notebook (v6)."""
import json
from pathlib import Path

P = Path('results/final_kaggle/batch_httpx/model_agent_batch_httpx.ipynb')
Q = chr(39)          # single quote
BS = chr(92)         # backslash
ESCQ = BS + Q        # the escaped quote form used inside the embedded literal

old = ("WHEELHOUSE_DIR = Path(" + ESCQ + "/kaggle/input/datasets/metric/"
       "gemma-4-developer-agent-wheelhouse" + ESCQ + ")")
new = ("WHEELHOUSE_DIR = (lambda cands: next((p for p in cands if any(p.glob(" + ESCQ
       + "*.whl" + ESCQ + "))), cands[-1]))([Path(" + ESCQ
       + "/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse" + ESCQ
       + "), Path(" + ESCQ + "/kaggle/input/gemma-4-developer-agent-wheelhouse"
       + ESCQ + ")])")

nb = json.loads(P.read_text(encoding='utf-8'))
src = ''.join(nb['cells'][1]['source'])
assert old in src, 'anchor missing'
src = src.replace(old, new, 1)
nb['cells'][1]['source'] = src
P.write_text(json.dumps(nb, indent=1), encoding='utf-8')
compile(src, 'cell1', 'exec')
print('fallback installed')
