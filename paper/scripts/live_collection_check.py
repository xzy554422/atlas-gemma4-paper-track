"""Direction 4, step 2: live per-repo collection check in the sandbox container.

For each repo representative snapshot: extract, install existing PREDEPS plus the
extra modules discovered by mining, then run `pytest --collect-only -q` with
--continue-on-collection-errors so every missing module is enumerated in one pass.

Writes results/dep_cache/live_collection.json.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper' / 'scripts'))

import p1d_h1_container as P  # noqa: E402

OUT = ROOT / 'results' / 'dep_cache' / 'live_collection.json'

CASES = [
    # (instance_id, repo, extra modules to install beyond PREDEPS)
    ('rich_3454', 'Textualize/rich', ['attrs']),
    ('fastapi_11194', 'fastapi/fastapi',
     ['attrs', 'inline-snapshot', 'dirty-equals', 'orjson', 'sqlalchemy',
      'pydantic-settings', 'a2wsgi', 'strawberry-graphql', 'flask', 'sqlmodel']),
    ('requests_7502', 'psf/requests', ['pytest-httpbin']),
    ('httpx_3672', 'encode/httpx', ['pytest-httpbin']),
]

MISSING_RE = re.compile(r"ModuleNotFoundError: No module named '([a-zA-Z0-9_.]+)'")


def start(name: str):
    wl = P.host_path(str(ROOT / 'data' / 'wheels_local'))
    we = P.host_path(str(ROOT / 'data' / 'wheels_extra'))
    cmd = (f'docker run --rm -d --name {name} {P.DOCKER_LIMITS} '
           f'-v "{wl}:/wheels:ro" -v "{we}:/wheels_extra:ro" {P.IMAGE} sleep 7200')
    r = P.sh(cmd)
    return (name, None) if r.returncode == 0 else ('', r.stderr.strip()[:200])


def collect(c: str) -> dict:
    junit = '/tmp/_collect.xml'
    r = P.dm(c, 'bash -c "cd /workspace && timeout 300 python3 -s -m pytest tests/ '
                f'--collect-only -q -p no:anyio -o python_classes=\'Test* *Test\' '
                f'--continue-on-collection-errors --junitxml={junit} 2>&1 | tail -25"',
             timeout=360)
    out = r.stdout or r.stderr or ''
    # Enumerate missing modules from the error summary lines.
    modules = sorted(set(MISSING_RE.findall(out)))
    m = re.search(r'(\d+) tests? collected', out)
    e = re.search(r'(\d+) errors?', out.split('tests collected')[-1] if m else '')
    return {'exit_code': r.returncode, 'collected': int(m.group(1)) if m else None,
            'collection_errors': int(e.group(1)) if e else (0 if not modules else None),
            'missing_modules': modules, 'tail': out[-700:]}


def main() -> int:
    results = {}
    for iid, repo, extras in CASES:
        print(f'=== {repo} ({iid}) ===', flush=True)
        name = f'atlas-dep-{iid}'
        c, err = start(name)
        if not c:
            results[iid] = {'repo': repo, 'fatal': err}
            continue
        t0 = time.time()
        try:
            secs, note = P.prep_sandbox(c, iid, repo)
            install = P.dm(c, 'bash -c "python3 -m pip install --no-index --no-build-isolation '
                              '-q --find-links=/wheels --find-links=/wheels_extra '
                              + ' '.join(extras) + ' > /tmp/pip.log 2>&1; echo PIP_RC=$?; '
                              'tail -3 /tmp/pip.log"', timeout=900)
            pip_out = install.stdout or ''
            pip_rc = pip_out.split('PIP_RC=')[1].splitlines()[0] if 'PIP_RC=' in pip_out else '?'
            col = collect(c)
            results[iid] = {'repo': repo, 'prep_seconds': secs,
                            'pip_rc': pip_rc, 'pip_tail': pip_out[-200:],
                            'elapsed': round(time.time() - t0, 1), **col}
            print(json.dumps({k: results[iid][k] for k in
                              ('pip_rc', 'collected', 'collection_errors', 'missing_modules')},
                             ensure_ascii=False), flush=True)
        finally:
            P.sh(f'docker rm -f {name}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('written:', OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
