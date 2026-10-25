"""Direction 4, step 3 (G4.3): full baseline test run per repo with JUnit receipts.

Runs each repo's complete suite (no test_patch) in a fresh sandbox with the completed
dependency cache, writes results/dep_cache/junit_<repo>.xml and baseline_runs.json,
and compares counts with historical P1D receipts where available.
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

OUT_DIR = ROOT / 'results' / 'dep_cache'

CASES = [
    ('rich_3454', 'Textualize/rich', ['attrs'], 1500),
    ('fastapi_11194', 'fastapi/fastapi',
     ['attrs', 'inline-snapshot', 'dirty-equals', 'orjson', 'sqlalchemy',
      'pydantic-settings', 'a2wsgi', 'strawberry-graphql', 'flask', 'sqlmodel',
      'pyyaml', 'email-validator', 'pyjwt', 'passlib', 'bcrypt', 'ujson'], 3000),
    ('requests_7502', 'psf/requests', ['pytest-httpbin'], 1200),
    ('httpx_3672', 'encode/httpx', ['pytest-httpbin'], 1200),
]

HISTORICAL = {
    'Textualize/rich': 'results/p1d_junit/rich_3454_baseline_full.xml',
    'fastapi/fastapi': 'results/p1d_junit/fastapi_11194_baseline_full.xml',
    'psf/requests': 'results/p1d_junit/requests_7502_baseline_full.xml',
}


def historical_counts(repo: str) -> dict | None:
    path = ROOT / HISTORICAL.get(repo, '')
    if not path.is_file():
        return None
    stats = P.junit_stats(str(path))
    return {k: stats.get(k) for k in ('tests', 'failures', 'errors', 'skipped')}


def main() -> int:
    results = {}
    for iid, repo, extras, timeout in CASES:
        print(f'=== full run {repo} ({iid}) ===', flush=True)
        name = f'atlas-full-{iid}'
        wl = P.host_path(str(ROOT / 'data' / 'wheels_local'))
        we = P.host_path(str(ROOT / 'data' / 'wheels_extra'))
        ok = P.sh(f'docker run --rm -d --name {name} {P.DOCKER_LIMITS} '
                  f'-v "{wl}:/wheels:ro" -v "{we}:/wheels_extra:ro" {P.IMAGE} sleep 7200').returncode == 0
        if not ok:
            results[iid] = {'repo': repo, 'fatal': 'container start failed'}
            continue
        t0 = time.time()
        try:
            P.prep_sandbox(name, iid, repo)
            P.dm(name, 'bash -c "python3 -m pip install --no-index --no-build-isolation -q '
                       '--find-links=/wheels --find-links=/wheels_extra '
                       + ' '.join(extras) + ' > /tmp/pip.log 2>&1"', timeout=900)
            junit_c = f'/tmp/_base_{iid}.xml'
            rc, secs, tail, timed_out = P.run_pytest(
                name, ['tests/'], junit_c, timeout=timeout, cont_on_error=True)
            receipt = OUT_DIR / f'junit_{repo.split("/")[-1]}.xml'
            copied = P.sh(f'docker cp {name}:{junit_c} "{P.host_path(str(receipt))}"').returncode == 0
            stats = P.junit_stats(str(receipt)) if copied else {'error': 'junit copy failed'}
            results[iid] = {
                'repo': repo, 'exit_code': rc, 'seconds': secs, 'timed_out': timed_out,
                'verdict': P.verdict(rc, stats, timed_out),
                'junit': stats, 'historical': historical_counts(repo),
                'elapsed_total': round(time.time() - t0, 1),
            }
            print(json.dumps({k: results[iid][k] for k in
                              ('exit_code', 'seconds', 'verdict', 'junit', 'historical')},
                             ensure_ascii=False), flush=True)
        finally:
            P.sh(f'docker rm -f {name}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'baseline_runs.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('written:', OUT_DIR / 'baseline_runs.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
