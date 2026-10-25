"""Phase 2 verification for the v5 agent smoke patch (requests_7502).

Runs three arms in the same prepared sandbox (fresh swebench container):
  baseline  - snapshot + test_patch only      -> the new test must FAIL (FAIL_TO_PASS)
  agent     - snapshot + agent_patch + test_patch -> the measured arm
  gold      - snapshot + gold patch + test_patch  -> sanity that the task/env can pass

A competition-style resolution verdict for the agent arm requires: pytest exit 0,
complete JUnit with zero failures/errors, and the required test node present and
passed (not skipped). This mirrors HARNESS_README.md section 8.2 step 5 locally.
"""
from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper' / 'scripts'))

import p1d_h1_container as P  # noqa: E402

IID = 'requests_7502'
REPO = 'psf/requests'
TARGET = 'tests/test_requests.py::TestRequests::test_post_named_tempfile'
CONTAINER = 'atlas-verify-requests-7502'
OUT_DIR = ROOT / 'results' / 'final_kaggle' / 'agent_smoke_v5'


def node_status(junit_path: Path, node_name: str) -> str:
    try:
        root = ET.parse(junit_path).getroot()
    except ET.ParseError:
        return 'unparseable'
    for case in root.iter('testcase'):
        name = case.get('name', '')
        if node_name in name or name == node_name:
            if case.find('skipped') is not None:
                return 'skipped'
            if case.find('failure') is not None:
                return 'failed'
            if case.find('error') is not None:
                return 'error'
            return 'passed'
    return 'missing'


def run_arm(container: str, label: str, agent_patch: str, gold_patch: str,
            test_patch: str, junit_host: Path) -> dict:
    P.reset_workspace(container)
    for label_patch, text in (('agent', agent_patch), ('gold', gold_patch)):
        if text:
            ok, how = P.apply_patch(container, text)
            if not ok:
                return {'arm': label, 'fatal': f'{label_patch} patch application failed'}
    ok, how = P.apply_patch(container, test_patch)
    if not ok:
        return {'arm': label, 'fatal': 'test_patch application failed'}
    junit_container = f'/tmp/_verify_{label}.xml'
    rc, secs, tail, timed_out = P.run_pytest(container, [TARGET], junit_container)
    receipt = OUT_DIR / f'junit_{label}.xml'
    copied = P.sh(f'docker cp {container}:{junit_container} "{P.host_path(str(receipt))}"').returncode == 0
    stats = P.junit_stats(str(receipt)) if copied else {'error': 'junit docker cp failed'}
    verdict = P.verdict(rc, stats, timed_out)
    required = node_status(receipt, 'test_post_named_tempfile') if copied else 'no_junit'
    return {
        'arm': label, 'exit_code': rc, 'seconds': secs, 'timed_out': timed_out,
        'verdict': verdict, 'junit': stats, 'required_node': required, 'tail': tail[:300],
    }


def start_container_verified(name: str) -> tuple[str, str | None]:
    """Same limits as P.start_container but also mounts data/wheels_extra so the
    sandbox can install pytest-httpbin (present in the official dependency cache,
    absent from our local image)."""
    wheels_local = P.host_path(str(ROOT / 'data' / 'wheels_local'))
    wheels_extra = P.host_path(str(ROOT / 'data' / 'wheels_extra'))
    cmd = (f'docker run --rm -d --name {name} {P.DOCKER_LIMITS} '
           f'-v "{wheels_local}:/wheels:ro" -v "{wheels_extra}:/wheels_extra:ro" '
           f'{P.IMAGE} sleep 7200')
    r = P.sh(cmd)
    if r.returncode != 0:
        return '', r.stderr.strip()[:300]
    return name, None


def main() -> int:
    task = next(json.loads(line) for line in (ROOT / 'data' / 'tasks.jsonl')
                .open(encoding='utf-8') if json.loads(line)['instance_id'] == IID)
    agent_patch = (OUT_DIR / 'agent_patch.diff').read_text(encoding='utf-8')
    if not agent_patch.strip():
        print('agent patch is empty; nothing to verify')
        return 1

    import shutil
    docker = shutil.which('docker')
    assert docker, 'docker not found'

    started = time.time()
    result = {'instance_id': IID, 'target': TARGET, 'arms': {}, 'configuration':
              'local docker sandbox (swebench-sandbox), official hermetic pytest command; '
              'local absolute diagnostic, not an official competition score'}

    container, err = start_container_verified(CONTAINER)
    if not container:
        print('container failed:', err)
        return 1
    try:
        secs, note = P.prep_sandbox(container, IID, REPO)
        extra = P.dm(container, 'bash -c "python3 -m pip install --no-index --no-build-isolation '
                                '-q --find-links=/wheels --find-links=/wheels_extra '
                                'pytest-httpbin > /tmp/pip.log 2>&1; echo PIP_RC=$?; '
                                'tail -2 /tmp/pip.log"', timeout=600)
        pip_tail = (extra.stdout or extra.stderr or '')
        pip_rc = pip_tail.split('PIP_RC=')[1].splitlines()[0] if 'PIP_RC=' in pip_tail else '?'
        result['prep'] = {'seconds': secs, 'note': note[:300],
                          'pytest_httpbin_install': {'exit_code': pip_rc,
                                                     'tail': pip_tail[-260:]}}
        test_patch = task['test_patch']
        if not test_patch.endswith('\n'):
            test_patch += '\n'
        arms = [
            ('baseline', '', ''),
            ('agent', agent_patch, ''),
            ('gold', '', task['patch']),
        ]
        for label, agent_patch_arm, gold_patch_arm in arms:
            print(f'=== arm {label} ===', flush=True)
            arm = run_arm(container, label, agent_patch_arm, gold_patch_arm,
                          test_patch, OUT_DIR / f'junit_{label}.xml')
            result['arms'][label] = arm
            print(json.dumps(arm, ensure_ascii=False)[:400], flush=True)
    finally:
        P.sh(f'docker rm -f {CONTAINER}')
    result['elapsed_seconds'] = round(time.time() - started, 2)

    agent = result['arms'].get('agent', {})
    base = result['arms'].get('baseline', {})
    gold = result['arms'].get('gold', {})
    result['resolution'] = {
        'baseline_fails': base.get('verdict') == 'fail',
        'agent_resolves': (agent.get('verdict') == 'pass'
                           and agent.get('required_node') == 'passed'),
        'gold_resolves': (gold.get('verdict') == 'pass'
                          and gold.get('required_node') == 'passed'),
        'resolved': bool(
            base.get('verdict') == 'fail'
            and agent.get('verdict') == 'pass'
            and agent.get('required_node') == 'passed'),
    }
    out = OUT_DIR / 'verification.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['resolution'], indent=1))
    print('written:', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
