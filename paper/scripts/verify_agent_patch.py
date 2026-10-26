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
    """Match full pytest identity and aggregate every parametrized case."""
    try:
        root = ET.parse(junit_path).getroot()
    except (ET.ParseError, OSError):
        return 'unparseable'
    parts = node_name.split('::')
    if len(parts) < 2:
        return 'missing'  # Bare names cannot establish full test identity.
    name = parts[-1]
    expected_class = '.'.join([parts[0].removesuffix('.py').replace('/', '.'), *parts[1:-1]])
    statuses = []
    for case in root.iter('testcase'):
        if len(parts) > 1 and case.get('classname', '') != expected_class:
            continue
        actual = case.get('name', '')
        if actual == name or ('[' not in name and actual.startswith(name + '[')):
            statuses.append(next((status for tag, status in
                                  (('error', 'error'), ('failure', 'failed'), ('skipped', 'skipped'))
                                  if case.find(tag) is not None), 'passed'))
    return aggregate_status(statuses)


def aggregate_status(statuses) -> str:
    for status in ('no_junit', 'unparseable', 'missing', 'error', 'failed', 'skipped'):
        if status in statuses:
            return status
    return 'passed' if statuses and all(s == 'passed' for s in statuses) else 'missing'


def case_outcomes(receipt: Path) -> dict:
    """Map 'classname::name' -> passed/failed/skipped/error for every test case."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(receipt).getroot()
    except (ET.ParseError, OSError):
        return {}
    out = {}
    for case in root.iter('testcase'):
        key = (case.get('classname', '') + '::' + case.get('name', '')).lstrip(':')
        if case.find('skipped') is not None:
            out[key] = 'skipped'
        elif case.find('failure') is not None:
            out[key] = 'failed'
        elif case.find('error') is not None:
            out[key] = 'error'
        else:
            out[key] = 'passed'
    return out


def required_cases(arm: dict, names) -> dict:
    """Required-case outcomes, tolerating classname prefixes in the receipt."""
    cases = arm.get('cases') or {}
    if not cases:
        return dict(arm.get('required_statuses') or {})
    out = {}
    for name in names:
        wanted = name.rsplit('::', 1)[-1].split('[')[0]
        hits = [v for k, v in cases.items() if k.rsplit('::', 1)[-1].split('[')[0] == wanted]
        if not hits:
            out[name] = 'missing'
        elif any(h == 'failed' or h == 'error' for h in hits):
            out[name] = 'failed'
        elif all(h == 'skipped' for h in hits):
            out[name] = 'skipped'
        else:
            out[name] = 'passed'
    return out


def accepted(arm: dict) -> bool:
    stats = arm.get('junit') or {}
    return bool(not arm.get('fatal') and arm.get('exit_code') == 0
                and not arm.get('timed_out') and P.junit_complete(stats)
                and stats['failures'] == 0 and stats['errors'] == 0
                and arm.get('required_statuses')
                and aggregate_status(list(arm['required_statuses'].values())) == 'passed')


def resolution(arms: dict) -> dict:
    """SWE-bench-style delta acceptance with an explicit gold control.

    resolved requires, on the required tests only:
      - baseline: every required test fails (setup errors or missing receipts -> unknown)
      - agent:    every required test passes, receiving a complete receipt
      - gold:     the same transition (guards against a broken environment)
      - no regression: no case that passed on baseline fails after the agent patch
    Whole-file exit status is reported separately because unrelated environment-bound
    cases (offline network tests) legitimately fail in both baseline and gold.
    """
    base, agent, gold = (arms.get(key, {}) for key in ('baseline', 'agent', 'gold'))
    names = list((agent.get('required_statuses') or base.get('required_statuses') or {}).keys())
    base_req = required_cases(base, names)
    agent_req = required_cases(agent, names)
    gold_req = required_cases(gold, names)

    base_stats = base.get('junit') or {}
    complete = lambda arm: bool(not arm.get('fatal') and not arm.get('timed_out')
                                and arm.get('exit_code') in (0, 1)
                                and P.junit_complete(arm.get('junit') or {}))
    # FAIL_TO_PASS may be a subset: a required test that already passed on baseline
    # (a PASS_TO_PASS-style case) must stay passing, it is not required to fail.
    fail_to_pass = [n for n, v in base_req.items() if v == 'failed']
    already_passing = [n for n, v in base_req.items() if v == 'passed']
    baseline_valid = bool(complete(base) and base_req and fail_to_pass
                          and all(v in ('passed', 'failed') for v in base_req.values()))
    agent_ok = bool(complete(agent) and agent_req
                    and all(v == 'passed' for v in agent_req.values()))
    gold_ok = bool(complete(gold) and gold_req
                   and all(v == 'passed' for v in gold_req.values()))

    base_cases, agent_cases = base.get('cases') or {}, agent.get('cases') or {}
    regressions = sorted(k for k, v in base_cases.items()
                         if v == 'passed' and agent_cases.get(k) in ('failed', 'error'))

    known = baseline_valid and bool(base_cases) and bool(agent_cases)
    resolved = bool(baseline_valid and agent_ok and gold_ok and not regressions)
    return {
        'required_names': names,
        'baseline_required': aggregate_status(list(base_req.values())),
        'agent_required': aggregate_status(list(agent_req.values())),
        'gold_required': aggregate_status(list(gold_req.values())),
        'baseline_fails': baseline_valid,
        'agent_resolves': agent_ok,
        'gold_resolves': gold_ok,
        'fail_to_pass': fail_to_pass,
        'already_passing': already_passing,
        'regressions': regressions,
        'whole_file_failures': {
            'baseline': base_stats.get('failures'), 'agent': (agent.get('junit') or {}).get('failures'),
            'gold': (gold.get('junit') or {}).get('failures')},
        'resolved': resolved,
        'status': 'resolved' if resolved else ('unresolved' if known and gold_ok else 'unknown'),
    }


def run_arm(container: str, label: str, agent_patch: str, gold_patch: str,
            test_patch: str, junit_host: Path, targets=None,
            required_names=None) -> dict:
    receipt = Path(junit_host)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.unlink(missing_ok=True)
    junit_container = f'/tmp/_verify_{label}.xml'
    clean = P.dm(container, f'rm -f {junit_container}')
    if clean.returncode != 0:
        return {'arm': label, 'fatal': 'cannot remove stale container JUnit'}
    P.reset_workspace(container)
    for label_patch, text in (('agent', agent_patch), ('gold', gold_patch)):
        if text:
            ok, how = P.apply_patch(container, text)
            if not ok:
                return {'arm': label, 'fatal': f'{label_patch} patch application failed'}
    ok, how = P.apply_patch(container, test_patch)
    if not ok:
        return {'arm': label, 'fatal': 'test_patch application failed'}
    rc, secs, tail, timed_out = P.run_pytest(container, targets or [TARGET], junit_container,
                                            rootdir='/workspace')
    receipt.unlink(missing_ok=True)
    copied = P.sh(f'docker cp {container}:{junit_container} "{P.host_path(str(receipt))}"').returncode == 0
    stats = P.junit_stats(str(receipt)) if copied else {'error': 'junit docker cp failed'}
    verdict = P.verdict(rc, stats, timed_out)
    names = required_names or (targets or [TARGET])
    per_name = {name: (node_status(receipt, name) if copied else 'no_junit') for name in names}
    statuses = list(per_name.values())
    required = ('all_passed' if statuses and all(st == 'passed' for st in statuses)
                else ('missing' if 'missing' in statuses else statuses[0]))
    return {
        'arm': label, 'exit_code': rc, 'seconds': secs, 'timed_out': timed_out,
        'verdict': verdict, 'junit': stats, 'required_node': required,
        'required_statuses': per_name, 'cases': case_outcomes(receipt) if copied else {},
        'tail': tail[:300],
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

    output_dir = ROOT / 'results/audit_reverify' / ('agent_smoke_v5_' + time.strftime('%Y%m%d_%H%M%S'))
    output_dir.mkdir(parents=True, exist_ok=False)
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
                          test_patch, output_dir / f'junit_{label}.xml')
            result['arms'][label] = arm
            print(json.dumps(arm, ensure_ascii=False)[:400], flush=True)
    finally:
        P.sh(f'docker rm -f {CONTAINER}')
    result['elapsed_seconds'] = round(time.time() - started, 2)

    result['resolution'] = resolution(result['arms'])
    out = output_dir / 'verification.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['resolution'], indent=1))
    print('written:', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
