"""Direction 1 Phase 2: batch three-arm verification for agent patches.

For every <batch_dir>/<iid>/agent_patch.diff: fresh sandbox, derive the required test
node ids from task.test_patch (added test functions and their enclosing classes),
then run baseline / agent / gold arms with the official hermetic pytest command and
the completed dependency cache. Writes <iid>/verification.json plus a combined
results/dep_cache/d1_verifications.json.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper' / 'scripts'))

import p1d_h1_container as P  # noqa: E402
from verify_agent_patch import node_status, run_arm, start_container_verified  # noqa: E402

EXTRA_MODULES = {
    'Textualize/rich': ['attrs'],
    'fastapi/fastapi': ['attrs', 'inline-snapshot', 'dirty-equals', 'orjson', 'sqlalchemy',
                        'pydantic-settings', 'a2wsgi', 'strawberry-graphql', 'flask',
                        'sqlmodel', 'pyyaml', 'email-validator', 'pyjwt', 'passlib',
                        'bcrypt', 'ujson'],
    'psf/requests': ['pytest-httpbin'],
    'encode/httpx': ['pytest-httpbin'],
}


def derive_targets(test_patch: str) -> list[str]:
    """Local git-apply of the test patch to find added test node ids."""
    import subprocess
    import tarfile
    tmp = Path(tempfile.mkdtemp())
    # Apply against a throwaway copy of any repo-less scratch: build a minimal fake tree
    # from the patch itself is unreliable for class context, so callers pass the
    # snapshot path instead. Fallback: parse hunks for function context headers.
    targets = []
    current_class = None
    for line in test_patch.splitlines():
        m = re.match(r'@@ -\d+,\d+ \+\d+,\d+ @@ ?(?:\w+\s+)?(?:def (\w+))?', line)
        if line.startswith('@@'):
            m2 = re.search(r'@@ -\d+,\d+ \+\d+,\d+ @@ ?.*?class (\w+)', line)
            if m2:
                current_class = m2.group(1)
            m3 = re.search(r'@@ -\d+,\d+ \+\d+,\d+ @@ ?.*?def (\w+)', line)
            if m3 and not m3.group(1).startswith('test'):
                pass
            continue
        if line.startswith('+'):
            m = re.match(r'\+\s*def (test_\w+)\(', line)
            if m:
                targets.append(m.group(1))
        m = re.match(r'^\s*class (\w+)', line)
        if m and not line.startswith(('+', '-')):
            current_class = m.group(1)
    return targets


def required_tests(test_patch: str) -> list[str]:
    """Test function names added or modified by the test patch (hunk context or new defs)."""
    required = []
    for line in test_patch.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            m = re.match(r'\+\s*def (test_\w+)\(', line)
            if m and m.group(1) not in required:
                required.append(m.group(1))
        if line.startswith('@@'):
            m = re.search(r'@@ [^@]+ @@ \w+\s+(?:def )?(test_\w+)', line)
            if m and m.group(1) not in required:
                required.append(m.group(1))
    return required


def files_of_patch(test_patch: str) -> list[str]:
    return sorted(set(re.findall(r'^\+\+\+ b/(\S+)', test_patch, re.M)))


def targets_with_class(snapshot_tgz: Path, test_patch: str) -> list[str]:
    """Kept for diagnostics: apply test_patch to resolve classes (robust path)."""
    import subprocess
    import tarfile
    tmp = Path(tempfile.mkdtemp())
    with tarfile.open(snapshot_tgz, 'r:gz') as tar:
        tar.extractall(tmp)

    def git(*args):
        return subprocess.run(['git', '-C', str(tmp), *args], capture_output=True, text=True)

    git('config', '--global', '--add', 'safe.directory', '*')
    patch = test_patch if test_patch.endswith('\n') else test_patch + '\n'
    pfile = tmp / 'tp.diff'
    pfile.write_text(patch, encoding='utf-8')
    for flags in (['-p1'], ['-p1', '--ignore-whitespace'], ['-p0']):
        if git('apply', *flags, str(pfile)).returncode == 0:
            break
    diff_files = sorted(set(re.findall(r'^\+\+\+ b/(\S+)', test_patch, re.M)))
    targets = []
    for rel in diff_files:
        path = tmp / rel
        if not path.is_file():
            continue
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        cls = None
        for i, line in enumerate(lines, 1):
            m = re.match(r'^class (\w+)', line)
            if m:
                cls = m.group(1)
            m = re.match(r'\s*def (test_\w+)\(', line)
            if m:
                node = f'{rel}::' + (cls + '::' if cls else '') + m.group(1)
                targets.append(node)
    return targets


def main() -> int:
    batch_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results/final_kaggle/batch_rich'
    tasks_index = {}
    for line in (ROOT / 'data/tasks.jsonl').open(encoding='utf-8'):
        if line.strip():
            record = json.loads(line)
            tasks_index[record['instance_id']] = record

    combined_path = ROOT / 'results/dep_cache' / 'd1_verifications.json'
    combined = json.loads(combined_path.read_text(encoding='utf-8')) if combined_path.exists() else {}

    for task_dir in sorted(batch_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        iid = task_dir.name
        patch_file = task_dir / 'agent_patch.diff'
        if not patch_file.is_file():
            print(f'[skip] {iid}: no patch file')
            continue
        agent_patch = patch_file.read_text(encoding='utf-8')
        task = tasks_index.get(iid)
        if not task:
            print(f'[skip] {iid}: not in tasks.jsonl')
            continue
        if not agent_patch.strip():
            print(f'[skip] {iid}: empty patch')
            combined[iid] = {'agent_patch_chars': 0, 'resolved': False,
                             'note': 'empty patch; no verification run'}
            continue
        if iid in combined and combined[iid].get('resolution', {}).get('resolved') is not None:
            print(f'[skip] {iid}: already verified')
            continue

        repo = task['repo']
        targets = files_of_patch(task['test_patch'])
        required_names = required_tests(task['test_patch'])
        if not targets or not required_names:
            combined[iid] = {'agent_patch_chars': len(agent_patch), 'resolved': False,
                             'note': 'no test targets derivable'}
            continue
        print(f'=== verify {iid} ({repo}) targets={targets[:3]}{"..." if len(targets) > 3 else ""} ===',
              flush=True)

        started = time.time()
        container, err = start_container_verified(f'atlas-v1-{iid}')
        if not container:
            combined[iid] = {'fatal': err}
            continue
        result = {'instance_id': iid, 'repo': repo, 'test_files': targets,
                  'required_tests': required_names,
                  'agent_patch_chars': len(agent_patch), 'arms': {}}
        try:
            P.prep_sandbox(container, iid, repo)
            P.dm(container, 'bash -c "python3 -m pip install --no-index --no-build-isolation '
                            '-q --find-links=/wheels --find-links=/wheels_extra '
                            + ' '.join(EXTRA_MODULES[repo]) + ' > /tmp/pip.log 2>&1"', timeout=900)
            test_patch = task['test_patch']
            if not test_patch.endswith('\n'):
                test_patch += '\n'
            gold = task['patch']
            if not gold.endswith('\n'):
                gold += '\n'
            for label, ap, gp in (('baseline', '', ''), ('agent', agent_patch, ''),
                                  ('gold', '', gold)):
                junit_host = task_dir / f'junit_{label}.xml'
                arm = run_arm(container, label, ap, gp, test_patch, junit_host,
                              targets=targets, required_names=required_names)
                result['arms'][label] = arm
                statuses = arm.get('required_statuses') or {}
                print(f'  {label}: {arm.get("verdict")} required={statuses}', flush=True)
        finally:
            P.sh(f'docker rm -f atlas-v1-{iid}')

        base = result['arms'].get('baseline', {})
        agent = result['arms'].get('agent', {})
        gold_arm = result['arms'].get('gold', {})

        def worst(arm):
            vals = list((arm.get('required_statuses') or {}).values())
            for bad in ('missing', 'error', 'failed', 'skipped', 'no_junit'):
                if bad in vals:
                    return bad
            return 'passed' if vals and all(v == 'passed' for v in vals) else 'unknown'

        result['resolution'] = {
            'baseline_required': worst(base),
            'agent_required': worst(agent),
            'gold_required': worst(gold_arm),
        }
        result['resolution']['baseline_fails'] = result['resolution']['baseline_required'] in ('failed', 'error')
        result['resolution']['agent_resolves'] = result['resolution']['agent_required'] == 'passed'
        result['resolution']['gold_resolves'] = result['resolution']['gold_required'] == 'passed'
        result['resolution']['resolved'] = bool(
            result['resolution']['baseline_fails'] and result['resolution']['agent_resolves'])
        result['elapsed_seconds'] = round(time.time() - started, 1)
        (task_dir / 'verification.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        combined[iid] = {'agent_patch_chars': result['agent_patch_chars'],
                         'resolution': result['resolution'],
                         'elapsed_seconds': result['elapsed_seconds']}
        combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + '\n',
                                 encoding='utf-8')
        print(f'  resolved={result["resolution"]["resolved"]}', flush=True)
    print('combined:', json.dumps(combined, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
