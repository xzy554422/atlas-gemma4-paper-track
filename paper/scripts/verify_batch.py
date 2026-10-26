"""Direction 1 Phase 2: batch three-arm verification for agent patches.

For every <batch_dir>/<iid>/agent_patch.diff: fresh sandbox, derive the required test
node ids from task.test_patch (added test functions and their enclosing classes),
then run baseline / agent / gold arms with the official hermetic pytest command and
the completed dependency cache. Writes <iid>/verification.json plus a combined
results/dep_cache/d1_verifications.json.
"""
from __future__ import annotations

import json
import argparse
import ast
import hashlib
import subprocess
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper' / 'scripts'))

import p1d_h1_container as P  # noqa: E402
from verify_agent_patch import resolution, run_arm, start_container_verified  # noqa: E402

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
    """Resolve changed tests in patched source, including file and lexical class."""
    import tarfile
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        with tarfile.open(snapshot_tgz, 'r:gz') as tar:
            tar.extractall(tmp, filter='data')
        patch = tmp / '_verification_test.diff'
        patch.write_text(test_patch.rstrip('\n') + '\n', encoding='utf-8')
        for flags in (['-p1'], ['-p1', '--ignore-whitespace'], ['-p0']):
            run = subprocess.run(['git', 'apply', *flags, str(patch)], cwd=tmp,
                                 capture_output=True)
            if run.returncode == 0:
                break
        else:
            raise ValueError('test patch cannot be applied to snapshot')
        changed = {}
        rel, line_no = None, 0
        for line in test_patch.splitlines():
            if line.startswith('+++ b/'):
                rel = line[6:]
                changed.setdefault(rel, set())
            elif line.startswith('@@'):
                line_no = int(re.search(r'\+(\d+)', line).group(1))
            elif rel and line.startswith('+'):
                changed[rel].add(line_no)
                line_no += 1
            elif rel and line.startswith(' '):
                line_no += 1
        targets = []
        def visit(node, classes, rel, touched):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    visit(child, classes + [child.name], rel, touched)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                    if child.name.startswith('test_') and any(start <= n <= child.end_lineno for n in touched):
                        targets.append('::'.join([rel, *classes, child.name]))
                else:
                    visit(child, classes, rel, touched)
        for rel, touched in changed.items():
            if rel.endswith('.py'):
                visit(ast.parse((tmp / rel).read_text(encoding='utf-8')), [], rel, touched)
        return sorted(set(targets))


def fingerprint(task, agent_patch):
    """Content-address code, tests, gold, snapshot, wheels and Docker image."""
    digest = hashlib.sha256()
    def add(value):
        digest.update(value.encode('utf-8'))
        digest.update(b'\0')
    add(json.dumps(task, sort_keys=True))
    add(agent_patch)
    for path in [Path(__file__), Path(__file__).with_name('verify_agent_patch.py'),
                 Path(P.__file__), Path(__file__).with_name('run_support.py'),
                 ROOT / 'data/snapshots' / (task['instance_id'] + '.tgz')]:
        add(path.name)
        if path.is_file():
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
        else:
            add('MISSING')
    add(json.dumps(EXTRA_MODULES, sort_keys=True))
    add(P.DOCKER_LIMITS)
    image = P.sh(f'docker image inspect {P.IMAGE} --format "{{{{.Id}}}}"')
    add(image.stdout if image.returncode == 0 else 'IMAGE_UNAVAILABLE')
    for folder in ('wheels', 'wheels_local', 'wheels_extra'):
        for path in sorted((ROOT / 'data' / folder).glob('*')):
            if path.is_file():
                add(folder + '/' + path.name)
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--ledger', type=Path)
    args = parser.parse_args()
    batch_dir = args.batch_dir
    output_dir = args.output_dir or ROOT / 'results/audit_reverify' / (batch_dir.name + '_' + time.strftime('%Y%m%d_%H%M%S'))
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_index = {}
    for line in (ROOT / 'data/tasks.jsonl').open(encoding='utf-8'):
        if line.strip():
            record = json.loads(line)
            tasks_index[record['instance_id']] = record

    combined_path = args.ledger or output_dir / 'ledger.json'
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    if combined_path.resolve() == (ROOT / 'results/dep_cache/d1_verifications.json').resolve():
        raise ValueError('refusing to overwrite archived d1 ledger')
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
        ledger_key = batch_dir.name + '/' + iid
        cache_key = fingerprint(task, agent_patch)
        result_dir = output_dir / iid / cache_key / str(time.time_ns())
        result_dir.mkdir(parents=True, exist_ok=True)
        if not agent_patch.strip():
            print(f'[skip] {iid}: empty patch')
            combined[ledger_key] = {'agent_patch_chars': 0, 'resolved': False, 'status': 'unknown',
                             'note': 'empty patch; no verification run'}
            continue
        if (combined.get(ledger_key, {}).get('fingerprint') == cache_key
                and combined[ledger_key].get('resolution', {}).get('status') in ('resolved', 'unresolved')):
            print(f'[skip] {iid}: already verified ({ledger_key})')
            continue

        repo = task['repo']
        targets = files_of_patch(task['test_patch'])
        try:
            required_names = targets_with_class(ROOT / 'data/snapshots' / (iid + '.tgz'), task['test_patch'])
        except (OSError, ValueError, SyntaxError) as exc:
            required_names = []
            print(f'[unknown] {iid}: {exc}', flush=True)
        if not targets or not required_names:
            combined[ledger_key] = {'agent_patch_chars': len(agent_patch), 'resolved': False, 'status': 'unknown',
                             'note': 'no test targets derivable'}
            continue
        print(f'=== verify {iid} ({repo}) targets={targets[:3]}{"..." if len(targets) > 3 else ""} ===',
              flush=True)

        started = time.time()
        container, err = start_container_verified(f'atlas-v1-{iid}')
        if not container:
            combined[ledger_key] = {'fatal': err, 'resolved': False, 'status': 'unknown',
                                    'fingerprint': cache_key}
            continue
        result = {'instance_id': iid, 'repo': repo, 'test_files': targets,
                  'required_tests': required_names,
                  'agent_patch_chars': len(agent_patch), 'arms': {}}
        try:
            P.prep_sandbox(container, iid, repo)
            P.dm(container, 'bash -c "python3 -m pip install --no-index --no-build-isolation '
                            '-q --find-links=/wheels --find-links=/wheels_extra '
                            + ' '.join(EXTRA_MODULES.get(repo, [])) + ' > /tmp/pip.log 2>&1"', timeout=900)
            test_patch = task['test_patch']
            if not test_patch.endswith('\n'):
                test_patch += '\n'
            gold = task['patch']
            if not gold.endswith('\n'):
                gold += '\n'
            for label, ap, gp in (('baseline', '', ''), ('agent', agent_patch, ''),
                                  ('gold', '', gold)):
                junit_host = result_dir / f'junit_{label}.xml'
                arm = run_arm(container, label, ap, gp, test_patch, junit_host,
                              targets=targets, required_names=required_names)
                result['arms'][label] = arm
                statuses = arm.get('required_statuses') or {}
                print(f'  {label}: {arm.get("verdict")} required={statuses}', flush=True)
        finally:
            P.sh(f'docker rm -f atlas-v1-{iid}')

        result['resolution'] = resolution(result['arms'])
        result['fingerprint'] = cache_key
        result['elapsed_seconds'] = round(time.time() - started, 1)
        (result_dir / 'verification.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        combined[ledger_key] = {'fingerprint': cache_key, 'evidence_dir': str(result_dir),
                         'agent_patch_chars': result['agent_patch_chars'],
                         'resolution': result['resolution'],
                         'elapsed_seconds': result['elapsed_seconds']}
        combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + '\n',
                                 encoding='utf-8')
        print(f'  resolved={result["resolution"]["resolved"]}', flush=True)
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('combined:', json.dumps(combined, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
