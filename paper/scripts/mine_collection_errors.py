"""Direction 4, step 1: mine historical JUnit receipts for collection-error signatures.

Groups per-repo error messages from results/p1d_junit/*.xml (and the frozen 2026-10-17
copy), extracts missing-module names, and writes results/dep_cache/historical_errors.json.
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JUNIT_DIRS = [ROOT / 'results' / 'p1d_junit', ROOT / 'results' / '_frozen_2026-10-17']
OUT = ROOT / 'results' / 'dep_cache' / 'historical_errors.json'

MODULE_RE = re.compile(r"ModuleNotFoundError: No module named '([a-zA-Z0-9_.]+)'")
IMPORT_RE = re.compile(r"ImportError: cannot import name '[^']+' from '([a-zA-Z0-9_.]+)'")
GENERIC_IMPORT_RE = re.compile(r"ModuleNotFoundError: No module named '([a-zA-Z0-9_.]+)'")


def repo_of(path: Path) -> str:
    name = path.name.split('_')[0]
    return {'rich': 'Textualize/rich', 'httpx': 'encode/httpx',
            'fastapi': 'fastapi/fastapi', 'requests': 'psf/requests'}.get(name, name)


def main() -> int:
    per_repo = defaultdict(lambda: {'files': 0, 'with_collection_error': 0,
                                    'modules': Counter(), 'samples': []})
    scanned = 0
    for junit_dir in JUNIT_DIRS:
        if not junit_dir.is_dir():
            continue
        for path in junit_dir.glob('*.xml'):
            rel = path.name
            repo = repo_of(path)
            bucket = per_repo[repo]
            bucket['files'] += 1
            scanned += 1
            try:
                root = ET.parse(path).getroot()
            except ET.ParseError:
                continue
            text_fragments = []
            for err in root.iter('error'):
                text_fragments.append((err.get('message') or '') + '\n' + (err.text or ''))
            for case in root.iter('testcase'):
                for failure in case.iter('failure'):
                    text_fragments.append((failure.get('message') or ''))
            joined = '\n'.join(text_fragments)
            if 'collection' not in joined and 'ModuleNotFoundError' not in joined \
                    and 'ImportError' not in joined:
                continue
            found = set(MODULE_RE.findall(joined)) | set(IMPORT_RE.findall(joined))
            if not found:
                continue
            bucket['with_collection_error'] += 1
            bucket['modules'].update(found)
            if len(bucket['samples']) < 3:
                bucket['samples'].append({'file': rel, 'modules': sorted(found),
                                          'head': joined[:220]})

    out = {'scanned_files': scanned,
           'per_repo': {repo: {'files': b['files'],
                               'with_collection_error': b['with_collection_error'],
                               'modules': dict(b['modules'].most_common()),
                               'samples': b['samples']}
                        for repo, b in sorted(per_repo.items())}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'scanned {scanned} junit files')
    for repo, b in sorted(out['per_repo'].items()):
        print(f"{repo:<18} files={b['files']:<4} collection_errors={b['with_collection_error']:<4} "
              f"modules={list(b['modules'])[:8]}")
    print('written:', OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
