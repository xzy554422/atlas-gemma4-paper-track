"""Direction 4 (G4.2): build the dependency-cache manifest.

Maps each repo to its validated extra-module set and the wheel files (with sha256)
that satisfy them, plus the residual baseline-failure classification.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WHEEL_DIRS = [ROOT / 'data' / 'wheels_local', ROOT / 'data' / 'wheels_extra']
OUT = ROOT / 'results' / 'dep_cache' / 'manifest.json'

EXTRA_DEPS = {
    'Textualize/rich': ['attrs'],
    'fastapi/fastapi': ['attrs', 'inline-snapshot', 'dirty-equals', 'orjson', 'sqlalchemy',
                        'pydantic-settings', 'a2wsgi', 'strawberry-graphql', 'flask',
                        'sqlmodel', 'pyyaml', 'email-validator', 'pyjwt', 'passlib',
                        'bcrypt', 'ujson'],
    'psf/requests': ['pytest-httpbin'],
    'encode/httpx': ['pytest-httpbin'],
}

RESIDUALS = {
    'Textualize/rich':
        '14 baseline failures: pygments-version-sensitive rendering tests '
        '(test_syntax/test_pretty family); irreducible without pinning older pygments.',
    'fastapi/fastapi':
        '5 baseline failures: 3 async tests require a plugin disabled by the official '
        '`-p no:anyio` flag; 1 fastapi-cli path assertion (environment); 1 openapi '
        'schema drift (library version). 131 skipped are upstream skips.',
    'psf/requests':
        '12 baseline failures: 8 proxy-environment tests and 4 connect-timeout tests '
        'that require network behaviour unavailable in the offline sandbox '
        '(--network=none).',
    'encode/httpx': 'fully green baseline (128/128).',
}


def main() -> int:
    wheels = {}
    for directory in WHEEL_DIRS:
        for wheel in sorted(directory.glob('*.whl')):
            data = wheel.read_bytes()
            wheels[wheel.name] = {
                'dir': directory.name,
                'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest(),
            }
    manifest = {
        'schema_version': 1,
        'generated': '2026-10-25',
        'validated_on': 'swebench-sandbox:latest, python 3.13, official hermetic pytest command',
        'repos': {},
        'wheel_files': wheels,
        'wheel_count': len(wheels),
        'residual_baseline_failures': RESIDUALS,
    }
    import re

    def dashify(name: str) -> str:
        return re.sub(r'[-_.]+', '-', name).lower()

    dashed = {dashify(name): name for name in wheels}
    for repo, modules in EXTRA_DEPS.items():
        entry = {'extra_modules': modules, 'wheels': []}
        for module in modules:
            stem = dashify(module)
            matches = [orig for dashed_name, orig in dashed.items()
                       if dashed_name.startswith(stem + '-')]
            entry['wheels'].extend(matches or [f'MISSING:{module}'])
        manifest['repos'][repo] = entry
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    missing = [w for e in manifest['repos'].values() for w in e['wheels'] if w.startswith('MISSING')]
    print(f'repos: {len(manifest["repos"])} | wheel files indexed: {len(wheels)} | missing: {missing}')
    print('written:', OUT)
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
