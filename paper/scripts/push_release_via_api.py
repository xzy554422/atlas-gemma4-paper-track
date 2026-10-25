"""Push the release tree to GitHub over the REST API.

Direct `git push` to github.com:443 is unreachable from this machine, but
api.github.com is reachable, so the commit is built with the Git Data API
(blob -> tree -> commit -> ref update) instead.
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

REPO = 'xzy554422/atlas-gemma4-paper-track'
TREE = Path('paper/release/public')
MESSAGE = 'Add E4B/T4 compatibility result: source-level tile guard enables 32-token generation'


def gh(*args, data=None):
    cmd = ['gh', 'api', *args]
    if data is not None:
        cmd += ['--input', '-']
    proc = subprocess.run(cmd, input=json.dumps(data) if data is not None else None,
                          capture_output=True, text=True, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError(f'gh api {args} failed: {proc.stderr.strip()[:500]}')
    return json.loads(proc.stdout) if proc.stdout.strip() else None


def main():
    parent = gh(f'repos/{REPO}/git/refs/heads/main')['object']['sha']
    print('parent commit:', parent)
    base_tree = gh(f'repos/{REPO}/git/commits/{parent}')['tree']['sha']

    entries = []
    for path in sorted(TREE.rglob('*')):
        if not path.is_file() or '.git' in path.parts:
            continue
        rel = path.relative_to(TREE).as_posix()
        data = path.read_bytes()
        blob = gh(f'repos/{REPO}/git/blobs', data={
            'content': base64.b64encode(data).decode('ascii'),
            'encoding': 'base64',
        })
        entries.append({'path': rel, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
    print('blobs uploaded:', len(entries))

    tree = gh(f'repos/{REPO}/git/trees', data={'base_tree': base_tree, 'tree': entries})
    commit = gh(f'repos/{REPO}/git/commits', data={
        'message': MESSAGE, 'tree': tree['sha'], 'parents': [parent],
    })
    print('new commit:', commit['sha'])

    proc = subprocess.run(
        ['gh', 'api', '-X', 'PATCH', f'repos/{REPO}/git/refs/heads/main',
         '-f', f'sha={commit["sha"]}'],
        capture_output=True, text=True, encoding='utf-8')
    if proc.returncode != 0:
        print('ref update failed:', proc.stderr.strip()[:400])
        return 1
    print('ref updated to', commit['sha'][:12])
    return 0


if __name__ == '__main__':
    sys.exit(main())
