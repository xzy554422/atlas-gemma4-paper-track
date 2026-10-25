"""Direction 1 (G1.2): compile the resolution-rate summary from all verifications."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VER = ROOT / 'results' / 'dep_cache' / 'd1_verifications.json'
BATCHES = {k: ROOT / 'results/final_kaggle' / f'batch_{k}' / 'batch_summary.json'
           for k in ('rich', 'fastapi', 'requests', 'httpx')}
OUT = ROOT / 'results' / 'dep_cache' / 'd1_resolution_summary.json'


def main() -> int:
    ver = json.loads(VER.read_text(encoding='utf-8'))
    runs = {}
    for repo_batch, path in BATCHES.items():
        if not path.is_file():
            continue
        d = json.loads(path.read_text(encoding='utf-8'))
        for iid, t in d['tasks'].items():
            runs[iid] = t

    rows = []
    for iid, v in sorted(ver.items()):
        run = runs.get(iid, {})
        repo = next((t['repo'] for t in json.loads(
            (ROOT / 'results/dep_cache/d1_task_sample.json').read_text(encoding='utf-8'))
            if t['instance_id'] == iid), '?')
        res = v.get('resolution') or {}
        rows.append({
            'instance_id': iid, 'repo': repo,
            'patch_chars': v.get('agent_patch_chars', run.get('patch_chars', 0)),
            'termination': run.get('termination'),
            'turns': run.get('turns'), 'tool_calls': run.get('tool_calls_used'),
            'output_tokens': (run.get('totals') or {}).get('output_tokens'),
            'loop_seconds': run.get('elapsed_seconds'),
            'verified': 'resolution' in v,
            'resolved': res.get('resolved', False),
            'outcome': (
                'resolved' if res.get('resolved')
                else 'not_resolved' if res.get('agent_resolves') is False and res.get('gold_resolves')
                else 'timeout_unknown' if res.get('agent_required') in ('no_junit', 'unknown')
                else 'not_verifiable' if not v.get('verified', True)
                else 'unverified'),
        })

    def rate(subset):
        verified = [r for r in subset if r['outcome'] in ('resolved', 'not_resolved')]
        resolved = sum(1 for r in verified if r['outcome'] == 'resolved')
        return f'{resolved}/{len(verified)}' + (f' ({resolved/len(verified)*100:.1f}%)' if verified else '')

    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r['repo']].append(r)
    summary = {
        'schema_version': 1,
        'configuration': 'E4B qat-w4a16 @ T4, FP16, TP1, reduced harness protocol, '
                         '24 tool calls / 20 min budgets; local three-arm verification '
                         '(baseline/agent/gold) with competition-style resolution criterion',
        'tasks_total': len(rows),
        'agent_runs_completed': sum(1 for r in rows if r['turns']),
        'patches_nonempty': sum(1 for r in rows if r['patch_chars'] > 0),
        'patches_empty': sum(1 for r in rows if r['patch_chars'] == 0),
        'resolution_overall_verified': rate(rows),
        'resolution_overall_strict_all_tasks': (
            f"{sum(1 for r in rows if r['outcome'] == 'resolved')}/{len(rows)}"),
        'by_repo': {repo: {'resolution_verified': rate(rs),
                           'tasks': len(rs),
                           'empty_patches': sum(1 for r in rs if r['patch_chars'] == 0),
                           'mean_turns': round(sum(r['turns'] or 0 for r in rs) / len(rs), 1),
                           'mean_output_tokens': round(sum(r['output_tokens'] or 0 for r in rs) / len(rs))}
                    for repo, rs in sorted(by_repo.items())},
        'failure_modes': dict(Counter(
            'empty_patch' if r['patch_chars'] == 0 else
            'timeout_unknown' if r['outcome'] == 'timeout_unknown' else
            'resolved' if r['outcome'] == 'resolved' else
            'not_verifiable' if r['outcome'] == 'not_verifiable' else 'submitted_not_resolved'
            for r in rows)),
        'rows': rows,
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in (
        'tasks_total', 'patches_nonempty', 'resolution_overall_verified',
        'resolution_overall_strict_all_tasks', 'by_repo', 'failure_modes')},
        ensure_ascii=False, indent=1))
    print('written:', OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
