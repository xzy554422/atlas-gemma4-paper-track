"""Direction 3 summary: verification-guidance + run_command ablation, three arms.

Arms (same 10-task subset, identical budgets):
  control : Direction 1 runs (no run_command, no guidance)
  d3_sel  : run_command + selected-test guidance
  d3_full : run_command + full-file guidance
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VER = ROOT / 'results/dep_cache/d1_verifications.json'
D1 = ROOT / 'results/dep_cache/d1_resolution_summary.json'
OUT = ROOT / 'results/dep_cache/d3_ablation_summary.json'
SUBSET = [t['instance_id'] for t in json.loads(
    (ROOT / 'results/dep_cache/d2_task_subset.json').read_text(encoding='utf-8'))]

ARM_DIRS = {
    'd3_sel': ROOT / 'results/final_kaggle/d3_collector/atlas-agent-d3-sel',
    'd3_full': ROOT / 'results/final_kaggle/d3_full_collector/atlas-agent-d3-full',
}


def main() -> int:
    ver = json.loads(VER.read_text(encoding='utf-8'))
    d1 = json.loads(D1.read_text(encoding='utf-8'))
    control = {r['instance_id']: r for r in d1['rows'] if r['instance_id'] in SUBSET}

    arms = {'control': {}}
    for iid in SUBSET:
        row = control[iid]
        arms['control'][iid] = {'patch_chars': row['patch_chars'],
                                'resolved': row['resolved'],
                                'termination': row['termination']}
    for arm, path in ARM_DIRS.items():
        summary_path = path / 'batch_summary.json'
        batch = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {'tasks': {}}
        arms[arm] = {}
        for iid in SUBSET:
            t = batch['tasks'].get(iid, {})
            v = ver.get(f'{path.name}/{iid}', {})
            res = v.get('resolution', {})
            arms[arm][iid] = {
                'patch_chars': t.get('patch_chars'),
                'termination': t.get('termination'),
                'error_type': t.get('error_type'),
                'verified': bool(res),
                'resolved': res.get('resolved', False),
            }

    def resolved_count(arm):
        return sum(1 for iid in SUBSET if arms[arm][iid].get('resolved'))

    def nonempty(arm):
        return sum(1 for iid in SUBSET if (arms[arm][iid].get('patch_chars') or 0) > 0)

    def verified(arm):
        if arm == 'control':
            return sum(1 for iid in SUBSET if control[iid].get('verified'))
        return sum(1 for iid in SUBSET if arms[arm][iid].get('verified'))

    summary = {
        'schema_version': 1,
        'design': 'paired ablation on the 10-task subset: control (D1) vs +run_command '
                  '+selected-test guidance vs +run_command +full-file guidance',
        'tasks': SUBSET,
        'arms': arms,
        'resolution_by_arm': {arm: f'{resolved_count(arm)}/{verified(arm)} verified '
                                   f'({resolved_count(arm)}/{len(SUBSET)} of all)'
                              for arm in arms},
        'nonempty_patches_by_arm': {arm: nonempty(arm) for arm in arms},
        'findings': [
            'run_command + verification guidance did not increase resolution: '
            f'control {resolved_count("control")}/10, d3_sel {resolved_count("d3_sel")}/10, '
            f'd3_full {resolved_count("d3_full")}/10 (verified denominators '
            f'{verified("control")}/{verified("d3_sel")}/{verified("d3_full")}).',
            'requests_6589 flipped resolved->not-resolved under d3_sel (patch 383 vs 546 '
            'chars): trajectory variance across arms is real.',
            'httpx_3672 produced its first non-empty patch (2,106 chars) under d3_full '
            'but did not pass the required tests.',
            'd3_full hit a runner NameError on 5/10 tasks (recorded per-task); d3_sel on '
            '4/10 — a defect in the extended-tooling path to fix before reuse, not a '
            'model result.',
        ],
        'scope': 'descriptive, n=10 per arm; no significance claims',
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary['resolution_by_arm'], indent=1))
    print(json.dumps(summary['nonempty_patches_by_arm'], indent=1))
    print('written:', OUT)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
