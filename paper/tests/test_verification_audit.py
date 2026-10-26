"""Focused regressions for local three-arm verification evidence."""
import sys
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import verify_agent_patch as V
import verify_batch as B


def xml(path, cases):
    suite = ET.Element('testsuite', tests=str(len(cases)), failures=str(sum(s == 'failure' for _, _, s in cases)),
                       errors=str(sum(s == 'error' for _, _, s in cases)), skipped=str(sum(s == 'skipped' for _, _, s in cases)))
    for cls, name, status in cases:
        case = ET.SubElement(suite, 'testcase', classname=cls, name=name)
        if status != 'passed':
            ET.SubElement(case, status)
    ET.ElementTree(suite).write(path)


@pytest.mark.parametrize('bad', ['failure', 'error', 'skipped'])
def test_all_parameters_and_identity(tmp_path, bad):
    path = tmp_path / 'receipt.xml'
    xml(path, [('tests.test_x.Other', 'test_it[a]', 'passed'),
               ('tests.test_x.TestX', 'test_it[a]', 'passed'),
               ('tests.test_x.TestX', 'test_it[b]', bad)])
    assert V.node_status(path, 'tests/test_x.py::TestX::test_it') != 'passed'
    assert V.node_status(path, 'tests/test_x.py::TestX::test_it[a]') == 'passed'
    assert V.node_status(path, 'tests/test_y.py::TestX::test_it') == 'missing'


def arm(status='passed', rc=0, failures=0, errors=0):
    return dict(exit_code=rc, timed_out=False, required_statuses={'tests/test_x.py::test_it': status},
                junit=dict(tests=2, failures=failures, errors=errors, skipped=0, evidence_complete=True))


@pytest.mark.parametrize('mutation', [dict(exit_code=1), dict(timed_out=True),
                                      dict(junit={}), dict(junit=dict(tests=2, failures=1, errors=0, skipped=0, evidence_complete=True))])
def test_acceptance_requires_complete_success(mutation):
    a = arm()
    a.update(mutation)
    assert not V.accepted(a)


def test_gold_and_actual_baseline_failure_required():
    base = arm('failed', 1, failures=1)
    assert V.resolution(dict(baseline=base, agent=arm(), gold=arm()))['resolved']
    for baseline, gold in [(arm('error', 1, errors=1), arm()), (base, arm(rc=2))]:
        result = V.resolution(dict(baseline=baseline, agent=arm(), gold=gold))
        assert not result['resolved']
        assert result['status'] == 'unknown'


def test_receipt_destination_and_freshness(tmp_path, monkeypatch):
    receipt = tmp_path / 'new' / 'agent.xml'
    receipt.parent.mkdir()
    receipt.write_text('stale')
    calls = []
    monkeypatch.setattr(V.P, 'dm', lambda c, cmd: calls.append(cmd) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(V.P, 'reset_workspace', lambda c: None)
    monkeypatch.setattr(V.P, 'apply_patch', lambda *args: (True, 'ok'))
    def run(*args, **kwargs):
        assert not receipt.exists()
        assert calls == ['rm -f /tmp/_verify_agent.xml']
        return 0, 0, '', False
    monkeypatch.setattr(V.P, 'run_pytest', run)
    def copy(cmd):
        assert V.P.host_path(str(receipt)) in cmd
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(V.P, 'sh', copy)
    result = V.run_arm('container', 'agent', '', '', 'test patch', receipt)
    assert not receipt.exists()
    assert not V.accepted(result)


def test_targets_have_lexical_identity(tmp_path):
    import subprocess
    import tarfile
    source = tmp_path / 'source'
    (source / 'tests').mkdir(parents=True)
    old = 'class TestA:\n    def test_same(self):\n        assert True\n\ndef test_same():\n    assert True\n'
    (source / 'tests/test_x.py').write_text(old)
    snapshot = tmp_path / 'snapshot.tgz'
    with tarfile.open(snapshot, 'w:gz') as tar:
        tar.add(source / 'tests', arcname='tests')
    patch = 'diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1,6 +1,6 @@\n class TestA:\n     def test_same(self):\n-        assert True\n+        assert False\n \n def test_same():\n     assert True\n'
    assert B.targets_with_class(snapshot, patch) == ['tests/test_x.py::TestA::test_same']


def test_empty_patch_ledger_is_isolated(tmp_path, monkeypatch):
    import json
    root = tmp_path / 'root'
    (root / 'data').mkdir(parents=True)
    (root / 'data/tasks.jsonl').write_text(json.dumps(dict(instance_id='x', repo='repo')) + '\n')
    batch = tmp_path / 'batch'
    (batch / 'x').mkdir(parents=True)
    (batch / 'x/agent_patch.diff').write_text('')
    output = tmp_path / 'new_output'
    monkeypatch.setattr(B, 'ROOT', root)
    monkeypatch.setattr(B, 'fingerprint', lambda *args: 'hash')
    monkeypatch.setattr(sys, 'argv', ['verify_batch', str(batch), '--output-dir', str(output)])
    assert B.main() == 0
    assert json.loads((output / 'ledger.json').read_text())['batch/x']['status'] == 'unknown'
    assert not (root / 'results/dep_cache/d1_verifications.json').exists()


def test_fingerprint_invalidates_patch_test_and_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(B, 'ROOT', tmp_path)
    monkeypatch.setattr(B.P, 'sh', lambda cmd: SimpleNamespace(returncode=0, stdout='image-A'))
    task = dict(instance_id='x', test_patch='test-A', patch='gold')
    first = B.fingerprint(task, 'patch-A')
    assert B.fingerprint(task, 'patch-B') != first
    assert B.fingerprint(dict(task, test_patch='test-B'), 'patch-A') != first
    wheels = tmp_path / 'data/wheels_extra'
    wheels.mkdir(parents=True)
    (wheels / 'dependency.whl').write_bytes(b'changed-environment')
    assert B.fingerprint(task, 'patch-A') != first


def test_bare_function_name_cannot_match(tmp_path):
    path = tmp_path / 'receipt.xml'
    xml(path, [('tests.test_x', 'test_it', 'passed')])
    assert V.node_status(path, 'test_it') == 'missing'
