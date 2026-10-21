"""Safety and determinism contracts for the standalone experiment runners."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
timing = importlib.import_module("p1_timing")
p1d = importlib.import_module("p1d_h1_container")
h2 = importlib.import_module("p1c_h2_localization")


def test_all_tasks_selects_129_and_legacy_all_stays_four():
    select = getattr(timing, "select_task_ids", None)
    assert callable(select), "timing needs explicit task-selection support"
    tasks = {json.loads(line)["instance_id"]: json.loads(line)
             for line in (ROOT / "data/tasks.jsonl").read_text(encoding="utf-8").splitlines()}
    assert len(select(tasks, all_tasks=True)) == 129
    assert select(tasks, legacy_all=True) == ["httpx_3672", "requests_7205", "rich_3454", "fastapi_11194"]


def test_manifest_handles_crlf_and_rejects_unknown_id(tmp_path):
    select = getattr(timing, "select_task_ids", None)
    assert callable(select), "timing needs explicit task-manifest support"
    manifest = tmp_path / "ids.txt"
    manifest.write_bytes(b"a\r\nb\r\n")
    assert select({"a": {}, "b": {}}, manifest=manifest) == ["a", "b"]
    manifest.write_text("missing", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        select({"a": {}}, manifest=manifest)


def test_timing_existing_output_is_preserved_and_resume_skips_complete(tmp_path, monkeypatch):
    out = tmp_path / "fresh.json"
    complete = {"instance_id": "httpx_3672", "finished": "now",
                "verdict": {"f2p_expected_fail": True, "p2p_expected_pass": True},
                "f2p_exit_code": 1, "p2p_exit_code": 0,
                "f2p_pytest_timed_out": False, "p2p_pytest_timed_out": False,
                "f2p_junit": absolute()["junit"], "p2p_junit": absolute("pass")["junit"]}
    out.write_text(json.dumps({"httpx_3672": complete}), encoding="utf-8")
    before = out.read_bytes()
    monkeypatch.setattr(timing, "run_task", lambda *args: pytest.fail("unexpected Docker execution"))
    with pytest.raises((FileExistsError, SystemExit)):
        timing.main(["httpx_3672", "--output", str(out)])
    assert out.read_bytes() == before
    assert timing.main(["httpx_3672", "--output", str(out), "--junit-dir", str(tmp_path / "junit"), "--resume"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["httpx_3672"] == complete


def test_timing_resume_retries_incomplete_task(tmp_path, monkeypatch):
    out = tmp_path / "fresh.json"
    out.write_text(json.dumps({"httpx_3672": {"fatal": "interrupted"}}), encoding="utf-8")
    monkeypatch.setattr(timing, "run_task", lambda iid, tasks: {"instance_id": iid, "finished": "new", "verdict": {"p2p_expected_pass": False, "f2p_expected_fail": True}})
    assert timing.main(["httpx_3672", "--output", str(out), "--junit-dir", str(tmp_path / "junit"), "--resume"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["httpx_3672"]["finished"] == "new"


def test_original_timing_result_cannot_be_resumed(monkeypatch):
    monkeypatch.setattr(timing, "run_task", lambda *args: pytest.fail("original outputs must be protected"))
    with pytest.raises((ValueError, SystemExit)):
        timing.main(["httpx_3672", "--output", str(ROOT / "results/p1_timing.json"), "--resume"])


def test_other_frozen_result_cannot_be_used_as_resume_output(monkeypatch):
    monkeypatch.setattr(timing, "run_task", lambda *args: pytest.fail("frozen evidence must be protected"))
    with pytest.raises((ValueError, SystemExit)):
        timing.main(["httpx_3672", "--output", str(ROOT / "results/h1_baselines.json"), "--resume", "--dry-run"])


def absolute(verdict="fail"):
    return {"verdict": verdict, "exit_code": 1 if verdict == "fail" else 0,
            "timed_out": False, "n_targets": 1,
            "junit": {"tests": 1, "failures": 1 if verdict == "fail" else 0, "errors": 0, "skipped": 0, "evidence_complete": True, "bad": ["test::one"] if verdict == "fail" else []}}


def test_node_passed_does_not_match_same_function_in_another_file():
    stats = {"cases": [{"classname": "tests.test_other", "name": "test_shared", "status": "passed"}]}
    assert timing.node_passed("tests/test_expected.py::test_shared", stats) is None


def test_node_passed_does_not_accept_first_parameter_when_another_failed():
    stats = {"cases": [
        {"classname": "tests.test_one.TestCase", "name": "test_shared[a]", "status": "passed"},
        {"classname": "tests.test_one.TestCase", "name": "test_shared[b]", "status": "failure"}]}
    assert timing.node_passed("tests/test_one.py::TestCase::test_shared", stats) is False
    assert timing.node_passed("tests/test_one.py::TestCase::test_shared[a]", stats) is True


def test_coarse_parameterized_target_does_not_prove_complete_inventory():
    stats = {"cases": [{"classname": "tests.test_one", "name": "test_shared[a]", "status": "passed"}]}
    assert timing.node_passed("tests/test_one.py::test_shared", stats) is None


@pytest.mark.parametrize("module", [timing, p1d])
def test_partial_xml_cannot_establish_complete_evidence(tmp_path, module):
    path = tmp_path / "partial.xml"
    path.write_text('<testsuite tests="2" failures="0" errors="0" skipped="0"><testcase classname="tests.test_one" name="test_a"/></testsuite>', encoding="utf-8")
    stats = module.junit_stats(path)
    assert stats.get("evidence_complete") is False


def test_timing_resume_retries_timeout_and_missing_xml():
    result = {"finished": "now", "verdict": {"f2p_expected_fail": True, "p2p_expected_pass": False},
              "f2p_exit_code": 1, "p2p_exit_code": 1,
              "f2p_junit": absolute()["junit"], "p2p_junit": absolute()["junit"],
              "p2p_pytest_timed_out": True}
    assert not timing.timing_complete(result)
    result["p2p_pytest_timed_out"] = False
    result["p2p_junit"] = {"error": "missing XML"}
    assert not timing.timing_complete(result)


def test_p1d_all_skipped_is_unknown():
    stats = {"tests": 2, "failures": 0, "errors": 0, "skipped": 2, "bad": [], "evidence_complete": True}
    assert p1d.verdict(0, stats, False) == "unknown"


def test_skipped_subset_is_not_an_absolute_pass():
    stats = {"tests": 2, "failures": 0, "errors": 0, "skipped": 1, "bad": [], "evidence_complete": True}
    assert p1d.verdict(0, stats, False) == "fail"
    assert timing.timing_absolute_verdict(0, stats, False) == "fail"


def test_p1d_delta_zero_tests_and_incomplete_evidence_are_unknown():
    for stats in ({"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "bad": [], "evidence_complete": True},
                  {"tests": 2, "failures": 0, "errors": 0, "skipped": 0, "bad": [], "evidence_complete": False}):
        rec = {"target": {**absolute("pass"), "junit": stats}}
        p1d.apply_baseline_delta(rec, {"target": absolute()}, "target")
        assert rec["target"].get("delta_verdict") == "unknown"


def test_target_cont_auto_only_retries_missing_or_incomplete_xml():
    need = getattr(p1d, "needs_target_cont", None)
    assert callable(need), "P1D needs an explicit fallback predicate"
    assert not need(absolute("pass")["junit"], policy="auto")
    assert need({"error": "missing XML"}, policy="auto")
    assert need({"evidence_complete": False}, policy="auto")
    assert need(absolute("pass")["junit"], policy="1")
    assert not need({"error": "missing XML"}, policy="0")


def test_p1d_completed_baseline_needs_absolute_evidence_not_delta():
    complete = getattr(p1d, "task_complete", None)
    assert callable(complete), "P1D needs separate baseline completion semantics"
    runs = {"baseline": {"target": absolute()}, "gold": {"target": {**absolute("pass"), "delta_verdict": "clean"}}}
    assert complete({"runs": runs}, ["baseline", "gold"], modes=["target"])
    broken = copy.deepcopy(runs)
    del broken["baseline"]["target"]["junit"]
    assert not complete({"runs": broken}, ["baseline", "gold"], modes=["target"])


def test_p1d_resume_rejects_partial_absolute_junit_counts():
    record = absolute()
    del record["junit"]["failures"]
    assert not p1d.task_complete({"runs": {"baseline": {"target": record}}},
                                 ["baseline"], modes=["target"])


def test_p1d_cli_resume_skips_complete_and_preserves_historical_junit(tmp_path, monkeypatch):
    out = tmp_path / "new_run.json"
    result = {"runs": {"baseline": {"target": absolute()},
                       "gold": {"target": {**absolute("pass"), "delta_verdict": "clean"}}}}
    out.write_text(json.dumps({"httpx_3672": result}), encoding="utf-8")
    before = out.read_bytes()
    monkeypatch.setenv("P1D_VARIANTS", "baseline,gold")
    monkeypatch.setenv("P1D_FULL", "0")
    monkeypatch.setenv("P1D_SELECTED", "0")
    monkeypatch.setattr(p1d, "run_task", lambda *args: pytest.fail("unexpected Docker execution"))
    with pytest.raises(SystemExit):
        p1d.main(["httpx_3672", "--output", str(out)])
    assert out.read_bytes() == before
    assert p1d.main(["httpx_3672", "--output", str(out), "--resume", "--junit-dir", str(tmp_path / "junit")]) == 0
    assert out.read_bytes() == before
    with pytest.raises(SystemExit):
        p1d.main(["httpx_3672", "--output", str(tmp_path / "another.json"), "--junit-dir", str(ROOT / "results/p1d_junit")])


def test_p1d_unknown_delta_is_recorded_and_retryable():
    rec = {"target": absolute()}
    rec["target"]["junit"] = {"error": "missing XML"}
    p1d.apply_baseline_delta(rec, {"target": absolute()}, "target")
    assert rec["target"].get("delta_verdict") == "unknown"
    complete = getattr(p1d, "task_complete", None)
    assert callable(complete)
    task = {"finished": "now", "runs": {"baseline": {"target": absolute()}, "gold": rec}}
    assert not complete(task, ["baseline", "gold"], modes=["target"])
    assert complete(task, ["baseline", "gold"], modes=["target"], keep_unknown=True)


def test_p1d_summary_does_not_report_unknown_unknown_as_agreement(capsys):
    unknown = {**absolute(), "delta_verdict": "unknown"}
    p1d.summarize({"a": {"runs": {"gold": {"target": unknown, "selected": unknown}}}})
    output = capsys.readouterr().out
    assert "差分判定一致率" not in output


def test_h2_equal_scores_keep_archive_order():
    rank = getattr(h2, "rank_scores", None)
    assert callable(rank), "H2 needs deterministic tied-score ranking"
    assert list(rank(np.zeros(32))) == list(range(32))


def test_h2_cli_preserves_existing_output(tmp_path):
    assert callable(getattr(h2, "rank_scores", None)), "new CLI required before running the standalone command"
    out = tmp_path / "existing.json"
    out.write_bytes(b"preserve me")
    run = subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / "paper/scripts/p1c_h2_localization.py"), "--output", str(out)], capture_output=True, text=True)
    assert run.returncode != 0
    assert out.read_bytes() == b"preserve me"


def test_h2_hash_seed_does_not_change_metrics(tmp_path):
    graphs, emb = tmp_path / "graphs", tmp_path / "emb"
    graphs.mkdir()
    emb.mkdir()
    nodes = [f"pkg.file{i}.symbol" for i in range(32)]
    graph = {"nodes": [{"id": n, "text": "a" * (i + 1)} for i, n in enumerate(nodes)],
             "edges": [{"source": nodes[0], "target": n, "type": "calls"} for n in nodes[1:]]}
    (graphs / "repo_commit.json").write_text(json.dumps(graph), encoding="utf-8")
    np.savez(emb / "repo_commit.npz", **{n: np.ones(2) for n in nodes})
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"instance_id": "test", "repo": "org/repo", "base_commit": "commit", "problem_statement": "unmatched query", "patch": "diff --git a/pkg/file31.py b/pkg/file31.py\n--- a/pkg/file31.py\n+++ b/pkg/file31.py\n@@ -1 +1 @@\n-x\n+y\n"}) + "\n", encoding="utf-8")
    reports = []
    for seed in ("0", "42"):
        out = tmp_path / f"seed{seed}.json"
        code = f"import sys; sys.path.insert(0, {str(ROOT / 'paper/scripts')!r}); import p1c_h2_localization as h; h.TASKS={str(tasks)!r}; h.GRAPHS={str(graphs)!r}; h.EMB={str(emb)!r}; h.main(['--output', {str(out)!r}])"
        run = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code], env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        report = json.loads(out.read_text(encoding="utf-8"))
        report.pop("generated", None)
        report.pop("run_metadata", None)
        reports.append(report)
    assert reports[0]["n_tasks_evaluated"] == 1
    assert reports[0] == reports[1]
