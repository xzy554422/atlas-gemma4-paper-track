"""Audit regressions: denominator leakage, fabricated passes and unstable provenance."""
import importlib.util
import json
from pathlib import Path

import pytest


def load_audit():
    path = Path(__file__).parents[1] / "scripts" / "final_audit.py"
    assert path.exists(), "The final audit implementation has not been written"
    spec = importlib.util.spec_from_file_location("final_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mode(sec=10, delta="clean", **overrides):
    result = {"sec": sec, "delta_verdict": delta, "exit_code": 0,
              "timed_out": False, "n_targets": 1,
              "junit": {"tests": 1, "failures": 0, "errors": 0,
                        "skipped": 0, "bad": []}}
    result.update(overrides)
    return result


def record(**runs):
    return {"instance_id": "repo_1", "repo": "org/repo",
            "started": "2026-10-17T01:00:00", "finished": "2026-10-17T02:00:00",
            "test_patch_applied": True, "target_nodes": ["tests/test_a.py::test_a"],
            "runs": {"baseline": {"tp_applied": True, "target": mode(),
                                   "selected": mode(), "full": mode()}, **runs}}


def test_failed_injection_excluded_from_agreement_and_cost():
    audit = load_audit()
    records = {"repo_1": record(syntax_break={"perturb_ok": False,
               "gold_applied": True, "tp_applied": True,
               "target": mode(), "selected": mode(2), "full": mode(10)})}
    rows = audit.make_rows(records)
    summary = audit.summarize(rows, records)
    assert summary["nonbaseline"]["injection_failed"] == 1
    assert summary["nonbaseline"]["paired_target_selected"] == 0
    assert audit.cost_table(rows)[0]["n_pairs"] == 0


def test_unknown_is_not_agreement_or_pass():
    audit = load_audit()
    records = {"repo_1": record(gold={"gold_applied": True, "tp_applied": True,
               "target": mode(), "selected": {"n_targets": 0, "verdict": "empty"},
               "full": mode()})}
    rows = audit.make_rows(records)
    row = next(x for x in rows if x["variant"] == "gold")
    assert row["selected_absolute"] == "unknown"
    assert row["selected_delta"] == "unknown"
    assert row["target_required_status"] == "unknown"
    assert audit.summarize(rows, records)["nonbaseline"]["paired_target_selected"] == 0


def test_duplicates_choose_timestamp_before_quality_or_filename(tmp_path):
    audit = load_audit()
    older = record(gold={"target": mode()})
    newer = record(gold={"target": mode(3)})
    older["finished"] = "2026-10-16T23:59:59"
    (tmp_path / "p1d_h1_decision.r9.json").write_text(json.dumps({"repo_1": older}))
    (tmp_path / "p1d_h1_decision.json").write_text(json.dumps({"repo_1": newer}))
    chosen, provenance = audit.load_records(tmp_path)
    assert chosen["repo_1"]["runs"]["gold"]["target"]["sec"] == 3
    assert provenance[0]["selected_source"] == "p1d_h1_decision.json"
    assert provenance[0]["candidate_count"] == 2
    # Identical timestamps with missing dates use quality before explicit source priority.
    newer.pop("finished"); newer.pop("started")
    older.pop("finished"); older.pop("started")
    newer["runs"]["gold"]["selected"] = mode()
    (tmp_path / "p1d_h1_decision.r9.json").write_text(json.dumps({"repo_1": older}))
    (tmp_path / "p1d_h1_decision.json").write_text(json.dumps({"repo_1": newer}))
    chosen2, provenance2 = audit.load_records(tmp_path)
    assert "selected" in chosen2["repo_1"]["runs"]["gold"]
    assert provenance2[0]["timestamp_status"] == "missing"
    # File enumeration order cannot change selection.
    assert audit.load_records(tmp_path) == (chosen2, provenance2)


def test_required_test_gap_and_skipped_param_never_pass(tmp_path):
    audit = load_audit()
    xml = tmp_path / "j.xml"
    xml.write_text('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1">'
                   '<testcase classname="tests.test_a" name="test_a[one]"/>'
                   '<testcase classname="tests.test_a" name="test_a[two]"><skipped/></testcase>'
                   '</testsuite></testsuites>')
    observed = mode(junit={"tests": 2, "failures": 0, "errors": 0, "skipped": 1, "bad": []})
    result = audit.evaluate_mode(observed, ["tests/test_a.py::test_a[one]",
                                          "tests/test_a.py::test_a[two]"], xml)
    assert result["absolute"] == "pass"
    assert result["required_status"] == "fail"
    missing = audit.evaluate_mode(observed, ["tests/test_a.py::test_missing"], xml)
    assert missing["required_status"] == "unknown"
    coarse = audit.evaluate_mode(observed, ["tests/test_a.py::test_a[one]", "tests/test_a.py::test_missing"], xml)
    assert coarse["required_status"] == "unknown"  # no complete parameter inventory


def test_missing_junit_and_nonzero_exit_cannot_pass():
    audit = load_audit()
    assert audit.evaluate_mode({"exit_code": 0}, None)["absolute"] == "unknown"
    assert audit.evaluate_mode(mode(exit_code=4), None)["absolute"] == "fail"
    assert audit.evaluate_mode(mode(timed_out=True), None)["absolute"] == "timeout"


@pytest.mark.parametrize("stats", [
    {"tests": 1},
    {"tests": 1, "failures": 0, "errors": 0, "skipped": -1},
    {"tests": "1", "failures": 0, "errors": 0, "skipped": 0},
    {"tests": 1, "failures": False, "errors": 0, "skipped": 0},
])
def test_incomplete_or_invalid_junit_counts_are_unknown(stats):
    audit = load_audit()
    assert audit.evaluate_mode(mode(junit=stats), None)["absolute"] == "unknown"


def test_unknown_unknown_is_not_an_agreement():
    audit = load_audit()
    observed = record(gold={"gold_applied": True, "tp_applied": True,
                            "target": {}, "selected": {}, "full": {}})
    result = audit.summarize(audit.make_rows({"repo_1": observed}), {"repo_1": observed})
    assert result["nonbaseline"]["paired_target_selected"] == 0
    assert result["nonbaseline"]["agree_target_selected"] == 0


def test_incomplete_xml_cannot_prove_required_inventory(tmp_path):
    audit = load_audit()
    xml = tmp_path / "incomplete.xml"
    xml.write_text('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0">'
                   '<testcase classname="tests.test_a" name="test_a"/>'
                   '</testsuite></testsuites>')
    observed = mode(junit={"tests": 2, "failures": 0, "errors": 0, "skipped": 0})
    result = audit.evaluate_mode(observed, ["tests/test_a.py::test_a"], xml)
    assert result["required_status"] == "unknown"
    assert result["raw_junit_status"] == "incomplete"


def test_complete_param_inventory_and_coarse_inventory_have_different_proof(tmp_path):
    audit = load_audit()
    xml = tmp_path / "params.xml"
    xml.write_text('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0">'
                   '<testcase classname="tests.test_a" name="test_a[one]"/>'
                   '<testcase classname="tests.test_a" name="test_a[two]"/>'
                   '</testsuite></testsuites>')
    observed = mode(junit={"tests": 2, "failures": 0, "errors": 0, "skipped": 0})
    assert audit.evaluate_mode(observed, ["tests/test_a.py::test_a[one]", "tests/test_a.py::test_a[two]"], xml)["required_status"] == "pass"
    assert audit.evaluate_mode(observed, ["tests/test_a.py::test_a"], xml)["required_status"] == "unknown"


def test_collection_errors_are_distinct_from_fixture_setup_errors(tmp_path):
    audit = load_audit()
    xml = tmp_path / "errors.xml"
    xml.write_text('<testsuites><testsuite tests="2" failures="0" errors="2" skipped="0">'
                   '<testcase name="a"><error message="collection failure"/></testcase>'
                   '<testcase name="b"><error message="failed on setup"/></testcase>'
                   '</testsuite></testsuites>')
    observed = mode(exit_code=1, junit={"tests": 2, "failures": 0, "errors": 2, "skipped": 0})
    result = audit.evaluate_mode(observed, None, xml)
    assert result["collection_errors"] == 1


def test_equal_timestamp_source_priority_and_known_timestamp_beat_missing(tmp_path):
    audit = load_audit()
    first, resumed = record(), record()
    resumed["label"] = "resume"
    (tmp_path / "p1d_h1_decision.json").write_text(json.dumps({"repo_1": first}))
    (tmp_path / "p1d_h1_decision.r2.json").write_text(json.dumps({"repo_1": resumed}))
    assert audit.load_records(tmp_path)[0]["repo_1"]["label"] == "resume"
    resumed.pop("finished"); resumed.pop("started")
    (tmp_path / "p1d_h1_decision.r2.json").write_text(json.dumps({"repo_1": resumed}))
    assert "label" not in audit.load_records(tmp_path)[0]["repo_1"]


def test_resume_never_borrows_junit_from_older_run(tmp_path):
    audit = load_audit()
    directory = tmp_path / "results"
    (directory / "p1d_junit").mkdir(parents=True)
    (directory / "p1d_junit/repo_1_gold_target.xml").write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="tests.test_a" name="test_a"/></testsuite></testsuites>')
    observed = record(gold={"gold_applied": True, "tp_applied": True, "target": mode()})
    observed["_audit_source"] = "p1d_h1_decision.r1.json"
    row = next(r for r in audit.make_rows({"repo_1": observed}, directory) if r["variant"] == "gold")
    assert row["target_raw_junit_status"] == "missing"


def test_paired_cost_uses_same_rows_excludes_baseline_timeout_and_unpaired():
    audit = load_audit()
    records = {"repo_1": record(
        gold={"gold_applied": True, "tp_applied": True,
              "target": mode(), "selected": mode(2), "full": mode(10)},
        noop={"tp_applied": True, "target": mode(), "selected": mode(20), "full": mode(40)},
        revert={"tp_applied": True, "target": mode(), "selected": mode(9)},
        syntax_break={"perturb_ok": True, "gold_applied": True, "tp_applied": True,
                      "target": mode(), "selected": mode(2, timed_out=True), "full": mode(10)})}
    stats = audit.cost_table(audit.make_rows(records))[0]
    assert stats["n_pairs"] == 2
    assert stats["selected_mean_sec"] == 11
    assert stats["full_mean_sec"] == 25
    assert stats["selected_median_sec"] == 11
    assert stats["full_median_sec"] == 25
    assert stats["n_tasks"] == 1
    assert stats["reduction_ratio_of_sums"] == pytest.approx(0.56)


def test_true_target_cont_rescue_requires_original_unknown():
    audit = load_audit()
    records = {"repo_1": record(
        gold={"gold_applied": True, "tp_applied": True,
              "target": mode(), "target_cont": mode(), "selected": mode(), "full": mode()},
        noop={"tp_applied": True, "target": {}, "target_cont": mode(),
              "selected": mode(), "full": mode()})}
    summary = audit.summarize(audit.make_rows(records), records)
    assert summary["target_cont"]["nonbaseline_attempted"] == 2
    assert summary["target_cont"]["true_rescues"] == 1


def test_artifacts_build_without_changing_manifest(tmp_path):
    audit = load_audit()
    (tmp_path / "results").mkdir()
    (tmp_path / "results/p1d_h1_decision.json").write_text(json.dumps({"repo_1": record()}))
    output = tmp_path / "paper/data/final_audit"
    output.mkdir(parents=True)
    (output / "manifest.json").write_text('{"immutable": true}')
    summary = audit.build_audit(tmp_path, output)
    assert summary["universe"]["tasks"] == 1
    assert (output / "manifest.json").read_text() == '{"immutable": true}'
    for name in ("summary.json", "p1d_rows.csv", "cost_by_repo.csv", "claims.csv"):
        assert (output / name).is_file()
