#!/usr/bin/env python3
"""Rebuild inspectable P1D diagnostics without modifying research inputs.

CLI: python -B -X utf8 paper/scripts/final_audit.py --root . --output-dir paper/data/final_audit
All rates are descriptive task-state diagnostics, never official resolved scores.
Only original p1d_h1_decision*.json files are inputs; merged files are excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

VARIANTS = ("baseline", "gold", "noop", "revert", "syntax_break", "comment_only", "wrong_file")
MODES = ("target", "selected", "full", "target_cont")
KNOWN = {"clean", "regress"}
SELECTION_RULE = (
    "Maximum per-record finished timestamp, falling back to started; timestamped records "
    "outrank missing/invalid timestamps. Naive timestamps use UTC+08:00. Ties: more observed "
    "mode results, then JUnit-bearing results, then delta-bearing results; explicit source "
    "priority resume rN > shardN-M > main (higher numeric generation/index wins); final tie "
    "uses canonical record SHA256. No filesystem mtime or enumeration-order overwrite."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _timestamp(record):
    for field in ("finished", "started"):
        try:
            value = datetime.fromisoformat(record.get(field, "").replace("Z", "+00:00"))
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone(timedelta(hours=8)))
            return value.timestamp(), field
        except (ValueError, TypeError, AttributeError):
            continue
    return float("-inf"), "missing"


def _source_priority(name):
    match = re.fullmatch(r"p1d_h1_decision\.r(\d+)\.json", name)
    if match:
        return (2, int(match[1]), 0)
    match = re.fullmatch(r"p1d_h1_decision\.shard(\d+)-(\d+)\.json", name)
    if match:
        return (1, int(match[2]), int(match[1]))
    if name == "p1d_h1_decision.json":
        return (0, 0, 0)
    raise ValueError(f"Unrecognized original source name: {name}")


def _quality(record):
    modes = [m for run in record.get("runs", {}).values() for key in MODES
             if isinstance(m := run.get(key), dict) and m]
    return (len(modes), sum("tests" in m.get("junit", {}) for m in modes),
            sum(m.get("delta_verdict") in KNOWN for m in modes))


def load_records(results_dir: Path) -> tuple[dict, list]:
    """Return chosen records and a per-task provenance ledger (including duplicates)."""
    candidates = defaultdict(list)
    sources = sorted(p for p in Path(results_dir).glob("p1d_h1_decision*.json")
                     if not p.name.endswith("_merged.json"))
    for source in sources:
        priority = _source_priority(source.name)
        data = json.loads(source.read_text(encoding="utf-8"))
        for iid, record in data.items():
            if not isinstance(record, dict) or not record.get("runs"):
                continue
            stamp, stamp_field = _timestamp(record)
            candidates[iid].append({"source": source.name, "record": record,
                "timestamp": stamp, "timestamp_field": stamp_field,
                "quality": _quality(record), "source_priority": priority,
                "sha256": digest(record)})
    chosen, ledger = {}, []
    # Reproduce the former main-first, sorted-shards overwrite only for sensitivity.
    legacy_order = {p.name: i for i, p in enumerate(sorted(sources,
                   key=lambda p: (p.name != "p1d_h1_decision.json", p.name)))}
    for iid in sorted(candidates):
        options = candidates[iid]
        winner = max(options, key=lambda c: (c["timestamp"], c["quality"],
                                             c["source_priority"], c["sha256"]))
        legacy = max(options, key=lambda c: legacy_order[c["source"]])
        chosen[iid] = dict(winner["record"], _audit_source=winner["source"])
        ledger.append({"instance_id": iid, "selected_source": winner["source"],
            "selected_sha256": winner["sha256"], "candidate_count": len(options),
            "timestamp_status": winner["timestamp_field"],
            "legacy_source": legacy["source"],
            "legacy_selection_differs": winner["sha256"] != legacy["sha256"],
            "candidates": [{"source": c["source"], "started": c["record"].get("started"),
                            "finished": c["record"].get("finished"),
                            "timestamp_status": c["timestamp_field"], "quality": list(c["quality"]),
                            "source_priority": list(c["source_priority"]), "sha256": c["sha256"]}
                           for c in sorted(options, key=lambda c: c["source"])]})
    return chosen, ledger


def injection_status(variant, run):
    if variant == "baseline":
        return "not_applicable", "unmodified baseline"
    if run.get("tp_applied") is False or run.get("gold_applied") is False:
        return "failed", "patch application failed"
    if variant in {"syntax_break", "comment_only", "wrong_file"}:
        if run.get("perturb_ok") is False:
            return "failed", "perturb_ok=False"
        if run.get("perturb_ok") is not True:
            return "unknown", "perturbation application not recorded"
    if run.get("tp_applied") is not True:
        return "unknown", "test patch application not confirmed"
    if variant in {"gold", "syntax_break", "comment_only", "wrong_file"} and run.get("gold_applied") is not True:
        return "unknown", "gold application not confirmed"
    if variant == "revert":
        return "not_applicable", "apply then reset; equivalent to noop; intermediate apply status unrecorded"
    if variant == "noop":
        return "not_applicable", "unmodified control with test patch"
    return "success", "application flags confirm intended operation; no content diff retained"


def read_junit(path):
    if path is None or not path.is_file():
        return None, [], "raw JUnit unavailable"
    try:
        root = ET.parse(path).getroot()
        suites = [s for s in root.iter("testsuite") if not s.findall("testsuite")]
        if not suites:
            return None, [], "no testsuite in XML"
        stats = {key: sum(int(s.get(key, 0)) for s in suites)
                 for key in ("tests", "failures", "errors", "skipped")}
        cases = [{"classname": tc.get("classname", ""), "name": tc.get("name", ""),
                  "file": tc.get("file", ""),
                  "status": "fail" if tc.find("failure") is not None or tc.find("error") is not None
                            else "skipped" if tc.find("skipped") is not None else "pass"}
                 for tc in root.iter("testcase")]
        stats["collection_errors"] = sum("collection" in e.get("message", "").lower()
                                         for e in root.iter("error"))
        return stats, cases, "raw JUnit parsed"
    except (ET.ParseError, ValueError, OSError) as exc:
        return None, [], f"raw JUnit invalid: {exc}"


def _matches(required, case, allow_params=False):
    parts = required.split("::")
    filepart = parts[0].replace("\\", "/").removeprefix("./")
    if len(parts) < 2:
        return False
    expected_class = ".".join([filepart.removesuffix(".py").replace("/", "."), *parts[1:-1]])
    actual_class = case["classname"]
    class_ok = actual_class == expected_class or expected_class.endswith("." + actual_class)
    if case["file"]:
        actual_file = case["file"].replace("\\", "/")
        class_ok = class_ok or (filepart == actual_file or filepart.endswith("/" + actual_file)) and (
            len(parts) == 2 or actual_class.endswith(".".join(parts[1:-1])))
    name_ok = case["name"] == parts[-1]
    if allow_params and "[" not in parts[-1]:
        name_ok = name_ok or case["name"].startswith(parts[-1] + "[")
    return class_ok and name_ok


def evaluate_mode(mode, required_tests=None, junit_path=None):
    """Separate local absolute status from complete required-test evidence.

    A coarse parametrized function is insufficient to prove its whole inventory.
    No official required-test list was retained by the historical corpus.
    """
    recorded = mode.get("junit") or {}
    raw, cases, raw_reason = read_junit(junit_path)
    mismatched = raw is not None and any(key in recorded and recorded[key] != raw[key]
                     for key in ("tests", "failures", "errors", "skipped"))
    incomplete = raw is not None and raw["tests"] != len(cases)
    stats = recorded if "tests" in recorded else raw or {}
    counts_valid = all(type(stats.get(k)) is int and stats[k] >= 0
                       for k in ("tests", "failures", "errors", "skipped"))
    counts_valid = counts_valid and sum(stats[k] for k in ("failures", "errors", "skipped")) <= stats["tests"]
    rc = mode.get("exit_code")
    if mode.get("timed_out"):
        absolute, reason = "timeout", "timed_out=True"
    elif rc is not None and rc != 0:
        absolute, reason = "fail", f"nonzero exit code {rc}"
    elif rc is None or "tests" not in stats:
        absolute, reason = "unknown", "exit code or JUnit counts unavailable"
    elif not counts_valid:
        absolute, reason = "unknown", "JUnit counts incomplete, invalid, or inconsistent"
    elif stats.get("tests", 0) <= 0:
        absolute, reason = "unknown", "zero collected tests"
    elif stats.get("failures", 0) or stats.get("errors", 0):
        absolute, reason = "fail", "JUnit failure/error"
    elif stats.get("tests", 0) <= stats.get("skipped", 0):
        absolute, reason = "unknown", "no unskipped tests"
    else:
        absolute, reason = "pass", "exit 0 with positive unskipped tests and zero failure/error"
    required_status, required_reason = "unknown", "complete required-test list unavailable"
    missing = []
    if required_tests and raw is not None and not mismatched and not incomplete:
        param_gap = False
        unsuccessful = False
        for required in required_tests:
            matched = [c for c in cases if _matches(required, c, allow_params=True)]
            if not matched:
                missing.append(required)
            unsuccessful |= any(c["status"] != "pass" for c in matched)
            param_gap |= "[" not in required.split("::")[-1] and any("[" in c["name"] for c in matched)
        if unsuccessful:
            required_status, required_reason = "fail", "required testcase failed or skipped"
        elif missing or param_gap:
            required_reason = "required tests absent or complete parameter inventory unavailable"
        elif absolute == "pass":
            required_status, required_reason = "pass", "every explicitly enumerated required testcase passed"
        else:
            required_reason = "required matches present but run did not pass"
    elif required_tests:
        required_reason = "raw JUnit missing, incomplete, or inconsistent with recorded counts"
    return {"absolute": absolute, "reason": reason, "required_status": required_status,
            "required_reason": required_reason, "required_missing": missing,
            "raw_junit_status": "mismatch" if mismatched else "incomplete" if incomplete else "available" if raw is not None else "missing",
            "raw_junit_reason": raw_reason, "raw_cases": cases,
            "collection_errors": raw.get("collection_errors") if raw is not None else None,
            "junit_tests": stats.get("tests"), "junit_failures": stats.get("failures"),
            "junit_errors": stats.get("errors"), "junit_skipped": stats.get("skipped")}


def _junit_path(results_dir, source, iid, variant, mode):
    if results_dir is None:
        return None
    stem = source.removesuffix(".json").removeprefix("p1d_h1_decision")
    tag = stem.removeprefix(".").removeprefix("shard")
    folder = "p1d_junit" + ("_" + tag if tag else "")
    preferred = results_dir / folder / f"{iid}_{variant}_{mode}.xml"
    if preferred.is_file():
        return preferred
    # First-generation shard runners also wrote to the common directory.
    fallback = results_dir / "p1d_junit" / f"{iid}_{variant}_{mode}.xml"
    return fallback if stem.startswith(".shard") and fallback.is_file() else preferred


def make_rows(records, results_dir=None):
    rows = []
    for iid, task in sorted(records.items()):
        for variant in VARIANTS:
            if variant not in task.get("runs", {}):
                continue
            run = task["runs"][variant]
            status, reason = injection_status(variant, run)
            source = task.get("_audit_source", "p1d_h1_decision.json")
            row = {"instance_id": iid, "repo": task.get("repo", "unknown"), "variant": variant,
                   "source": source, "injection_status": status, "injection_reason": reason,
                   "eligible": variant != "baseline" and status in {"success", "not_applicable"},
                   "control_family": "unmodified" if variant in {"baseline", "noop", "revert"} else variant,
                   "official_resolved": "not_assessed"}
            for key in ("perturb_ok", "gold_applied", "tp_applied"):
                row[key] = run.get(key)
            for key in MODES:
                mode = run.get(key) or {}
                path = _junit_path(results_dir, source, iid, variant, key)
                evidence = evaluate_mode(mode, task.get("required_tests"), path)
                row[key + "_present"] = bool(mode)
                row[key + "_delta"] = mode.get("delta_verdict") if mode.get("delta_verdict") in KNOWN else "unknown"
                row[key + "_exit_code"] = mode.get("exit_code")
                row[key + "_timed_out"] = bool(mode.get("timed_out"))
                row[key + "_sec"] = mode.get("sec")
                row[key + "_n_targets"] = mode.get("n_targets")
                row[key + "_empty_selection"] = mode.get("n_targets") == 0
                row[key + "_collection_errors"] = evidence["collection_errors"]
                row[key + "_junit_path"] = path.relative_to(results_dir.parent).as_posix() if path and results_dir else ""
                for field in ("absolute", "reason", "required_status", "required_reason", "raw_junit_status",
                              "junit_tests", "junit_failures", "junit_errors", "junit_skipped"):
                    row[key + "_" + field] = evidence[field]
                row[key + "_required_missing"] = json.dumps(evidence["required_missing"])
                cases = evidence["raw_cases"]
                nodes = task.get("target_nodes", [])
                matched = [[c for c in cases if _matches(node, c, True)] for node in nodes]
                row[key + "_ast_target_coverage_proxy"] = (
                    "unknown" if not nodes or not cases or evidence["raw_junit_status"] != "available"
                    else "gap" if any(not group for group in matched)
                    else "fail" if any(c["status"] != "pass" for group in matched for c in group)
                    else "observed_matches_pass")
                row[key + "_ast_target_proxy_limitation"] = "AST nodes; parameter inventory and official required set unverified"
            row["true_target_cont_rescue"] = (row["eligible"] and row["target_delta"] == "unknown"
                                                 and row["target_cont_delta"] in KNOWN)
            rows.append(row)
    return rows


def _distribution(values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "sum": sum(values)}


def _population(rows):
    nonbaseline = [r for r in rows if r["variant"] != "baseline"]
    eligible = [r for r in nonbaseline if r["eligible"]]
    paired = [r for r in eligible if r["target_delta"] in KNOWN and r["selected_delta"] in KNOWN]
    both_full = [r for r in eligible if r["full_delta"] in KNOWN and r["selected_delta"] in KNOWN]
    result = {"n": len(nonbaseline), "eligible": len(eligible),
              "injection_failed": sum(r["injection_status"] == "failed" for r in nonbaseline),
              "injection_unknown": sum(r["injection_status"] == "unknown" for r in nonbaseline),
              "injection_status_counts": dict(Counter(r["injection_status"] for r in nonbaseline)),
              "paired_target_selected": len(paired),
              "agree_target_selected": sum(r["target_delta"] == r["selected_delta"] for r in paired),
              "paired_full_selected": len(both_full),
              "agree_full_selected": sum(r["full_delta"] == r["selected_delta"] for r in both_full),
              "target_selected_confusion": dict(Counter(f'target={r["target_delta"]},selected={r["selected_delta"]}' for r in paired)),
              "full_selected_confusion": dict(Counter(f'full={r["full_delta"]},selected={r["selected_delta"]}' for r in both_full))}
    result["modes"] = {m: {"unknown_delta_all": sum(r[m + "_delta"] == "unknown" for r in nonbaseline),
                             "unknown_delta_eligible": sum(r[m + "_delta"] == "unknown" for r in eligible),
                             "absolute_all": dict(Counter(r[m + "_absolute"] for r in nonbaseline)),
                             "absolute_eligible": dict(Counter(r[m + "_absolute"] for r in eligible)),
                             "required_status_all": dict(Counter(r[m + "_required_status"] for r in nonbaseline)),
                             "timeouts_all": sum(r[m + "_timed_out"] for r in nonbaseline),
                             "empty_selection_all": sum(r[m + "_empty_selection"] for r in nonbaseline),
                             "junit_error_rows_all": sum(bool(r[m + "_junit_errors"]) for r in nonbaseline),
                             "collection_error_rows_all": sum(bool(r[m + "_collection_errors"]) for r in nonbaseline),
                             "collection_error_evidence_unknown": sum(r[m + "_collection_errors"] is None for r in nonbaseline),
                             "raw_junit_all": dict(Counter(r[m + "_raw_junit_status"] for r in nonbaseline))}
                       for m in MODES}
    return result


def summarize(rows, records):
    nonbaseline = [r for r in rows if r["variant"] != "baseline"]
    return {"universe": {"tasks": len(records), "observed_states": len(rows),
                         "expected_states": len(records) * len(VARIANTS),
                         "observed_nonbaseline_states": len(nonbaseline),
                         "expected_nonbaseline_states": len(records) * (len(VARIANTS) - 1),
                         "missing_states": [{"instance_id": iid, "variant": v}
                             for iid, record in sorted(records.items()) for v in VARIANTS
                             if v not in record.get("runs", {})]},
            "nonbaseline": _population(rows),
            "per_repo": {repo: _population([r for r in rows if r["repo"] == repo])
                         for repo in sorted({r["repo"] for r in rows})},
            "per_variant": {v: _population([r for r in rows if r["variant"] == v]) for v in VARIANTS[1:]},
            "absolute_controls": {v: {m: dict(Counter(r[m + "_absolute"] for r in rows if r["variant"] == v))
                                       for m in MODES[:3]} for v in ("gold", "noop", "revert")},
            "target_cont": {"nonbaseline_attempted": sum(r["target_cont_present"] for r in nonbaseline),
                            "true_rescues": sum(r["true_target_cont_rescue"] for r in nonbaseline),
                            "rescued_tasks": len({r["instance_id"] for r in nonbaseline if r["true_target_cont_rescue"]}),
                            "definition": "eligible original target delta unknown, target_cont delta known; supplemental only"},
            "setup": {"all": _distribution([r.get("account_A_setup_sec") for r in records.values()]),
                      "per_repo": {repo: _distribution([r.get("account_A_setup_sec") for r in records.values()
                                                       if r.get("repo") == repo])
                                   for repo in sorted({r.get("repo", "unknown") for r in records.values()})}}}


def _cost_exclusion(row):
    if row["variant"] == "baseline":
        return "baseline"
    if not row["eligible"]:
        return "injection_" + row["injection_status"]
    if row["selected_timed_out"] or row["full_timed_out"]:
        return "timeout_censored"
    values = [row[m + "_sec"] for m in ("selected", "full")]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        return "missing_or_invalid_pair"
    if row["selected_delta"] not in KNOWN or row["full_delta"] not in KNOWN:
        return "delta_evidence_unknown"
    if row["selected_empty_selection"] or row["full_empty_selection"]:
        return "empty_selection"
    return "included"


def cost_table(rows):
    """Same-row descriptive timing pairs; collapse within tasks before task summaries."""
    output = []
    for repo in ["ALL", *sorted({r["repo"] for r in rows})]:
        population = [r for r in rows if repo == "ALL" or r["repo"] == repo]
        pairs = [r for r in population if _cost_exclusion(r) == "included"]
        selected = _distribution([r["selected_sec"] for r in pairs])
        full = _distribution([r["full_sec"] for r in pairs])
        by_task = defaultdict(list)
        for row in pairs:
            by_task[row["instance_id"]].append(row)
        task_s = [statistics.mean(r["selected_sec"] for r in group) for group in by_task.values()]
        task_f = [statistics.mean(r["full_sec"] for r in group) for group in by_task.values()]
        output.append({"repo": repo, "n_pairs": len(pairs), "n_tasks": len(by_task),
            "selected_mean_sec": selected["mean"], "full_mean_sec": full["mean"],
            "selected_median_sec": selected["median"], "full_median_sec": full["median"],
            "selected_sum_sec": selected["sum"], "full_sum_sec": full["sum"],
            "paired_difference_mean_sec": statistics.mean(r["full_sec"] - r["selected_sec"] for r in pairs) if pairs else None,
            "reduction_ratio_of_sums": 1 - selected["sum"] / full["sum"] if full["sum"] else None,
            "task_selected_mean_sec": statistics.mean(task_s) if task_s else None,
            "task_full_mean_sec": statistics.mean(task_f) if task_f else None,
            "task_selected_median_sec": statistics.median(task_s) if task_s else None,
            "task_full_median_sec": statistics.median(task_f) if task_f else None,
            "task_reduction_ratio_of_sums": 1 - sum(task_s) / sum(task_f) if sum(task_f) else None,
            "exclusions": json.dumps(dict(Counter(_cost_exclusion(r) for r in population if _cost_exclusion(r) != "included")), sort_keys=True),
            "population": "nonbaseline successful operations and controls; known selected/full deltas; no timeouts; same paired rows",
            "limitation": "task-state timings repeated within task; task_* first average variants per task; no inferential independence; noop/revert equivalent; setup excluded; no model/agent loop"})
    return output


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _legacy_summary(results_dir):
    records = {}
    paths = sorted(results_dir.glob("p1d_h1_decision*.json"),
                   key=lambda p: (p.name != "p1d_h1_decision.json", p.name))
    for path in paths:
        if not path.name.endswith("_merged.json"):
            records.update(json.loads(path.read_text(encoding="utf-8")))
    rows = make_rows(records)
    target_known = [r for r in rows if r["variant"] != "baseline" and r["target_delta"] in KNOWN]
    eligible = [r for r in target_known if r["eligible"]]
    paired = [r for r in eligible if r["selected_delta"] in KNOWN]
    return {"selection": "main then lexicographic source overwrite (historical only)",
            "target_known": len(target_known),
            "agree_selected_on_target_known": sum(r["selected_delta"] == r["target_delta"] for r in target_known),
            "failed_injection_in_target_known": sum(r["injection_status"] == "failed" for r in target_known),
            "eligible_target_known": len(eligible), "eligible_paired": len(paired),
            "eligible_paired_agree": sum(r["selected_delta"] == r["target_delta"] for r in paired),
            "limitation": "sensitivity reference; excludes unknown from paired agreement; not official repair score"}


def build_audit(root: Path, output_dir: Path) -> dict:
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    results_dir = root / "results"
    records, ledger = load_records(results_dir)
    rows = make_rows(records, results_dir)
    for row in rows:
        row["cost_status"] = _cost_exclusion(row)
    summary = summarize(rows, records)
    summary["schema_version"] = 1
    summary["definitions"] = {
        "selection": SELECTION_RULE,
        "delta": "historical baseline-relative bad-set difference; clean does not mean repaired",
        "absolute": "local rc/JUnit positive-unskipped-test check; independent from delta and required coverage",
        "required": "all enumerated required tests must pass; unavailable list, XML gap, or unenumerated parametrization is unknown",
        "independence": "129 task identities, repeated variants; no independent-task claim from state count; noop and revert share control family",
        "official": "official resolved not assessed: corpus lacks official required-test inventory and official runner receipts"}
    summary["provenance"] = {"n_sources": len({c["source"] for e in ledger for c in e["candidates"]}),
        "n_records": sum(e["candidate_count"] for e in ledger),
        "duplicate_tasks": [e for e in ledger if e["candidate_count"] > 1],
        "legacy_changed_tasks": [e["instance_id"] for e in ledger if e["legacy_selection_differs"]]}
    summary["legacy_sensitivity"] = _legacy_summary(results_dir)
    tasks_path = root / "data/tasks.jsonl"
    if tasks_path.is_file():
        tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary["universe"]["task_manifest_records"] = len(tasks)
        summary["universe"]["unique_graph_keys"] = len({(t["repo"].split("/")[-1], t["base_commit"]) for t in tasks})
        summary["universe"]["missing_result_tasks"] = sorted(t["instance_id"] for t in tasks if t["instance_id"] not in records)
    h2_path = root / "paper/data/h2_localization.json"
    if h2_path.is_file():
        h2 = json.loads(h2_path.read_text(encoding="utf-8"))
        summary["h2_historical_proxy"] = {"source": "paper/data/h2_localization.json", "sha256": hashlib.sha256(h2_path.read_bytes()).hexdigest(),
            "n_tasks_evaluated": h2.get("n_tasks_evaluated"), "skipped": h2.get("skipped"), "summary": h2.get("summary"),
            "limitation": "historical issue/file-stem Jaccard proxy and gold-informed oracle; old ties may vary; not measured official retrieval or delivered context; pending fresh deterministic run"}
    costs = cost_table(rows)
    summary["paired_cost"] = costs
    input_paths = {results_dir / c["source"] for entry in ledger for c in entry["candidates"]}
    input_paths.update(root / r[m + "_junit_path"] for r in rows for m in MODES if r[m + "_junit_path"])
    input_paths.update(p for p in (tasks_path, h2_path) if p.is_file())
    input_evidence = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in sorted(input_paths) if path.is_file()]
    summary["input_evidence"] = {"path": "input_evidence.json", "n_files": len(input_evidence),
                                 "limitation": "Hashes observed during audit generation; original before-edit freeze remains manifest.json"}
    claims = []
    def claim(name, value, numerator=None, denominator=None, population="tasks", source="p1d_rows.csv", limitation="Descriptive local reconstruction; not official resolved"):
        claims.append(dict(claim=name, value=value, numerator=numerator, denominator=denominator,
                           population=population, source=source, limitation=limitation))
    claim("task_universe", len(records), len(records), len(records))
    claim("source_files", summary["provenance"]["n_sources"], source="provenance.json")
    claim("source_records", summary["provenance"]["n_records"], source="provenance.json")
    if "unique_graph_keys" in summary["universe"]:
        claim("unique_graph_keys", summary["universe"]["unique_graph_keys"], population="graph identities", source="data/tasks.jsonl")
    claim("nonbaseline_states", len(rows) - len(records), len(rows) - len(records), len(records), "six repeated states per task")
    nb = summary["nonbaseline"]
    for key in ("injection_failed", "injection_unknown", "eligible", "paired_target_selected"):
        claim(key, nb[key], nb[key], nb["n"], "nonbaseline task-states")
    den = nb["paired_target_selected"]
    claim("target_selected_delta_agreement", nb["agree_target_selected"] / den if den else None,
          nb["agree_target_selected"], den, "eligible task-states with both target and selected delta known",
          limitation="Baseline-relative diagnostic; states are clustered by task; clean includes unresolved base failures")
    for repo, population in summary["per_repo"].items():
        n, d = population["agree_target_selected"], population["paired_target_selected"]
        claim(f"target_selected_delta_agreement_{repo}", n / d if d else None, n, d,
              "eligible within-repo states with known target/selected delta")
    for m in MODES[:3]:
        mode_stats = nb["modes"][m]
        for name, n in (("unknown_delta", mode_stats["unknown_delta_all"]),
                        ("required_unknown", mode_stats["required_status_all"].get("unknown", 0)),
                        ("timeout", mode_stats["timeouts_all"]),
                        ("empty_selection", mode_stats["empty_selection_all"]),
                        ("raw_junit_missing", mode_stats["raw_junit_all"].get("missing", 0))):
            claim(f"{m}_{name}", n, n, nb["n"], "all nonbaseline task-states")
    claim("full_selected_delta_agreement", nb["agree_full_selected"] / nb["paired_full_selected"] if nb["paired_full_selected"] else None,
          nb["agree_full_selected"], nb["paired_full_selected"], "eligible paired full/selected task-states")
    for variant in ("gold", "noop", "revert"):
        for m in MODES[:3]:
            population = [r for r in rows if r["variant"] == variant]
            n = sum(r[m + "_absolute"] == "pass" for r in population)
            claim(f"{variant}_{m}_local_absolute_pass", n / len(population) if population else None,
                  n, len(population), f"all observed {variant} states", limitation="Local rc/JUnit only; required inventory unverified; revert equivalent to noop")
    claim("target_cont_true_rescues", summary["target_cont"]["true_rescues"], summary["target_cont"]["true_rescues"],
          summary["target_cont"]["nonbaseline_attempted"], "nonbaseline target_cont attempts", limitation="Known supplemental delta when original target delta unknown; not substitute for original target")
    for cost in costs:
        claim(f'paired_wallclock_reduction_{cost["repo"]}', cost["reduction_ratio_of_sums"],
              cost["full_sum_sec"] - cost["selected_sum_sec"], cost["full_sum_sec"],
              f'{cost["n_pairs"]} pairs across {cost["n_tasks"]} tasks', "cost_by_repo.csv", cost["limitation"])
        for metric in ("selected_mean_sec", "full_mean_sec", "selected_median_sec", "full_median_sec",
                       "task_selected_mean_sec", "task_full_mean_sec", "task_selected_median_sec", "task_full_median_sec",
                       "task_reduction_ratio_of_sums"):
            claim(f'{metric}_{cost["repo"]}', cost[metric], denominator=cost["n_tasks"] if metric.startswith("task_") else cost["n_pairs"],
                  population="task averages" if metric.startswith("task_") else "paired task-states", source="cost_by_repo.csv", limitation=cost["limitation"])
    claim("setup_sec_mean", summary["setup"]["all"]["mean"], summary["setup"]["all"]["sum"],
          summary["setup"]["all"]["n"], "one chosen setup per task", limitation="Historical dependency setup measurement; not complete container/model/queue cost")
    if "h2_historical_proxy" in summary:
        h2 = summary["h2_historical_proxy"]
        claim("historical_h2_evaluable_tasks", h2["n_tasks_evaluated"], population="tasks", source=h2["source"], limitation=h2["limitation"])
        for arm, metrics in (h2.get("summary") or {}).items():
            claim(f"historical_h2_{arm}_recall_at_50", metrics.get("recall@50"), None,
                  h2["n_tasks_evaluated"], "H2 evaluable tasks; file-level gold", h2["source"], h2["limitation"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "p1d_rows.csv", rows)
    _write_csv(output_dir / "cost_by_repo.csv", costs)
    _write_csv(output_dir / "claims.csv", claims)
    (output_dir / "provenance.json").write_text(json.dumps({"rule": SELECTION_RULE, "tasks": ledger}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "input_evidence.json").write_text(json.dumps({"schema_version": 1, "files": input_evidence}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_audit(args.root, args.output_dir)
    print(json.dumps({"universe": summary["universe"], "provenance": summary["provenance"],
                      "nonbaseline": summary["nonbaseline"], "target_cont": summary["target_cont"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
