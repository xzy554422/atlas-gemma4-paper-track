"""Small, dependency-free safeguards shared by research CLI runners."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET
from uuid import uuid4


def fresh_output(root, stem):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(root) / "results" / f"final_{stem}_{stamp}_{uuid4().hex[:6]}.json"


def protect_historical_path(path, root):
    """Historical result names remain read-only, even with explicit resume."""
    path = Path(path).resolve()
    root = Path(root).resolve()
    frozen_manifest = root / "paper/data/final_audit/manifest.json"
    if frozen_manifest.is_file() and path.is_relative_to(root):
        frozen = json.loads(frozen_manifest.read_text(encoding="utf-8"))
        relative = path.relative_to(root).as_posix()
        if any(entry.get("path") == relative for entry in frozen.get("files", [])):
            raise ValueError(f"Frozen input is read-only: {path}; choose a fresh final_* path")
    result_root = (Path(root) / "results").resolve()
    if path.is_relative_to(result_root):
        relative = path.relative_to(result_root)
        first = relative.parts[0]
        if first.startswith(("p1_", "p1d_", "h1_", "h1b_", "h2_localization", "swe_atlas_129_build")):
            raise ValueError(f"Historical result path is read-only: {path}; choose a fresh final_* path")
    return path


def prepare_output(path, root, resume=False):
    path = protect_historical_path(path, root)
    if path.exists() and not resume:
        raise FileExistsError(f"Output already exists: {path}; use --resume for a new-run output")
    if resume and not path.is_file():
        raise ValueError(f"--resume requires an existing output file: {path}")
    if resume:
        with path.open(encoding="utf-8") as fh:
            results = json.load(fh)
        if not isinstance(results, dict):
            raise ValueError("Resume output must be a JSON object keyed by instance_id")
    else:
        results = {}
    return path, results


def prepare_junit_dir(path, root, resume=False):
    path = protect_historical_path(path, root)
    if path.exists() and not resume:
        raise FileExistsError(f"JUnit directory already exists: {path}; choose a fresh directory")
    path.mkdir(parents=True, exist_ok=resume)
    return path


def prepare_receipt(path):
    """Remove one prior receipt before a fresh docker copy.

    A failed `docker cp` must leave no file that a resumed attempt could parse.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    return path


def select_ids(tasks, explicit=(), all_tasks=False, legacy_all=False, manifest=None,
               legacy_ids=()):
    sources = bool(explicit) + bool(all_tasks) + bool(legacy_all) + bool(manifest)
    if sources != 1:
        raise ValueError("Choose exactly one of task IDs, --all-tasks, --all, or --task-manifest")
    if manifest:
        text = Path(manifest).read_text(encoding="utf-8-sig")
        if text.lstrip().startswith("["):
            values = json.loads(text)
            iids = [v["instance_id"] if isinstance(v, dict) else v for v in values]
        else:
            iids = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        iids = list(tasks) if all_tasks else list(legacy_ids if legacy_all else explicit)
    iids = list(dict.fromkeys(str(iid).strip() for iid in iids))
    if not iids:
        raise ValueError("Task selection is empty")
    missing = [iid for iid in iids if iid not in tasks]
    if missing:
        raise ValueError(f"Unknown task IDs: {', '.join(missing)}")
    return iids


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_metadata(task_path, script_path, task_ids=None, **configuration):
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "task_source_sha256": sha256_file(task_path),
        "script_sha256": sha256_file(script_path),
        "task_ids": task_ids, **configuration,
    }


def write_json(path, value):
    """Replace only the caller's validated new-run output, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_junit_evidence(path):
    """Read all suites and verify declared counts against actual testcase nodes."""
    try:
        root = ET.parse(path).getroot()
        suites = [suite for suite in root.iter("testsuite")
                  if not any(child.tag == "testsuite" for child in suite)]
        if not suites:
            return {"error": "no testsuite", "evidence_complete": False}
        keys = ("tests", "failures", "errors", "skipped")
        stats = {key: sum(int(suite.attrib[key]) for suite in suites) for key in keys}
        if any(int(suite.attrib[key]) < 0 for suite in suites for key in keys):
            raise ValueError("negative JUnit count")
        cases = []
        for testcase in root.iter("testcase"):
            status = next((tag for tag in ("error", "failure", "skipped")
                           if testcase.find(tag) is not None), "passed")
            cases.append({"classname": testcase.get("classname", ""),
                          "name": testcase.get("name", ""),
                          "file": testcase.get("file", ""), "status": status})
        observed = {"tests": len(cases),
                    "failures": sum(case["status"] == "failure" for case in cases),
                    "errors": sum(case["status"] == "error" for case in cases),
                    "skipped": sum(case["status"] == "skipped" for case in cases)}
        stats["evidence_complete"] = all(stats[key] == observed[key] for key in keys)
        stats["cases"] = cases
        stats["bad"] = sorted(f"{case['classname']}::{case['name']}" for case in cases
                              if case["status"] in ("failure", "error"))
        if not stats["evidence_complete"]:
            stats["error"] = f"incomplete JUnit: declared counts do not match testcases ({observed})"
        return stats
    except (ET.ParseError, OSError, KeyError, ValueError) as exc:
        return {"error": f"unreadable JUnit: {exc}", "evidence_complete": False}


def junit_complete(stats, require_executed=True):
    """Completeness marker comes from checking XML, never from exit code alone."""
    keys = ("tests", "failures", "errors", "skipped")
    return bool(stats.get("evidence_complete") is True and not stats.get("error")
                and all(isinstance(stats.get(key), int) and stats[key] >= 0 for key in keys)
                and stats["tests"] > 0
                and (not require_executed or stats["tests"] > stats["skipped"]))
