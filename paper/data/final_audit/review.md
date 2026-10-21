# Post-fix independent implementation review — 2026-10-18

I re-read the current runner/library sources, final writeup, figures, release manifest and archive. I ran only the scoped regression suite:

`python -B -X utf8 -m pytest paper/tests/test_final_audit.py paper/tests/test_runner_regressions.py atlas/tests/test_regressions.py -q -p no:cacheprovider`

Result: **89 passed, 1 skipped** (the skip is the platform-dependent symlink test). No Docker, model, snapshot-wide test, or third-party snapshot execution was run.

## Previous findings

| Finding | Status | Current evidence |
|---|---|---|
| R1 stale JUnit can be borrowed on resume | **Resolved** | Both runners now call `prepare_receipt()` before copying, check `docker cp` status and destination existence, and synthesize incomplete/missing evidence when the copy fails. The latest scoped suite is 89 passed, 1 skipped. No Docker run was repeated here. |
| R2 insufficient JUnit can become pass/clean | **Resolved in current predicates** | `run_support.read_junit_evidence()` checks declared counts against testcase nodes; `junit_complete()` requires complete XML and at least one unskipped test. `p1d_h1_container.verdict()` and `apply_baseline_delta()` now return `unknown` for incomplete/unexecuted evidence; all-skipped is not pass. The scoped tests passed. |
| R3 timing node matching ignores file/parameters | **Resolved for the inspected matcher** | `p1_timing.node_passed()` now matches normalized file/class/function, exact parameters when requested, and returns unknown for coarse parameterized inventories. `timing_complete()` also requires complete JUnit evidence. The scoped tests passed. A real JUnit receipt from Docker was not rerun here. |
| R4 resume provenance/configuration can be relabeled | **Resolved for checked fields** | Resume now preserves the prior sidecar, appends `resume_attempts`, and rejects mismatched task-source hash, script hash, image, Docker limits, variants and modes. This preserves original provenance and prevents the prior relabeling bug. The check does not include every operational knob (for example target-cont policy, keep-unknown and receipt directory), so those remain a narrower follow-up risk. |
| R5 include depth and agent-reference depth conflated | **Resolved** | `atlas/atlas/emit.py:283-289` now carries separate `include_stack` (limit 10) and `config_stack` (limit 50), with shared cycle detection. Scoped regression tests passed. |
| R6 target_cont reruns ordinary successful XML | **Resolved** | `p1d_h1_container.needs_target_cont()` now uses `junit_complete(stats)` instead of an empty `bad` list, so valid clean XML does not trigger the fallback. Scoped regression tests passed. |

## Remaining implementation issue

### R7 — P2: P1D test selection is still gold-derived for every variant

`p1d_h1_container.py:617-626` computes `gold_files` and `select_tests(...)` once before the variant loop, then reuses `sel` for noop, revert, syntax, comment and wrong-file states at `:675-678`. This is a fixed gold-location diagnostic; it is not candidate-only selection recomputed from each variant's actual changed files, despite the runner's candidate-only wording. The writeup currently discloses that gold/test-patch information is used for offline interventions, but the method should explicitly say the selected set is frozen from gold locations for this historical experiment. Do not present these states as an end-to-end issue-only selector comparison. Minimal repair is either to correct the method wording and limitation, or run a separately labelled experiment that recomputes selection from each candidate diff.

## Final paper, figures, Kaggle and release checks

- The writeup explicitly labels the T4/E4B result as a bounded compatibility failure and says **“the agent was not run”** and **“not an agent score.”** It does not convert the failure into repair, agent, or official-resolved evidence.
- The paper's central denominators match the audited summary: 129 tasks, 774 nonbaseline states, 80 failed operations, 642 known paired states and 611 agreements. The supplemental target-cont run reports 170 attempts and zero rescues; it is described as supplemental rather than acceptance evidence.
- The three figure files and source CSVs are present. `verification.json` reports repeat generation identical and no changed frozen research files.
- The release archive is readable. All 50 manifest file bytes and hashes match their archive-relative entries; no `.whl`, wheelhouse, snapshot, `.tgz`, or `.npz` entries are present. This directly verifies the intended exclusion of model/third-party bulk inputs.
- The release manifest includes wrapper hashes using archive-relative names and explicitly excludes only its own self-hash. A literal archive-to-manifest check reports 50 manifest entries present with matching hashes in the 51-entry ZIP (the extra entry is the manifest itself).
- `SUBMISSION.md` correctly says the Writeup is **not submitted** because the SDK lacks a create/submit operation, and it distinguishes the private preflight URL from a public Writeup URL. No formal submission should be claimed until the UI action is completed.

The remaining implementation risk is R7. R4 has a narrower unchecked-knob caveat described above. The scoped tests passing does not establish a new Docker receipt run or official Kaggle submission.




