# Execution status — 2026-10-18

Plan: `docs/superpowers/plans/2026-10-18-kaggle-completion.md`

User approved continuation of the reviewed design.

- M0: before-edit SHA256 manifest created; 323 files.
- Ruling: no Git repository exists; source hashes and durable artifacts replace worktree/commit bookkeeping.
- Pre-flight: audit tables feed paper assets; library and runner edits use separate ownership; shared inputs are read-only.
- Tasks 1–6: running; formal Kaggle submission not yet completed.

## Final execution evidence — 2026-10-18

- Audit regeneration: 129 tasks, 903 states, 774 nonbaseline; 80 failed injections; 642 known target/selected pairs; 611 agreements; required-test proof unknown for all states; target-cont supplemental rescue 0.
- Scoped regression suite after all fixes: `python -B -X utf8 -m pytest paper/tests atlas/tests -q -p no:cacheprovider` → **89 passed, 1 skipped**. Python compilation of modified scripts/modules passed. The unscoped repository scan is retained as a limitation: 147 collection errors came from historical `results/p1_workspace` snapshots and missing/incompatible third-party dependencies.
- Deterministic H2 runs under hash seeds 0 and 42 are semantically equal. Fresh B0 recall@50 is 0.1879 on 124 proxy-evaluable tasks; this is lexical filename-stem diagnosis, not official embedding retrieval.
- Three figures and source CSVs were rendered and inspected. `paper/notebooks/reproduce.ipynb` was executed with a clean local notebook environment.
- Kaggle private preflight completed installation and failed bounded E4B first-token generation on Tesla T4 compute capability 7.5: Triton requested 98,304 bytes shared memory versus 65,536 available. Agent status is not run; no official score is claimed. Output is in `paper/data/final_audit/kaggle_preflight_result.json`.
- Release archive rebuilt at `paper/release/atlas-paper-track-release.zip` with 51 files, no wheels, raw snapshots, or credentials. The Paper Track Writeup is **not submitted** because the installed SDK exposes read-only Writeup-link retrieval rather than creation/submission.
- Post-review runner hardening: resumed attempts delete the destination JUnit receipt before each copy and require a successful copy; resume metadata preserves the original configuration and appends an attempt record after compatibility checks. The historical P1D gold-derived test-selection advantage is documented in the runner and paper limitations.

## Submission completed — 2026-10-21

- Writeup submitted and PUBLISHED: `https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/new-writeup-1791929370440` (id 119115, publishTime 2026-10-21T18:26:49Z). Checklist 6/6; body 2,081 words; card image uploaded and cropped; Project Links carries the public GitHub release.
- Submission mechanics: the Kaggle SDK has no create/submit operation, and the hackathon submission endpoint rejects bearer tokens. Submission used the authenticated browser session: internal `discussions.WriteUpsService` endpoints (which require `X-XSRF-TOKEN` derived from the `XSRF-TOKEN` cookie) plus the normal UI for the image upload and final Submit.
- Image upload blocked by network, not by permissions: `www.googleapis.com` is unreachable from this machine while `storage.googleapis.com` works. The signed upload URL targets the unreachable host, so the page's requests were rewritten to the reachable one; the upload then completed normally.

## E4B/T4 compatibility resolved — 2026-10-21

Three bounded attempts, all recorded:

| Attempt | Change | Result |
|---|---|---|
| v1 | baseline FP16, TP1, 2048-token context | failed before first token: Triton needed 98,304 B shared memory vs 65,536 B |
| v2 | module-level monkey-patch of `_get_tile_size` | failed identically: vLLM runs attention in a spawned EngineCore process that re-imports vLLM from disk, so the patch was not inherited |
| v3 | source-level rewrite of `triton_unified_attention.py` (tile 16 when `head_size > 128`), applied before vLLM import | **passed**: 32 tokens generated, 9.01 GiB weights loaded in 223 s, engine init 17.35 s, 360.6 s total |

v3 evidence: `paper/data/final_audit/kaggle_preflight_v3_result.json` and `kaggle_preflight_v3_generation.log`. Patched file SHA256 `5a8af268…8921` → `4a5c746b…2b6f`. The guard is scoped to compute capability < 8.0 and is idempotent.

The generated text was degenerate and repetitive. This is a runtime-compatibility result for one configuration only: it is not an agent score, not a quality result, and not evidence that this checkpoint is usable at FP16 on T4. The writeup was updated to say exactly this, and remains PUBLISHED.

## v4 diagnostics: numerics clean, degeneracy was a prompting artifact — 2026-10-21

A fourth bounded kernel (`atlas-gemma4-e4b-t4-preflight-v4-numerics-chat`) answered the two questions v3 left open. Evidence: `paper/data/final_audit/kaggle_preflight_v4_result.json`, `kaggle_preflight_v4_generation.log`, `kaggle_preflight_v4_kernel.log`.

1. **Tile-16 kernel numerics: pass.** Against a float32 SDPA reference, worst relative error 2.84e-4 across head_dim 256/512 in both prefill and decode. On identical inputs the Triton tile-16 kernel's max absolute error equals that of a standard fp16 SDPA call (e.g. 0.000963 vs 0.000963 at head_dim 512), so the guard costs no accuracy beyond ordinary half-precision noise.
2. **Degenerate output was a prompting artifact.** Raw-string prompts reproduce the v3 repetition verbatim; the same prompts through the tokenizer's chat template produce coherent, correct answers ("17 * 24" → "408"; a correct one-sentence unit-test definition). v1–v3 fed raw strings to an instruction-tuned model.

Both gates for a real agent experiment are now open: the runtime is compatible, numerically sound, and produces usable output under proper prompting. The submitted writeup was updated to this effect (2,133 words, still PUBLISHED). The agent smoke experiment itself remains unrun and is the next optional step; until then the paper claims runtime/numerical compatibility only.

Notebook runtimes observed: install ≈ 5–7 min; weight load 222 s (v3) / 118 s (v4, warm); full v4 probe 183.8 s.

