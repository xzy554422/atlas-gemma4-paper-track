# Kaggle submission handoff

Target: Gemma 4 Developer Agent — Paper Track. Deadline checked 2026-10-18: 2026-11-12 23:59 UTC (2026-11-13 07:59 Singapore).

## SUBMITTED — 2026-10-21

| Field | Value |
|---|---|
| Status | **Submitted** (server `contentState=PUBLISHED`, `publishTime=2026-10-21T18:26:49.557Z`) |
| Public URL | https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/new-writeup-1791929370440 |
| Writeup ID | 119115 (topic 749056) |
| Title | ATLAS: Auditing Static Graph Diagnostics for Repository Repair |
| Subtitle | An offline audit of test-selection and localization diagnostics |
| Word count | 1,989 words (limit 3,000) |
| Submission checklist | 6 of 6 complete |
| Project Links | https://github.com/xzy554422/atlas-gemma4-paper-track |
| Card image | Uploaded and cropped (cover 1120x560 at top=280, thumbnail 1120x1120) |

Verified after submission: the writeup page renders the title, subtitle, full body and images, and shows a
"Submitted" badge with no Submit button. Kaggle returns HTTP 404 to plain `curl` for all writeup URLs
(anti-bot), so accessibility was verified in the browser instead, not with curl.

Editing is still possible before the deadline: retract, edit, and resubmit. The slug remains
`new-writeup-1791929370440` unless renamed.

## Prepared locally

- English writeup: `paper/draft/writeup.md` (source) and `paper/draft/writeup-kaggle.md` (submitted text, absolute image URLs).
- References: `paper/draft/references.bib`; six inline citations in the body.
- Three PNG/SVG figures with source CSVs and an input/output manifest.
- Executed CPU reproduction notebook and a private Kaggle E4B/T4 preflight notebook, plus `model_preflight_v2.ipynb` with the SM 7.5 tile patch for a later retry.
- Archive: `atlas-paper-track-release.zip`, with `artifact-manifest.json`; published at the GitHub URL above (51 files, 50 manifest hashes verified).

## Cloud state

Kaggle API account `yzx22442255` is authenticated. Competition membership was verified for the main competition
and Paper Track. The reported weekly GPU quota was 30 hours total, 0 hours used, refresh 2026-10-24. A private
preflight kernel was pushed at:

https://www.kaggle.com/code/yzx22442255/atlas-gemma4-e4b-t4-preflight

The kernel completed. The bounded probe observed one Tesla T4 (compute capability 7.5) and failed before
generation because Triton unified attention requested 98,304 bytes of shared memory while the hardware limit is
65,536 bytes. Installation passed; the agent was not run. This is recorded in
`paper/data/final_audit/kaggle_preflight_result.json`. The result is a compatibility failure for this
configuration, not a resolved score.

## Scope statement for the submission

The study measures offline diagnostic agreement, command-time observations, deterministic lexical localization
proxies, and a reproducible resource. It does not claim an official resolved score, repair success rate,
end-to-end agent speedup, or a faithful reproduction of the competition's embedding search.
