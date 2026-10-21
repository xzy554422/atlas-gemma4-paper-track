# Kaggle submission handoff

Target: Gemma 4 Developer Agent — Paper Track. Deadline checked 2026-10-18: 2026-11-12 23:59 UTC (2026-11-13 07:59 Singapore).

## Prepared locally

- English writeup: `paper/draft/writeup.md` (2,062 words by the release counter; under the 3,000-word limit).
- References: `paper/draft/references.bib`.
- Three PNG/SVG figures with source CSVs and an input/output manifest.
- Executed CPU reproduction notebook and a private Kaggle E4B/T4 preflight notebook.
- Archive: `atlas-paper-track-release.zip`, with `artifact-manifest.json`.

## Verified account and cloud state

Kaggle API account `yzx22442255` is authenticated. Competition membership was verified for the main competition and Paper Track. The reported weekly GPU quota was 30 hours total, 0 hours used, refresh 2026-10-24. A private preflight kernel was pushed at:

https://www.kaggle.com/code/yzx22442255/atlas-gemma4-e4b-t4-preflight

The kernel completed. The bounded probe observed one Tesla T4 (compute capability 7.5) and failed before generation because Triton unified attention requested 98,304 bytes of shared memory while the hardware limit is 65,536 bytes. Installation passed; the agent was not run. This is recorded in `paper/data/final_audit/kaggle_preflight_result.json`. The result is a compatibility failure for this configuration, not a resolved score.

## Required final UI action

The Kaggle SDK available in this environment can push private notebooks and read resolved Writeup links, but it did not expose a create/submit Writeup operation. Therefore this record does not claim that the Paper Track Writeup is submitted. The remaining action is to open the Paper Track page, paste the writeup and references/figure links as required by the current UI, submit it, and record the resulting public Writeup URL. Do not present the private preflight URL as a Writeup URL.

## Scope statement for the submission

The study measures offline diagnostic agreement, command-time observations, deterministic lexical localization proxies, and a reproducible resource. It does not claim an official resolved score, repair success rate, end-to-end agent speedup, or a faithful reproduction of the competition's embedding search.
