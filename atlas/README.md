# Atlas SWE

Atlas is a small CPU library for inspecting repository call graphs, selecting
potentially affected tests, comparing retrieval proxies, and generating declarative
agent submission files. It accompanies the Gemma 4 Developer Agent Paper Track
offline study. It does not run a model, repair repositories, or reproduce an
official competition score.

## Installation

Python 3.10 or newer is required. NumPy and PyYAML are runtime dependencies;
PyYAML is required for validation and is no longer an optional regex fallback.
From the project root:

```powershell
python -m pip install ./atlas
python -m pip install "./atlas[dev]"
```

To build a wheel using an environment with setuptools >=68 and wheel installed:

```powershell
python -m pip wheel --no-deps --no-build-isolation --wheel-dir paper/release/wheels ./atlas
```

The wheel contains the four library modules and a PEP 561 `py.typed` marker.
Large graphs, embeddings, repository snapshots, and official runtime wheels are
not packaged as Atlas data. Type hints are supplied, but static type checking
coverage is not claimed.

## Data and graph inspection

Obtain the original graphs and embeddings through the competition's permitted
data access process. This package does not distribute or license those inputs.
The source checkout includes `datasets/swe_atlas_129.jsonl` with task mappings;
reuse remains subject to the original benchmark and repository licenses.

```python
from atlas.graphio import graph_key_for, load_graph, load_embeddings

key = graph_key_for(task)  # task contains repo and base_commit
graph = load_graph(key, root="data")
node_ids, vectors = load_embeddings(key, root="data")
```

The expected paths are `data/graphs/<key>.json` and
`data/embeddings/<key>.npz`. Graph JSON has `nodes` with `id` and `text`, and
`edges` with `source`, `target`, and `type: calls`. Embedding NPZ files map node
names to numeric vectors. Only open trusted NPZ inputs: the current loader uses
NumPy's `allow_pickle=True` for compatibility with the research inputs.
Embeddings are L2-normalized by default. Cosine ties sort by node name, independent
of NPZ insertion order. Graph keys are repository plus commit, not task IDs;
multiple tasks may share one graph.

## Retrieval budgets

```python
from atlas.locate import LocateConfig, run_locate

results = run_locate(graph, "client timeout", cfg=LocateConfig(seed_k=10, max_k=50))
```

B0 sorts a lexical stem Jaccard proxy by score, then node name. B1 reserves all
`min(seed_k, max_k)` available B0 seeds before adding neighbors, visiting seeds in
rank order and each seed's outgoing and incoming neighbors in sorted order. B2
starts from B0's first seed and adds sorted one-hop neighbors. Both stop at
`max_k`; `expanded` counts only retained non-seed nodes. `seed_k=0` makes B1
empty; `max_k=0` makes all three empty. Negative node, seed, and character budgets
raise `ValueError`. `hops` is reserved metadata: these builders currently expand
exactly one hop.

`char_cap=0` allocates zero text characters while preserving node rankings.
Character allocation models snippet text only, without serialization overhead,
tool-call envelopes, or a tokenizer. `tokens_est` is input text length divided by
four, not measured model tokens. These proxies do not establish actual context
delivery or downstream repair benefit. Lexical fallback behavior is inspired by
the harness description; it is not a claim of exact official implementation.

## Local submission precheck

```python
from atlas.emit import emit_agent_yaml, emit_skill, write_submission

report = write_submission(
    "submission",
    emit_agent_yaml(instructions="Read the issue and inspect relevant tests.",
                    skills=["skills/inspect"]),
    skills={"inspect": emit_skill("inspect", "Inspect dependencies before editing.")},
)
assert report["ok"]
```

The generated discriminator is `agent_class: LlmAgent`. Skill references name
directories containing `SKILL.md`. The validator requires one root YAML config
and exactly one allowed model variant across recursive YAML, included mappings,
referenced agents, and standalone YAML. Documented provider prefixes are
normalized. `!include` loads `.md`/`.txt` as raw UTF-8 and `.yaml`/`.yml`
recursively, relative to the containing file. Includes and `config_path`
references reject absolute paths, null bytes, `..`, symlinks/junctions, escapes,
and missing files. Include cycles are rejected and nested YAML `!include` expansion
is limited to depth 10. Agent `config_path` traversal has its separate depth-50
limit and cycle check; an eleven-link agent hierarchy is therefore valid while a
51-link hierarchy is rejected. These depth limits are local structural checks and
do not prove that the official runtime accepts every schema field.

Allowed submission suffixes are `.yaml`, `.yml`, `.md`, `.txt`, `.py`, `.json`,
and `.safetensors`. JSON must parse; adapter config rank is checked against 128.
Safetensors files must be at `adapters/<name>/adapter_model.safetensors`; the
local precheck does not decode or verify their tensor headers. Other weights and
archives, including `.bin`, `.pt`, `.pth`, `.npz`, and `.zip`, are rejected.
Total unpacked size must be strictly below 3 GiB (configurable in tests).

**A successful local check is not official compilation or runtime validation.**
Atlas does not implement the complete ADK schema, all resource limits, skill
script policies, adapter compatibility, or model/GPU compatibility. Validate the
generated files with the official runtime before using competition resources.
The source project's `HARNESS_README.md` §§2–3 documents the reference contract.

## Reproduce library checks

From the full project checkout, after installing development dependencies:

```powershell
python -B -X utf8 -m pytest atlas/tests/test_regressions.py -q -p no:cacheprovider
python -B -X utf8 atlas/tests/test_graph_pipeline.py
python -B -X utf8 atlas/tests/test_emit.py
```

The graph pipeline script requires the separately obtained project data and
research helper scripts. The emit script uses a small configurable size boundary,
so it does not allocate a 3 GiB test file. A real symlink regression is skipped
when Windows cannot create the fixture; injected checks still exercise link
rejection. For full offline evidence reproduction, use the project's release
instructions and audited tables; the library checks do not regenerate study
results. Static call graphs may omit dynamic calls and isolate tests, and suffix
matching may be ambiguous. Unresolved or unobserved outcomes must remain explicit.
