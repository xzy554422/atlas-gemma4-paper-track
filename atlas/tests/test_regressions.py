"""Boundary regressions for local submission checks and deterministic retrieval."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas.emit import DEFAULT_MODEL, SubmissionLimits, ValidationError, emit_agent_yaml, validate
from atlas.graphio import RepositoryGraph, cosine_topk
from atlas.locate import LocateConfig, build_b0, build_b1, build_b2, pack_to_char_budget


def put(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def agent(root, extra=""):
    return put(root, "agent.yaml", f"name: root\nmodel: {DEFAULT_MODEL}\n{extra}")


def test_relative_includes_parse_yaml_and_raw_prompts(tmp_path):
    agent(tmp_path, "instruction: !include prompts/system.md\ngenerate_content_config: !include configs/sampling.yaml\n")
    put(tmp_path, "prompts/system.md", "# Prompt\nUse {problem_description}: safely.")
    put(tmp_path, "configs/sampling.yaml", "temperature: 0\nstop_sequences: !include stops.yaml\n")
    put(tmp_path, "configs/stops.yaml", "- STOP\n")
    assert validate(str(tmp_path))["models"] == [DEFAULT_MODEL]


@pytest.mark.parametrize("path", ["../outside.md", "/absolute.md", "C:\\outside.md", "\\\\server\\share", "x/../../outside.md"])
def test_include_rejects_unsafe_paths(tmp_path, path):
    agent(tmp_path, f"instruction: !include '{path}'\n")
    with pytest.raises(ValidationError):
        validate(str(tmp_path))


def test_include_cycle_is_a_validation_error(tmp_path):
    agent(tmp_path, "generate_content_config: !include one.yaml\n")
    put(tmp_path, "one.yaml", "again: !include agent.yaml\n")
    with pytest.raises(ValidationError, match="cycle|循环"):
        validate(str(tmp_path))


def test_include_depth_is_bounded(tmp_path):
    agent(tmp_path, "generate_content_config: !include c0.yaml\n")
    for i in range(11):
        put(tmp_path, f"c{i}.yaml", f"nested: !include c{i+1}.yaml\n")
    put(tmp_path, "c11.yaml", "temperature: 0\n")
    with pytest.raises(ValidationError, match="depth|深度"):
        validate(str(tmp_path))


def _write_agent_chain(root, links):
    """Create exactly ``links`` config_path edges from root to a leaf."""
    if links < 1:
        raise ValueError(links)
    agent(root, "sub_agents:\n- config_path: c0.yaml\n")
    for i in range(links):
        if i == links - 1:
            put(root, f"c{i}.yaml", f"name: c{i}\nmodel: {DEFAULT_MODEL}\n")
        else:
            put(root, f"c{i}.yaml", f"name: c{i}\nsub_agents:\n- config_path: c{i+1}.yaml\n")


def test_agent_config_path_depth_11_is_allowed(tmp_path):
    _write_agent_chain(tmp_path, 11)
    assert validate(str(tmp_path))["ok"]


def test_agent_config_path_depth_51_is_rejected(tmp_path):
    _write_agent_chain(tmp_path, 51)
    with pytest.raises(ValidationError, match="config_path.*depth|配置.*深度"):
        validate(str(tmp_path))


def test_included_yaml_agent_references_resolve_from_its_own_directory(tmp_path):
    agent(tmp_path, "tools: !include configs/tools.yaml\n")
    put(tmp_path, "configs/tools.yaml", "- agent_tool:\n    config_path: tool.yaml\n")
    put(tmp_path, "configs/tool.yaml", f"name: tool\nmodel: {DEFAULT_MODEL}\n")
    assert validate(str(tmp_path))["ok"]


def test_include_symlink_cannot_read_outside_submission(tmp_path):
    root = tmp_path / "submission"
    root.mkdir()
    outside = put(tmp_path, "outside.md", "private")
    agent(root, "instruction: !include prompt.md\n")
    try:
        (root / "prompt.md").symlink_to(outside)
        if not (root / "prompt.md").is_symlink():
            pytest.skip("filesystem did not create symlink")
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValidationError):
        validate(str(root))


@pytest.mark.parametrize("extra", ["instruction: !include missing.md\n", "instruction: !include agent.py\n", "bad: [unfinished\n", "instruction: !include \"bad\\0path.md\"\n", "sub_agents:\n- config_path: ../escape.yaml\n"])
def test_malformed_includes_yaml_and_unsafe_references_fail_closed(tmp_path, extra):
    agent(tmp_path, extra)
    with pytest.raises(ValidationError):
        validate(str(tmp_path))


def test_agent_reference_cycles_are_rejected(tmp_path):
    agent(tmp_path, "sub_agents:\n- config_path: agent.yaml\n")
    with pytest.raises(ValidationError, match="cycle|循环"):
        validate(str(tmp_path))


def test_safetensors_outside_adapter_directory_are_rejected(tmp_path):
    agent(tmp_path)
    put(tmp_path, "weights.safetensors", "x")
    with pytest.raises(ValidationError):
        validate(str(tmp_path))


@pytest.mark.parametrize("text", ["name: root\n", "name: root\nmodel: null\n", "name: root\nmodel: 42\n"])
def test_missing_or_invalid_model_cannot_pass(tmp_path, text):
    put(tmp_path, "agent.yaml", text)
    with pytest.raises(ValidationError):
        validate(str(tmp_path))


def test_recursive_model_aliases_and_references(tmp_path):
    put(tmp_path, "agent.yaml", "name: root\nagent_class: SequentialAgent\nsub_agents:\n- config_path: sub_agents/worker.yaml\n")
    put(tmp_path, "sub_agents/worker.yaml", f"name: worker\nmodel: openai/{DEFAULT_MODEL}\n")
    assert validate(str(tmp_path))["models"] == [DEFAULT_MODEL]
    put(tmp_path, "sub_agents/worker.yaml", "name: worker\nmodel: gemma-4-e2b-it-qat-w4a16-ct\ntools:\n- agent_tool:\n    config_path: nested/tool.yaml\n")
    put(tmp_path, "sub_agents/nested/tool.yaml", f"name: tool\nmodel: {DEFAULT_MODEL}\n")
    with pytest.raises(ValidationError):
        validate(str(tmp_path))


def test_missing_referenced_agent_cannot_pass(tmp_path):
    agent(tmp_path, "sub_agents:\n- config_path: missing.yaml\n")
    with pytest.raises(ValidationError):
        validate(str(tmp_path))


@pytest.mark.parametrize("filename", ["extra.bin", "weights.pt", "extra.zip", "graph.npz", "no_extension"])
def test_unsupported_files_rejected(tmp_path, filename):
    agent(tmp_path)
    put(tmp_path, filename, "x")
    with pytest.raises(ValidationError):
        validate(str(tmp_path))


def test_allowed_files_and_json_syntax(tmp_path):
    agent(tmp_path)
    put(tmp_path, "skills/demo/scripts/run.py", "print('demo')\n")
    put(tmp_path, "adapters/demo/adapter_config.json", '{"r": 64}')
    put(tmp_path, "adapters/demo/adapter_model.safetensors", "placeholder; header validation belongs to the official runtime")
    assert validate(str(tmp_path))["lora_ranks"] == [64]
    put(tmp_path, "adapters/demo/adapter_config.json", "invalid json")
    with pytest.raises(ValidationError):
        validate(str(tmp_path))


def test_size_limit_can_be_checked_without_large_files(tmp_path):
    agent(tmp_path)
    with pytest.raises(ValidationError, match="体积"):
        validate(str(tmp_path), SubmissionLimits(max_total_size_bytes=1))


def test_emit_uses_documented_agent_discriminator():
    assert emit_agent_yaml()["agent_class"] == "LlmAgent"


def graph():
    return RepositoryGraph(key="fixture", nodes=["match.alpha", "match.beta", "other.a", "other.b"],
                           texts={"match.alpha": "alpha", "match.beta": "beta"},
                           out_adj={"match.alpha": {"other.b", "other.a"}})


def test_b1_preserves_seeds_before_expanding():
    result = build_b1(graph(), "match", LocateConfig(seed_k=2, max_k=3))
    assert result.ranked == ["match.alpha", "match.beta", "other.a"]
    assert result.seed_nodes == ["match.alpha", "match.beta"]
    assert result.expanded == 1


def test_b2_respects_max_k():
    result = build_b2(graph(), "match", LocateConfig(max_k=2))
    assert result.ranked == ["match.alpha", "other.a"]
    assert result.expanded == 1


@pytest.mark.parametrize("builder", [build_b0, build_b1, build_b2])
def test_zero_node_budget_returns_no_nodes(builder):
    result = builder(graph(), "match", LocateConfig(max_k=0))
    assert result.ranked == result.seed_nodes == []


@pytest.mark.parametrize("kwargs", [{"max_k": -1}, {"seed_k": -1}, {"char_cap": -1}])
def test_negative_budgets_are_rejected(kwargs):
    with pytest.raises(ValueError):
        build_b1(graph(), "match", LocateConfig(**kwargs))


def test_seed_budget_zero_and_larger_than_node_cap():
    assert build_b1(graph(), "match", LocateConfig(seed_k=0)).ranked == []
    result = build_b1(graph(), "match", LocateConfig(seed_k=100, max_k=2))
    assert result.ranked == result.seed_nodes == ["match.alpha", "match.beta"]
    assert result.expanded == 0


def test_negative_character_cap_rejected_and_zero_is_empty():
    with pytest.raises(ValueError):
        pack_to_char_budget(["text"], -1)
    assert pack_to_char_budget(["text"], 0) == ([0], 0)


def test_cosine_ties_break_by_node_name_independent_of_npz_order():
    ids = ["zeta", "alpha", "beta"]
    vectors = np.ones((3, 2), dtype=np.float32)
    assert [name for name, _ in cosine_topk(ids, vectors, np.ones(2), 2)] == ["alpha", "beta"]
