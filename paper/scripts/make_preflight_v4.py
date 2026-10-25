"""Generate model_preflight_v4.ipynb from v3, embedding kaggle_preflight_v4.py."""
import ast
import json
from pathlib import Path

V3 = Path('paper/notebooks/model_preflight_v3.ipynb')
PROBE = Path('paper/scripts/kaggle_preflight_v4.py')
OUT = Path('paper/notebooks/model_preflight_v4.ipynb')


def decode_embedded(cell_source: str) -> str:
    tree = ast.parse(cell_source)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'write_text' and node.args):
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
    raise ValueError('probe script literal not found')


def patch_section(source: str) -> str:
    start = source.index('def _candidate_module_paths(')
    ends = [source.index(m) for m in ('def run_preflight(', 'def save(') if m in source]
    return source[start:min(ends)]


def main():
    nb = json.loads(V3.read_text(encoding='utf-8'))
    v3_probe = decode_embedded(''.join(nb['cells'][2]['source']))
    v4_probe = PROBE.read_text(encoding='utf-8')

    # The verified patch code must be byte-identical between v3 and v4.
    assert patch_section(v3_probe) == patch_section(v4_probe), 'patch section drifted'

    # Runner wrapper: everything before the write_text literal and from "MODEL = Path"
    # onward. The old split-on-encoding trick breaks because the probe itself contains
    # `encoding='utf-8'` literals.
    cell2 = ''.join(nb['cells'][2]['source'])
    marker = '\nMODEL = Path('
    head = cell2[:cell2.index('.write_text(')]
    tail = cell2[cell2.index(marker):]
    nb['cells'][2]['source'] = [head + '.write_text(' + json.dumps(v4_probe)
                                + ", encoding='utf-8')" + tail]

    nb['cells'][0]['source'] = (
        "# ATLAS: bounded Gemma 4 E4B runtime probe (v4, kernel numerics + chat template)\n"
        "\n"
        "v3 proved that the source-level tile guard makes the quantized E4B checkpoint run on\n"
        "a Tesla T4, but the 32 generated tokens were degenerate. v4 answers two questions and\n"
        "nothing else:\n"
        "\n"
        "1. **Kernel numerics** — the tile-16 Triton attention kernel is compared against a\n"
        "   float32 SDPA reference for head_dim 256/512, prefill and decode. If it disagrees,\n"
        "   the guard is unusable and every downstream experiment is void.\n"
        "2. **Chat template** — v1-v3 fed raw strings to an instruction-tuned model; v4 repeats\n"
        "   the same prompts through the tokenizer's chat template (with a manual Gemma-format\n"
        "   fallback) to test whether the degenerate output is a prompting artifact.\n"
        "\n"
        "The numeric check runs before the model load and aborts the probe if it fails.\n"
        "Everything stays bounded: one attempt, FP16, tensor parallelism 1, 2,048-token\n"
        "context, subprocess with external timeout. This probe establishes runtime\n"
        "compatibility and output character only; it never measures agent repair performance."
    )
    nb['cells'][3]['source'] = (
        "The JSON and logs in notebook outputs are the evidence. `numeric_check` reports the\n"
        "tile used and worst relative error against the float32 reference; `chat_probe`\n"
        "records raw vs templated completions for three prompts. A passing numeric check with\n"
        "coherent templated output permits a subsequent single-task agent smoke experiment;\n"
        "anything else closes this line without silently changing configuration."
    )

    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')

    embedded = decode_embedded(''.join(nb['cells'][2]['source']))
    compile(embedded, 'probe_v4_embedded.py', 'exec')
    assert 'numeric_check' in embedded and 'chat_probe' in embedded
    assert embedded.index('apply_sm75_tile_patch()') < embedded.index('from vllm import LLM')
    print('v4 written:', OUT, '| embedded chars:', len(embedded))


if __name__ == '__main__':
    main()
