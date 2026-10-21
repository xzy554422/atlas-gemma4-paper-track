"""Generate model_preflight_v3.ipynb: patch vLLM's tile size at the source-file level.

v2 patched the imported module in the parent process, but vLLM runs attention inside a
spawned EngineCore subprocess that re-imports vLLM from disk, so the monkey-patch was
lost and the T4 still requested 98,304 B of shared memory. v3 rewrites the installed
`triton_unified_attention.py` source before vLLM is imported, so every process
(including the spawned engine core) sees the guard.
"""
import json
from pathlib import Path

V2 = Path('paper/notebooks/model_preflight_v2.ipynb')
OUT = Path('paper/notebooks/model_preflight_v3.ipynb')
BACKSLASH = chr(92)


def decode_embedded(cell_source: str) -> str:
    """Extract the probe script literal from a cell that calls write_text(...)."""
    import ast
    tree = ast.parse(cell_source)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'write_text' and node.args):
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
    raise ValueError('probe script literal not found in cell source')


PATCH_FN = '''def _candidate_module_paths():
    """Locate triton_unified_attention.py without importing the module."""
    import glob as _glob
    paths = []
    try:
        import importlib.util
        spec = importlib.util.find_spec('vllm.v1.attention.ops.triton_unified_attention')
        if spec and spec.origin:
            paths.append(Path(spec.origin))
    except Exception:
        pass
    for pattern in ('/usr/local/lib/python*/dist-packages/vllm/v1/attention/ops/triton_unified_attention.py',
                    '/usr/local/lib/python*/site-packages/vllm/v1/attention/ops/triton_unified_attention.py'):
        paths.extend(Path(p) for p in _glob.glob(pattern))
    seen, unique = set(), []
    for path in paths:
        key = str(path)
        if key not in seen and path.is_file():
            seen.add(key)
            unique.append(path)
    return unique


ANCHOR = ('    # Default behavior\\n'
          '    if is_prefill:\\n'
          '        return 32\\n')
GUARD = ('    # SM75_TILE_GUARD: Turing cannot allocate 3*tile*head*2 bytes of shared\\n'
         '    # memory for 512-wide heads (vLLM issue #38918). Fall back to tile 16.\\n'
         '    if head_size > 128:\\n'
         '        return 16\\n')


def apply_sm75_tile_patch(extra_paths=None) -> dict:
    """Rewrite the installed vLLM source so spawned engine processes inherit the guard."""
    info = {'source_files': [], 'capability_major': None, 'applied_modules': [], 'verify': None}
    try:
        import torch
        if not torch.cuda.is_available():
            info['capability_major'] = 'no_cuda'
            return info
        major = torch.cuda.get_device_capability()[0]
        info['capability_major'] = major
        if major >= 8:
            info['skipped'] = 'capability >= 8.0'
            return info
    except Exception as error:
        info['capability_major'] = 'error: ' + str(error)
        return info

    paths = list(extra_paths or []) + _candidate_module_paths()
    for path in paths:
        try:
            text = path.read_text(encoding='utf-8')
        except OSError as error:
            info['source_files'].append({'path': str(path), 'status': 'unreadable: ' + str(error)})
            continue
        if 'SM75_TILE_GUARD' in text:
            info['source_files'].append({'path': str(path), 'status': 'already_patched'})
            continue
        if ANCHOR not in text:
            info['source_files'].append({'path': str(path), 'status': 'anchor_not_found'})
            continue
        patched = text.replace(ANCHOR, GUARD + ANCHOR, 1)
        try:
            path.write_text(patched, encoding='utf-8')
        except OSError as error:
            info['source_files'].append({'path': str(path), 'status': 'unwritable: ' + str(error)})
            continue
        import hashlib
        info['source_files'].append({
            'path': str(path), 'status': 'patched',
            'sha256_before': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'sha256_after': hashlib.sha256(patched.encode('utf-8')).hexdigest(),
        })

    # Also patch any already-imported copy so the parent process stays consistent.
    import importlib
    for name in ('vllm.v1.attention.ops.triton_unified_attention',
                 'vllm.attention.ops.triton_unified_attention',
                 'vllm.v1.attention.ops.triton_unified_attention_diffkv'):
        try:
            module = importlib.import_module(name)
            original = module._get_tile_size
        except Exception:
            continue
        def patched_fn(*args, _original=original, **kwargs):
            if args and isinstance(args[0], int) and args[0] > 128:
                return 16
            return _original(*args, **kwargs)
        module._get_tile_size = patched_fn
        info['applied_modules'].append(name)
    if info['applied_modules']:
        try:
            first = importlib.import_module(info['applied_modules'][0])
            info['verify'] = {'head512_decode': first._get_tile_size(512, 0, 2, False),
                              'head512_prefill': first._get_tile_size(512, 0, 2, True)}
        except Exception as error:
            info['verify'] = 'error: ' + str(error)
    return info


'''


def main():
    nb = json.loads(V2.read_text(encoding='utf-8'))
    probe = decode_embedded(''.join(nb['cells'][2]['source']))

    start = probe.index('def apply_sm75_tile_patch(')
    end = probe.index('def run_preflight(')
    probe = probe[:start] + PATCH_FN + probe[end:]

    # v3 intro
    nb['cells'][0]['source'] = (
        "# ATLAS: bounded Gemma 4 E4B runtime probe (v3, source-level SM 7.5 guard)\n"
        "\n"
        "v2 applied the tile-size fix by monkey-patching the imported module, but vLLM runs\n"
        "attention inside a spawned EngineCore subprocess that re-imports vLLM from disk, so\n"
        "the patch was lost and the T4 still requested 98,304 B of shared memory.\n"
        "\n"
        "v3 rewrites the installed `triton_unified_attention.py` before vLLM is imported:\n"
        "`_get_tile_size` returns tile 16 when `head_size > 128`. Every process, including\n"
        "the spawned engine core, then reads the guarded source. The rewrite records SHA256\n"
        "before/after and is confined to compute capability < 8.0. Everything else stays\n"
        "bounded: one attempt, FP16, tensor parallelism 1, 2,048-token context, 32 generated\n"
        "tokens, subprocess with external timeout. This probe establishes generation\n"
        "compatibility only; it never measures agent repair performance."
    )

    # Rebuild cell 2 with a clean JSON-escaped embedding.
    cell2 = ''.join(nb['cells'][2]['source'])
    head = cell2.split("(WORK / 'kaggle_preflight.py').write_text(", 1)[0]
    tail = cell2.split("', encoding='utf-8')", 1)[1]
    nb['cells'][2]['source'] = [head + "(WORK / 'kaggle_preflight.py').write_text("
                                + json.dumps(probe) + ", encoding='utf-8')" + tail]

    nb['cells'][3]['source'] = (
        "The JSON and logs in notebook outputs are the evidence. `sm75_tile_patch` records\n"
        "which source files were rewritten, their SHA256 before and after, and the tile-size\n"
        "verification. A successful short generation permits a subsequent agent smoke\n"
        "experiment; a failed run closes this compatibility attempt without silently\n"
        "switching model or configuration."
    )

    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')

    # Validate: the embedded script must compile and carry the source-level guard.
    embedded = decode_embedded(''.join(nb['cells'][2]['source']))
    compile(embedded, 'probe_v3.py', 'exec')
    assert 'SM75_TILE_GUARD' in embedded
    assert 'ANCHOR' in embedded
    assert embedded.index('apply_sm75_tile_patch') < embedded.index('from vllm import LLM')
    print('v3 written:', OUT, '| embedded chars:', len(embedded))


if __name__ == '__main__':
    main()
