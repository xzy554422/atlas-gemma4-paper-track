"""Verify the v3 source-rewrite patch against the real vLLM 0.19.1 function text."""
import ast
import json
import sys
import tempfile
import types
from pathlib import Path

NB = Path('paper/notebooks/model_preflight_v3.ipynb')

VLLM_FUNC = '''def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization."""
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    return 16 if element_size >= 2 else 32
'''


def extract_probe() -> str:
    nb = json.loads(NB.read_text(encoding='utf-8'))
    src = ''.join(nb['cells'][2]['source'])
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'write_text' and node.args):
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
    raise SystemExit('probe literal not found')


def fake_torch(major=7):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability():
            return (major, 5)
    torch = types.ModuleType('torch')
    torch.cuda = FakeCuda
    sys.modules['torch'] = torch


def main():
    probe = extract_probe()
    ns = {}
    exec(compile(probe, 'probe_v3.py', 'exec'), ns)
    apply = ns['apply_sm75_tile_patch']

    with tempfile.TemporaryDirectory() as tmp:
        mod_path = Path(tmp) / 'triton_unified_attention.py'
        mod_path.write_text(VLLM_FUNC, encoding='utf-8')

        # Case 1: Turing (capability 7.5) -> file must be rewritten.
        fake_torch(7)
        info = apply(extra_paths=[mod_path])
        print('capability:', info['capability_major'])
        print('source_files:', json.dumps(info['source_files'], indent=1))
        text = mod_path.read_text(encoding='utf-8')
        assert 'SM75_TILE_GUARD' in text, 'guard not written'
        assert text.count('SM75_TILE_GUARD') == 1, 'guard written twice'
        before = info['source_files'][0]['sha256_before']
        after = info['source_files'][0]['sha256_after']
        assert before != after, 'hash unchanged'
        print('hashes differ as expected:', before[:12], '->', after[:12])

        # Execute the patched file and confirm the tile values.
        mod_ns = {'_is_gemma3_attention': lambda h, s: s == 1024 and h in (128, 256)}
        exec(compile(text, 'patched.py', 'exec'), mod_ns)
        tile = mod_ns['_get_tile_size']
        checks = [((512, 0, 2, True), 16), ((512, 0, 2, False), 16),
                  ((256, 0, 2, True), 16), ((128, 0, 2, False), 16)]
        for args, expect in checks:
            got = tile(*args)
            print(f'_get_tile_size{args} -> {got} (expect {expect})')
            assert got == expect, (args, got, expect)

        smem_before = 3 * 32 * 512 * 2
        smem_after = 3 * 16 * 512 * 2
        print(f'shared memory: {smem_before} B -> {smem_after} B (T4 limit 65536)')
        assert smem_after <= 65536 < smem_before

        # Case 2: idempotent - a second run must not double-insert.
        info2 = apply(extra_paths=[mod_path])
        statuses = [f['status'] for f in info2['source_files']]
        print('second run statuses:', statuses)
        assert 'already_patched' in statuses
        assert mod_path.read_text(encoding='utf-8').count('SM75_TILE_GUARD') == 1

    # Case 3: capability >= 8.0 must leave files untouched.
    with tempfile.TemporaryDirectory() as tmp:
        mod_path = Path(tmp) / 'triton_unified_attention.py'
        mod_path.write_text(VLLM_FUNC, encoding='utf-8')
        fake_torch(8)
        info3 = apply(extra_paths=[mod_path])
        print('capability 8.0 skipped:', info3.get('skipped'))
        assert info3.get('skipped') == 'capability >= 8.0'
        assert 'SM75_TILE_GUARD' not in mod_path.read_text(encoding='utf-8')

    print('V3 SOURCE PATCH OK')


if __name__ == '__main__':
    main()
