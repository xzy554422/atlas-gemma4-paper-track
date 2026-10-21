"""Verify the SM 7.5 tile patch embedded in model_preflight_v2.ipynb, using a fake vLLM module."""
import json
import sys
import types

BACKSLASH = chr(92)
NEWLINE_ESC = BACKSLASH + 'n'
QUOTE_ESC = BACKSLASH + "'"


def load_embedded_script(path='paper/notebooks/model_preflight_v2.ipynb'):
    nb = json.load(open(path, encoding='utf-8'))
    src = ''.join(nb['cells'][2]['source'])
    embedded = src.split("(WORK / 'kaggle_preflight.py').write_text('", 1)[1]
    embedded = embedded.rsplit("', encoding='utf-8')", 1)[0]
    return (embedded.replace(NEWLINE_ESC, '\n')
                    .replace(QUOTE_ESC, "'")
                    .replace(BACKSLASH + BACKSLASH, BACKSLASH))


def install_fake_vllm():
    mod = types.ModuleType('vllm.v1.attention.ops.triton_unified_attention')

    def _get_tile_size(head_size, sliding_window, element_size, is_prefill):
        if sliding_window == 1024 and head_size in (128, 256):
            return 32
        if is_prefill:
            return 32
        return 16 if element_size >= 2 else 32

    mod._get_tile_size = _get_tile_size

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability():
            return (7, 5)

    torch = types.ModuleType('torch')
    torch.cuda = FakeCuda
    sys.modules['torch'] = torch
    for name in ('vllm', 'vllm.v1', 'vllm.v1.attention', 'vllm.v1.attention.ops'):
        sys.modules[name] = types.ModuleType(name)
    sys.modules['vllm.v1.attention.ops.triton_unified_attention'] = mod
    return mod


def main():
    embedded = load_embedded_script()
    mod = install_fake_vllm()
    ns = {}
    exec(compile(embedded, 'probe.py', 'exec'), ns)

    before_decode = mod._get_tile_size(512, 0, 2, False)
    before_prefill = mod._get_tile_size(512, 0, 2, True)
    info = ns['apply_sm75_tile_patch']()
    print('patch info:', info)
    print(f'before patch: head512 decode={before_decode} prefill={before_prefill}')
    print('shared-memory estimate: 3 * tile * head * 2 bytes ->',
          f'{3 * before_prefill * 512 * 2} B before, {3 * 16 * 512 * 2} B after (T4 limit 65536)')
    assert before_prefill == 32, 'expected default prefill tile of 32'

    cases = [
        (512, 2, False, 16),
        (256, 2, False, 16),
        (128, 2, False, 16),
        (64, 2, False, 16),
        (512, 2, True, 16),
    ]
    for head, elem, prefill, expect in cases:
        got = mod._get_tile_size(head, 0, elem, prefill)
        label = f'head={head} elem={elem} prefill={prefill}'
        print(f'{label:34s} -> {got} (expect {expect})')
        assert got == expect, (label, got, expect)

    # Unpatched path must survive for fp8 (tile 32) when head_size <= 128.
    assert mod._get_tile_size(64, 0, 1, False) == 32, 'fp8 tile rule broken'
    print('fp8 head64 decode ->', mod._get_tile_size(64, 0, 1, False), '(expect 32)')
    print('PATCH LOGIC OK')


if __name__ == '__main__':
    main()
