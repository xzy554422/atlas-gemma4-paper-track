"""Bounded E4B runtime probe v4: chat-template quality check + tile-16 kernel numerics.

v3 established that a source-level `_get_tile_size` guard (tile 16 when head_size > 128)
lets the quantized E4B checkpoint generate on a Tesla T4, but the output was degenerate
and repetitive. v4 answers two questions and nothing else:

1. Kernel numerics: does the tile-16 Triton attention kernel agree with a reference
   scaled-dot-product implementation for the checkpoint's head dimensions (256/512)?
2. Chat template: does applying the model's chat template (absent in v1-v3, which fed
   raw strings) change the degenerate behaviour?

No agent runs; no official scoring; one bounded subprocess.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

MODEL_NAME = 'gemma-4-e4b-it-qat-w4a16-ct'


def model_config(model_path: Path) -> dict:
    model_path = Path(model_path)
    if not (model_path / 'config.json').is_file():
        raise FileNotFoundError(f'Missing model config: {model_path / "config.json"}')
    return {
        'model': str(model_path), 'dtype': 'half',
        'tensor_parallel_size': 1, 'max_model_len': 2048,
        'gpu_memory_utilization': 0.80, 'enforce_eager': True,
        'disable_log_stats': True,
        'limit_mm_per_prompt': {'image': 0, 'video': 0, 'audio': 0},
    }


def compatibility_gate(gpus: list[dict]) -> dict:
    if not gpus:
        return {'status': 'no_cuda_gpu', 'attention_risk': None}
    turing = any(tuple(gpu['capability']) < (8, 0) for gpu in gpus)
    return {
        'status': 'requires_runtime_test', 'attention_risk': turing,
        'note': 'Memory capacity alone does not establish attention-kernel support.',
    }


def _candidate_module_paths():
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


ANCHOR = ('    # Default behavior\n'
          '    if is_prefill:\n'
          '        return 32\n')
GUARD = ('    # SM75_TILE_GUARD: Turing cannot allocate 3*tile*head*2 bytes of shared\n'
         '    # memory for 512-wide heads (vLLM issue #38918). Fall back to tile 16.\n'
         '    if head_size > 128:\n'
         '        return 16\n')


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


def save(result: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def numeric_check(result: dict, output_path: Path) -> None:
    """Compare the tile-16 Triton kernel against a float32 SDPA reference."""
    import torch
    import torch.nn.functional as F
    stage = {'cases': [], 'status': 'failed'}
    result['numeric_check'] = stage
    save(result, output_path)
    try:
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention
        torch.manual_seed(0)
        device = 'cuda'
        stage['tile_prefill_head512'] = __import__(
            'vllm.v1.attention.ops.triton_unified_attention',
            fromlist=['_get_tile_size'])._get_tile_size(512, 0, 2, True)
        for head_dim, num_heads, kv_len, decode in (
            (512, 4, 128, False),
            (512, 4, 1, True),
            (256, 4, 128, False),
            (256, 4, 1, True),
        ):
            block_size = 16
            q_len = 1 if decode else kv_len
            num_blocks = (kv_len + block_size - 1) // block_size
            q = torch.randn(q_len, num_heads, head_dim, dtype=torch.float16, device=device)
            k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim,
                                  dtype=torch.float16, device=device)
            v_cache = torch.randn_like(k_cache)
            out = torch.empty_like(q)
            block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, -1)
            seqused_k = torch.tensor([kv_len], dtype=torch.int32, device=device)
            cu = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            unified_attention(q, k_cache, v_cache, out, cu, q_len, seqused_k, kv_len,
                              1.0 / math.sqrt(head_dim), True, (-1, -1), block_table,
                              0.0, None, None, None)
            k_seq = k_cache.reshape(-1, num_heads, head_dim)[:kv_len]
            v_seq = v_cache.reshape(-1, num_heads, head_dim)[:kv_len]
            ref = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0).float(),
                k_seq.transpose(0, 1).unsqueeze(0).float(),
                v_seq.transpose(0, 1).unsqueeze(0).float(),
                is_causal=not decode)
            ref = ref.squeeze(0).transpose(0, 1)
            triton_err = (out.float() - ref).abs().max().item()
            sdpa16 = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k_seq.transpose(0, 1).unsqueeze(0),
                v_seq.transpose(0, 1).unsqueeze(0),
                is_causal=not decode).squeeze(0).transpose(0, 1)
            sdpa_err = (sdpa16.float() - ref).abs().max().item()
            rel = triton_err / ref.abs().max().item()
            stage['cases'].append({
                'head_dim': head_dim, 'kv_len': kv_len, 'decode': decode,
                'triton_max_abs_err_fp32ref': round(triton_err, 6),
                'sdpa_fp16_max_abs_err_fp32ref': round(sdpa_err, 6),
                'triton_rel_err': round(rel, 6),
            })
        worst = max(c['triton_rel_err'] for c in stage['cases'])
        stage['worst_rel_err'] = worst
        stage['status'] = 'pass' if worst < 0.05 else 'fail'
    except Exception as error:
        stage['error_type'] = type(error).__name__
        stage['error'] = str(error)[:800]
    save(result, output_path)


def manual_gemma(text: str) -> str:
    return f'<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n'


def extract_text(chat_result):
    if isinstance(chat_result, list):
        return chat_result[0].outputs[0].text
    return chat_result.outputs[0].text


def chat_probe(result: dict, llm, output_path: Path) -> None:
    from vllm import SamplingParams
    stage = {'prompts': [], 'status': 'failed'}
    result['chat_probe'] = stage
    save(result, output_path)
    prompts = [
        ('sum', 'Write a Python function add(a, b) that returns their sum.'),
        ('math', 'What is 17 * 24? Reply with only the number.'),
        ('unit_test', 'Explain in one sentence what a unit test is.'),
    ]
    try:
        for name, text in prompts:
            entry = {'name': name, 'prompt': text}
            raw = llm.generate([text], SamplingParams(max_tokens=32, temperature=0))
            entry['raw_32tok'] = raw[0].outputs[0].text[:300]
            try:
                chat = llm.chat([{'role': 'user', 'content': text}],
                                SamplingParams(max_tokens=64, temperature=0))
                entry['chat_path'] = 'tokenizer_chat_template'
            except Exception as error:
                entry['chat_path'] = 'manual_gemma_format (template error: ' + str(error)[:100] + ')'
                chat = llm.generate([manual_gemma(text)],
                                    SamplingParams(max_tokens=64, temperature=0))
            entry['chat_64tok'] = extract_text(chat)[:300]
            stage['prompts'].append(entry)
            save(result, output_path)
        stage['status'] = 'pass'
    except Exception as error:
        stage['error_type'] = type(error).__name__
        stage['error'] = str(error)[:800]
    save(result, output_path)


def run_preflight(model_path: Path | None, *, generate: bool, output_path: Path) -> dict:
    started = time.monotonic()
    result = {
        'schema_version': 4, 'model': MODEL_NAME,
        'python': platform.python_version(), 'platform': platform.platform(),
        'generation_status': 'not_requested', 'agent_status': 'not_run',
        'configuration_scope': 'short-context runtime probe, not official scoring',
        'packages': {}, 'gpus': [],
    }
    for package in ('torch', 'vllm', 'transformers', 'compressed-tensors', 'swegemma', 'adk-submission'):
        try:
            result['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][package] = None
    try:
        import torch
        if torch.cuda.is_available():
            result['gpus'] = [
                {'name': torch.cuda.get_device_name(i),
                 'capability': list(torch.cuda.get_device_capability(i)),
                 'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    result['compatibility'] = compatibility_gate(result['gpus'])
    save(result, output_path)
    if not generate:
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        return result
    result['generation_status'] = 'failed'
    try:
        if not result['gpus']:
            raise RuntimeError('No CUDA GPU is available; no model loading attempted.')
        if model_path is None:
            raise FileNotFoundError('Provide an attached E4B checkpoint path.')
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        result['sm75_tile_patch'] = apply_sm75_tile_patch()
        save(result, output_path)
        numeric_check(result, output_path)
        if result['numeric_check']['status'] != 'pass':
            raise RuntimeError(
                'Tile-16 kernel numerics failed; skipping model load. '
                + json.dumps(result['numeric_check'])[:400])
        from vllm import LLM, SamplingParams
        checkpoint = json.loads((Path(model_path) / 'config.json').read_text(encoding='utf-8'))
        text = checkpoint.get('text_config', checkpoint)
        result['checkpoint_attention'] = {k: text.get(k) for k in ('head_dim', 'global_head_dim', 'dtype', 'torch_dtype')}
        load_started = time.monotonic()
        llm = LLM(**model_config(model_path))
        result['load_seconds'] = round(time.monotonic() - load_started, 3)
        result['generation_status'] = 'passed'
        result['agent_status'] = 'not_run'
        save(result, output_path)
        chat_probe(result, llm, output_path)
    except Exception as error:
        result['error_type'] = type(error).__name__
        result['error'] = str(error)[:1200]
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    save(result, output_path)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_preflight(args.model_path, generate=args.generate, output_path=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(args.generate and result.get('chat_probe', {}).get('status') != 'pass')


if __name__ == '__main__':
    sys.exit(main())
