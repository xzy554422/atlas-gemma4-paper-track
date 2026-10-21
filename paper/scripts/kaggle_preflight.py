"""Bounded E4B runtime probe. No model is loaded unless --generate is supplied.

The probe establishes generation compatibility only, never an agent score.
Run in a subprocess with an external timeout so failed GPU startup is bounded.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
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


def run_preflight(model_path: Path | None = None, *, generate: bool = False) -> dict:
    started = time.monotonic()
    result = {
        'schema_version': 1, 'model': MODEL_NAME,
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
    if generate:
        result['generation_status'] = 'failed'
        try:
            if not result['gpus']:
                raise RuntimeError('No CUDA GPU is available; no model loading attempted.')
            if model_path is None:
                raise FileNotFoundError('Provide an attached E4B checkpoint path.')
            config = model_config(model_path)
            checkpoint = json.loads((Path(model_path) / 'config.json').read_text(encoding='utf-8'))
            result['checkpoint_architectures'] = checkpoint.get('architectures')
            text = checkpoint.get('text_config', checkpoint)
            result['checkpoint_attention'] = {k: text.get(k) for k in ('head_dim', 'global_head_dim', 'dtype', 'torch_dtype')}
            result['runtime_config'] = config
            os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
            os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
            from vllm import LLM, SamplingParams
            llm = LLM(**config)
            outputs = llm.generate(['Write a Python function add(a, b) that returns their sum.'],
                                   SamplingParams(max_tokens=32, temperature=0))
            text_output = outputs[0].outputs[0].text
            token_count = len(outputs[0].outputs[0].token_ids)
            if not token_count:
                raise RuntimeError('The runtime returned zero generated tokens.')
            result.update(generation_status='passed', generated_text=text_output,
                          generated_tokens=token_count)
        except Exception as error:
            result['error_type'] = type(error).__name__
            result['error'] = str(error)[:1200]
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_preflight(args.model_path, generate=args.generate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(args.generate and result['generation_status'] != 'passed')


if __name__ == '__main__':
    sys.exit(main())
