"""Generate model_preflight_v2.ipynb from v1 with the SM 7.5 tile patch embedded."""
import copy
import json

NB = 'paper/notebooks/model_preflight.ipynb'
OUT = 'paper/notebooks/model_preflight_v2.ipynb'

nb = json.load(open(NB, encoding='utf-8'))
v2 = copy.deepcopy(nb)

v2['cells'][0]['source'] = (
    "# ATLAS: bounded Gemma 4 E4B runtime probe (v2, SM 7.5 tile patch)\n"
    "\n"
    "v1 failed before the first token on Tesla T4: Triton unified attention requested\n"
    "98,304 B of shared memory against the 65,536 B hardware limit (vLLM issue\n"
    "#38918). v2 applies the community-verified workaround before engine start: on\n"
    "compute capability < 8.0, `_get_tile_size` falls back to tile 16 when\n"
    "`head_size > 128`. Everything else stays bounded: one attempt, one model\n"
    "variant, FP16, tensor parallelism 1, 2,048-token context, 32 generated tokens,\n"
    "subprocess with external timeout. This probe establishes generation\n"
    "compatibility only; it never measures agent repair performance."
)

src = ''.join(v2['cells'][2]['source'])
assert src.count('def run_preflight(') == 1
assert src.count('from vllm import LLM, SamplingParams') == 1

patch_src = (
    "def apply_sm75_tile_patch() -> dict:\n"
    "    \"\"\"On capability < 8.0, force tile 16 for head_size > 128 (issue #38918).\"\"\"\n"
    "    info = {'applied_modules': [], 'capability_major': None, 'verify': None}\n"
    "    try:\n"
    "        import torch\n"
    "        if not torch.cuda.is_available():\n"
    "            info['capability_major'] = 'no_cuda'\n"
    "            return info\n"
    "        major = torch.cuda.get_device_capability()[0]\n"
    "        info['capability_major'] = major\n"
    "        if major >= 8:\n"
    "            return info\n"
    "    except Exception as error:\n"
    "        info['capability_major'] = 'error: ' + str(error)\n"
    "        return info\n"
    "    import importlib\n"
    "    names = ('vllm.v1.attention.ops.triton_unified_attention',\n"
    "             'vllm.attention.ops.triton_unified_attention',\n"
    "             'vllm.v1.attention.ops.triton_unified_attention_diffkv')\n"
    "    for name in names:\n"
    "        try:\n"
    "            module = importlib.import_module(name)\n"
    "            original = module._get_tile_size\n"
    "        except Exception:\n"
    "            continue\n"
    "        def patched(*args, _original=original, **kwargs):\n"
    "            if args and isinstance(args[0], int) and args[0] > 128:\n"
    "                return 16\n"
    "            return _original(*args, **kwargs)\n"
    "        module._get_tile_size = patched\n"
    "        info['applied_modules'].append(name)\n"
    "    if info['applied_modules']:\n"
    "        try:\n"
    "            first = importlib.import_module(info['applied_modules'][0])\n"
    "            info['verify'] = {'head512_decode': first._get_tile_size(512, 0, 2, False),\n"
    "                              'head128_decode': first._get_tile_size(128, 0, 2, False)}\n"
    "        except Exception as error:\n"
    "            info['verify'] = 'error: ' + str(error)\n"
    "    return info\n"
    "\n"
    "\n"
)

# Escape for embedding inside the single-quoted Python string in the cell source.
esc = patch_src.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
new_src = src.replace('def run_preflight(', esc + 'def run_preflight(', 1)
call_line = 'result["sm75_tile_patch"] = apply_sm75_tile_patch()\\n            '
new_src = new_src.replace(
    'from vllm import LLM, SamplingParams',
    call_line + 'from vllm import LLM, SamplingParams', 1)
assert new_src != src
assert new_src.count('apply_sm75_tile_patch') == 2, new_src.count('apply_sm75_tile_patch')

v2['cells'][2]['source'] = [new_src]

v2['cells'][3]['source'] = (
    "The JSON and logs in notebook outputs are the evidence. `sm75_tile_patch` in\n"
    "the result records which modules were patched and the tile-size verification.\n"
    "A successful short generation permits a subsequent agent smoke experiment; a\n"
    "failed run closes this second bounded compatibility attempt without silently\n"
    "switching model or configuration."
)

# Compile-check the embedded probe script.
embedded = new_src.split("(WORK / 'kaggle_preflight.py').write_text('", 1)[1]
embedded = embedded.rsplit("', encoding='utf-8')", 1)[0]
embedded = embedded.replace('\\n', '\n').replace("\\'", "'").replace('\\\\', '\\')
compile(embedded, 'kaggle_preflight_v2_embedded.py', 'exec')
print('embedded script compiles OK,', len(embedded), 'chars')

# Verify the patch call landed inside the generate branch of the embedded script.
assert 'result["sm75_tile_patch"] = apply_sm75_tile_patch()' in embedded
assert embedded.index('apply_sm75_tile_patch()') < embedded.index('from vllm import LLM')

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(v2, f, ensure_ascii=False, indent=1)
print('written', OUT, 'cells:', len(v2['cells']))
