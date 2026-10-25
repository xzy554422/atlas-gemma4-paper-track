"""Direction 3/5/6 runner upgrade: run_command, compaction strategies, guidance."""
from pathlib import Path

P = Path('paper/scripts/agent_smoke_v5.py')
s = P.read_text(encoding='utf-8')
BS = chr(92)

# --- 1. CLI flags -----------------------------------------------------------
s = s.replace(
    "    parser.add_argument('--localization-mode', type=str, default='index',\n"
    "                        choices=['index', 'oracle', 'b0'])",
    "    parser.add_argument('--localization-mode', type=str, default='index',\n"
    "                        choices=['index', 'oracle', 'b0'])\n"
    "    parser.add_argument('--compaction-strategy', type=str, default='truncate',\n"
    "                        choices=['truncate', 'state'])\n"
    "    parser.add_argument('--verification-guidance', type=str, default='none',\n"
    "                        choices=['none', 'selected', 'full'])\n"
    "    parser.add_argument('--allow-run-command', action='store_true')")

# --- 2. batch passes the flags through --------------------------------------
old_single = (
    "                  '--localization-mode', args.localization_mode,\n"
    "                  '--compaction-strategy', args.compaction_strategy,\n"
    "                  '--verification-guidance', args.verification_guidance,\n"
    "                  '--allow-run-command'] if False else single + ['--localization-mode', args.localization_mode,\n"
    "                  '--compaction-strategy', args.compaction_strategy,\n"
    "                  '--verification-guidance', args.verification_guidance] + (['--allow-run-command'] if args.allow_run_command else [])")
new_single = (
    "                  '--localization-mode', args.localization_mode,\n"
    "                  '--compaction-strategy', args.compaction_strategy,\n"
    "                  '--verification-guidance', args.verification_guidance,\n"
    "                  ] + (['--allow-run-command'] if args.allow_run_command else [])")
if old_single in s:
    s = s.replace(old_single, new_single, 1)
elif new_single not in s:
    raise AssertionError('single-arg anchor missing and new form absent')

# --- 3. main(): budgets + allow flag ----------------------------------------
old_budgets = (
    "        budgets = {'tool_calls': args.tool_budget, 'time_minutes': args.time_minutes,\n"
    "                   'localization_mode': args.localization_mode}")
new_budgets = (
    "        budgets = {'tool_calls': args.tool_budget, 'time_minutes': args.time_minutes,\n"
    "                   'localization_mode': args.localization_mode,\n"
    "                   'compaction_strategy': args.compaction_strategy,\n"
    "                   'verification_guidance': args.verification_guidance}\n"
    "        allow_run_command = args.allow_run_command")
assert old_budgets in s, 'budgets anchor missing'
s = s.replace(old_budgets, new_budgets, 1)

old_call = (
    "        summary = run_agent_loop(args.output_dir / 'workspace', task, llm, sampling,\n"
    "                                 budgets, args.output_dir, log)")
new_call = (
    "        summary = run_agent_loop(args.output_dir / 'workspace', task, llm, sampling,\n"
    "                                 budgets, args.output_dir, log,\n"
    "                                 allow_run_command=allow_run_command)")
assert old_call in s, 'call anchor missing'
s = s.replace(old_call, new_call, 1)

# --- 4. guidance section builder + run_agent_loop signature -----------------
NL = chr(10)
guidance_fn = (
    "def verification_guidance_section(task: dict, mode: str) -> str:" + NL
    + "    if mode == 'none':" + NL
    + "        return ''" + NL
    + "    files = sorted(set(re.findall(r'^" + chr(43) + chr(43) + chr(43)
      + " b/(\\S+)', task.get('test_patch', '') or '', re.M)))" + NL
    + "    if not files:" + NL
    + "        return ''" + NL
    + "    if mode == 'full':" + NL
    + "        return ('## Verification guidance (full)" + BS + "n'" + NL
    + "                'The repository test files are: ' + ', '.join(files) + '.' + BS + 'n')" + NL
    + "    fns = []" + NL
    + "    for line in (task.get('test_patch', '') or '').splitlines():" + NL
    + "        m = re.match(r'" + chr(43) + "\\s*def (test_" + BS + "w+)" + chr(92)
      + "((' + BS + '\", line)" + NL
    + "        if m:" + NL
    + "            fns.append(m.group(1))" + NL
    + "    if not fns:" + NL
    + "        return ''" + NL
    + "    return ('## Verification guidance (selected)" + BS + "n'" + NL
    + "            'The test patch adds these tests, which must pass after your fix: '" + NL
    + "            + ', '.join(fns) + '.' + BS + 'n')" + NL
    + NL + NL)

old_sig = (
    "def run_agent_loop(workspace: Path, task: dict, llm, sampling, budgets: dict,"
    + NL + "                   output_dir: Path, log) -> dict:")
new_sig = (guidance_fn + NL
           + "def run_agent_loop(workspace: Path, task: dict, llm, sampling, budgets: dict,"
           + NL + "                   output_dir: Path, log, allow_run_command: bool = False) -> dict:")
assert old_sig in s, 'sig anchor missing'
s = s.replace(old_sig, new_sig, 1)

# --- 5. executor + system instruction + guidance wiring ---------------------
old_exec = (
    "    executor = ToolExecutor(workspace, budgets['tool_calls'], budgets['time_minutes'] * 60, started)")
new_exec = (
    "    executor = ToolExecutor(workspace, budgets['tool_calls'], budgets['time_minutes'] * 60, started,"
    + NL + "                            allow_run_command=allow_run_command)")
assert old_exec in s, 'executor anchor missing'
s = s.replace(old_exec, new_exec, 1)

old_sys = (
    "    messages = [" + NL
    + "        {'role': 'system', 'content': SYSTEM_INSTRUCTION},")
new_sys = (
    "    messages = [" + NL
    + "        {'role': 'system', 'content': system_instruction(allow_run_command)},")
assert old_sys in s, 'sys anchor missing'
s = s.replace(old_sys, new_sys, 1)

old_loc = (
    "    mode = budgets.get('localization_mode', 'index')" + NL
    + "    localization_section = build_localization_section(workspace, task, mode)")
new_loc = (old_loc + NL
    + "    guidance_section = verification_guidance_section(" + NL
    + "        task, budgets.get('verification_guidance', 'none'))")
assert old_loc in s, 'loc anchor missing'
s = s.replace(old_loc, new_loc, 1)

old_fmt = (
    "            localization_section=localization_section," + NL
    + "            layout=workspace_layout(workspace))},")
new_fmt = (
    "            localization_section=localization_section," + NL
    + "            guidance_section=guidance_section," + NL
    + "            layout=workspace_layout(workspace))},")
assert old_fmt in s, 'fmt anchor missing'
s = s.replace(old_fmt, new_fmt, 1)

old_tmpl = '    "{symbols}' + BS + 'n"'
assert old_tmpl in s, 'tmpl anchor missing'
new_tmpl = '    "{symbols}' + BS + 'n"' + NL + '    "{guidance_section}' + BS + 'n"'
s = s.replace(old_tmpl, new_tmpl, 1)

# --- 6. compaction strategy branch ------------------------------------------
old_comp = (
    "        if context_chars(messages) > CONTEXT_CHAR_LIMIT:" + NL
    + "            viewed = '; '.join(f'{k} {v}' for k, v in executor.files_viewed.items()) or 'none'")
new_comp = (
    "        if context_chars(messages) > CONTEXT_CHAR_LIMIT:" + NL
    + "            strategy = budgets.get('compaction_strategy', 'truncate')" + NL
    + "            if strategy == 'state':" + NL
    + "                edits = ', '.join(executor.edits[-8:]) or 'none'" + NL
    + "                viewed = '; '.join(k + ' ' + str(v[-1]) for k, v in" + NL
    + "                                   list(executor.files_viewed.items())[-8:]) or 'none'" + NL
    + "                note = ('state: ' + str(executor.tool_calls_used) + ' tool calls; edits in '" + NL
    + "                        + edits + '; last views: ' + viewed)" + NL
    + "                before = context_chars(messages)" + NL
    + "                messages = [messages[0], messages[1]," + NL
    + "                            {'role': 'user', 'content': '[Context compaction] ' + note}]" + NL
    + "                record(turn, action='compact', note='state " + BS
      + "u2192 ' + str(context_chars(messages)) + ' chars')" + NL
    + "                turn += 1" + NL
    + "                continue" + NL
    + "            files_edited")
assert old_comp in s, 'comp anchor missing'
s = s.replace(old_comp, new_comp, 1)

P.write_text(s, encoding='utf-8')
compile(s, 'agent_smoke_v5.py', 'exec')
print('runner upgraded and compiles')
