"""CPU-only regression tests for smoke-agent inference boundaries."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'agent_smoke_v5.py'
spec = importlib.util.spec_from_file_location('agent_smoke_v5_focused', SCRIPT)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        # Deliberately expensive Unicode/token ratio plus template overhead.
        return [0] * (sum(len(m['content']) for m in messages) * 2 + 20 * len(messages))


class SmokeTests(unittest.TestCase):
    def test_request_and_completion_are_kept_separate(self):
        request = SimpleNamespace(prompt_token_ids=[1, 2], outputs=[object()])
        for result in (request, [request]):
            self.assertEqual(agent.extract_text(result), (request, request.outputs[0]))

    def test_token_budget_large_initial_and_history(self):
        for window in (6144, 8192):
            for strategy in ('truncate', 'state'):
                messages = [{'role': 'system', 'content': 'instructions'},
                            {'role': 'user', 'content': 'issue界' * 10000}]
                messages += [{'role': role, 'content': 'data界' * 4000}
                             for role in ('assistant', 'user') for _ in range(3)]
                fitted, count = agent.fit_prompt(messages, Tokenizer(), window - 1024,
                                                 'state', strategy)
                self.assertLessEqual(count + 1024, window)
                self.assertEqual(fitted[0], messages[0])
                self.assertEqual(messages[1]['content'], 'issue界' * 10000)
                self.assertIn('truncated', fitted[1]['content'])

    def test_unshrinkable_template_is_bounded(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [0] * 9000
        with self.assertRaisesRegex(ValueError, 'cannot fit'):
            agent.fit_prompt([{'role': 'system', 'content': 'x'},
                              {'role': 'user', 'content': 'y' * 10000}], tokenizer, 5120, '')
        self.assertLessEqual(tokenizer.apply_chat_template.call_count,
                             agent.MAX_PROMPT_SHRINKS + 1)

    def test_dispatch_command_permission(self):
        for allowed in (False, True):
            executor = agent.ToolExecutor(Path('.'), 5, 60, time.monotonic(), allowed)
            executor.run_command = Mock(return_value={'status': 'ok'})
            result = executor.dispatch('run_command', {'command': 'true'})
            self.assertEqual(executor.run_command.call_count, int(allowed))
            self.assertEqual(result['status'], 'ok' if allowed else 'error')

    def run_loop(self, root, chat, window=6144):
        llm = Mock()
        llm.get_tokenizer.return_value = Tokenizer()
        responses = iter(chat) if isinstance(chat, list) else None
        def checked_chat(messages, *args, **kwargs):
            self.assertLessEqual(agent.prompt_token_count(Tokenizer(), messages) + 1024,
                                 window)
            if responses is None:
                raise chat
            return next(responses)
        llm.chat.side_effect = checked_chat
        sampling = SimpleNamespace(max_tokens=1024, temperature=0.0)
        budgets = {'tool_calls': 5, 'time_minutes': 1, 'max_model_len': window,
                   'model_path': '/models/gemma-4-12b'}
        with patch.object(agent.ToolExecutor, 'extract_patch', return_value=''):
            summary = agent.run_agent_loop(root, {'problem_statement': '界' * 10000},
                                           llm, sampling, budgets, root, lambda _: None)
        return llm, summary

    def test_trace_summary_request_tokens_finish_reason_and_settings(self):
        def response(text, finish):
            return [SimpleNamespace(prompt_token_ids=[1] * 123,
                    outputs=[SimpleNamespace(text=text, token_ids=[2] * 7,
                                             finish_reason=finish)])]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm, summary = self.run_loop(root, [response('thinking', 'length'),
                response('TOOL_CALL: {"name":"submit_patch","arguments":{}}', 'stop')])
            self.assertEqual(summary['totals'], {'prompt_tokens': 246, 'output_tokens': 14})
            self.assertEqual(summary['model'], '/models/gemma-4-12b')
            self.assertEqual(summary['finish_reason'], 'stop')
            self.assertEqual(summary['settings']['max_tokens'], 1024)
            self.assertEqual(summary['settings']['max_model_len'], 6144)
            trace = [json.loads(line) for line in (root / 'trace.jsonl').read_text().splitlines()]
            self.assertEqual([row['prompt_tokens'] for row in trace], [123, 123])
            self.assertEqual([row['finish_reason'] for row in trace], ['length', 'stop'])
            self.assertEqual(summary['termination'], 'submit_patch')

    def test_context_errors_retry_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm, summary = self.run_loop(Path(tmp), RuntimeError('context length exceeded'))
            self.assertEqual(llm.chat.call_count, 2)
            self.assertEqual(summary['termination'], 'llm_error')
            calls = llm.chat.call_args_list
            self.assertLess(agent.prompt_token_count(Tokenizer(), calls[1].args[0]),
                            agent.prompt_token_count(Tokenizer(), calls[0].args[0]))

    def test_other_errors_do_not_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm, summary = self.run_loop(Path(tmp), RuntimeError('engine unavailable'))
            self.assertEqual(llm.chat.call_count, 1)
            self.assertEqual(summary['termination'], 'llm_error')


if __name__ == '__main__':
    unittest.main()
