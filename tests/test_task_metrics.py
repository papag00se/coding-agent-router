from __future__ import annotations

import unittest

from app import task_metrics
from app.task_metrics import extract_task_metrics


class TestTaskMetricsExtractor(unittest.TestCase):
    def test_metrics_are_quantified_counts(self) -> None:
        result = extract_task_metrics(
            prompt="Please update `src/utils.py` and explain why it failed?",
            trajectory=[
                {"role": "user", "content": "Open `src/utils.py`"},
                {"role": "assistant", "content": "```bash\nrg utils\n```"},
                {"role": "tool", "content": "Traceback (most recent call last):\nFile \"service.py\", line 12\nTypeError: boom"},
            ],
            metadata={"request_id": "abc"},
        )

        self.assertEqual(result["message_count"], 3)
        self.assertEqual(result["user_message_count"], 1)
        self.assertEqual(result["assistant_message_count"], 1)
        self.assertEqual(result["tool_message_count"], 1)
        self.assertGreaterEqual(result["file_reference_count"], 2)
        self.assertEqual(result["unique_file_reference_count"], 2)
        self.assertEqual(result["question_count"], 1)
        self.assertGreater(result["stack_trace_count"], 0)
        self.assertGreater(result["error_line_count"], 0)
        self.assertEqual(result["metadata_key_count"], 1)
        self.assertNotIn("derived_metrics", result)
        self.assertNotIn("raw_metrics", result)

    def test_prior_failure_count_comes_from_attempt_history(self) -> None:
        result = extract_task_metrics(
            prompt="Fix auth.py",
            trajectory={
                "attempts": [
                    {"status": "ok"},
                    {"status": "failed"},
                    {"status": "timeout"},
                    {"status": "low_confidence"},
                ]
            },
            metadata={},
        )

        self.assertEqual(result["prior_failure_count"], 3)
        self.assertEqual(result["message_count"], 0)

    def test_command_output_tokens_count_stdout_and_stderr(self) -> None:
        result = extract_task_metrics(
            prompt="Analyze this run",
            trajectory={
                "attempts": [
                    {"status": "failed", "stdout": "line one\nline two", "stderr": "error: boom"}
                ]
            },
            metadata={},
        )

        self.assertGreater(result["command_output_tokens"], 0)
        self.assertGreater(result["error_line_count"], 0)

    def test_private_helpers_cover_type_error_empty_lines_and_dict_shapes(self) -> None:
        class Unserializable:
            def __str__(self):
                return "fallback"

        self.assertEqual(task_metrics._stringify(None), "")
        self.assertEqual(task_metrics._stringify(Unserializable()), "fallback")
        self.assertEqual(task_metrics._line_count(""), 0)
        self.assertEqual(task_metrics._message_like_items({"messages": [{"role": "user", "content": "hi"}, {"content": "skip"}]}), [{"role": "user", "content": "hi"}])
        self.assertEqual(task_metrics._attempt_items({"attempts": "bad"}), [])
