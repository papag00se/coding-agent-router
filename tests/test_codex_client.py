from __future__ import annotations

import unittest
from unittest.mock import patch

from app.clients.codex_client import CodexCLIClient


class TestCodexCLIClient(unittest.TestCase):
    def test_exec_prompt_forces_non_router_provider_and_model(self):
        captured = {}

        def fake_run(cmd, cwd, capture_output, text, timeout, check):
            captured['cmd'] = cmd
            captured['cwd'] = cwd

            class Result:
                returncode = 0
                stdout = 'OK'
                stderr = ''

            return Result()

        client = CodexCLIClient(
            'codex',
            '.',
            30,
            model_provider='openai',
            model='gpt-5.4',
        )

        with patch('app.clients.codex_client.subprocess.run', side_effect=fake_run):
            result = client.exec_prompt('Reply with exactly OK.', workdir='/tmp/repo')

        self.assertEqual(result['stdout'], 'OK')
        self.assertEqual(captured['cwd'], '/tmp/repo')
        self.assertEqual(
            captured['cmd'],
            [
                'codex',
                'exec',
                '--skip-git-repo-check',
                '-c',
                'model_provider=openai',
                '-c',
                'model=gpt-5.4',
                '-C',
                '/tmp/repo',
                'Reply with exactly OK.',
            ],
        )
