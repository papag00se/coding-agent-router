from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.compaction.models import ChunkExtraction, SessionHandoff
from app.compaction.storage import CompactionStorage
from scripts import render_systemd_units


class TestStorageAndScripts(unittest.TestCase):
    def test_storage_returns_none_for_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CompactionStorage(Path(tmpdir))
            self.assertIsNone(storage.load_handoff("missing"))
            self.assertIsNone(storage.load_durable_memory("missing"))

            storage.save_chunk_extractions("session", [ChunkExtraction(chunk_id=1, objective="rename")])
            storage.save_handoff("session", SessionHandoff(stable_task_definition="rename"))
            self.assertEqual(storage.load_handoff("session").stable_task_definition, "rename")

    def test_write_units_writes_both_systemd_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with patch.object(render_systemd_units, "OUT_DIR", out_dir):
                render_systemd_units.write_units()
            self.assertTrue((out_dir / "coding-agent-router.service").exists())
            self.assertTrue((out_dir / "coding-agent-router-compaction.service").exists())
