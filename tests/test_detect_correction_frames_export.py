from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import detect_correction_frames_export as export_detector


class ExportDetectorTests(unittest.TestCase):
    def test_detects_matches_in_claude_export_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "conversations.json"
            output_sqlite = tmp_path / "claude.sqlite"
            output_summary = tmp_path / "claude_summary.json"

            input_path.write_text(
                json.dumps(
                    [
                        {
                            "uuid": "conv-claude-1",
                            "chat_messages": [
                                {
                                    "uuid": "user-1",
                                    "sender": "human",
                                    "text": "Explain steam.",
                                    "created_at": "2024-01-01T00:00:00Z",
                                },
                                {
                                    "uuid": "assistant-1",
                                    "sender": "assistant",
                                    "text": "It's not just hot water, it's water vapor.",
                                    "created_at": "2024-01-01T00:00:05Z",
                                },
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            exit_code = export_detector.main(
                [
                    "--input-path",
                    str(input_path),
                    "--output-sqlite",
                    str(output_sqlite),
                    "--output-summary",
                    str(output_summary),
                    "--overwrite",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_sqlite.exists())
            self.assertTrue(output_summary.exists())

            connection = sqlite3.connect(output_sqlite)
            try:
                pattern_id = connection.execute(
                    "SELECT pattern_id FROM correction_frame_instances"
                ).fetchone()[0]
                message_count = connection.execute(
                    "SELECT messages_scanned FROM run_metadata"
                ).fetchone()[0]
                source_kind = connection.execute(
                    "SELECT input_source_kind FROM run_metadata"
                ).fetchone()[0]
            finally:
                connection.close()

            self.assertEqual(pattern_id, "pronoun_not_pronoun")
            self.assertEqual(message_count, 1)
            self.assertEqual(source_kind, "claude-json")

    def test_detects_matches_in_chatgpt_html_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "chat.html"
            output_sqlite = tmp_path / "chat.sqlite"
            output_summary = tmp_path / "chat_summary.json"

            conversations = [
                {
                    "conversation_id": "conv-chatgpt-1",
                    "current_node": "assistant-1",
                    "default_model_slug": "gpt-4",
                    "mapping": {
                        "root": {
                            "id": "root",
                            "parent": None,
                            "children": ["system-1"],
                            "message": None,
                        },
                        "system-1": {
                            "id": "system-1",
                            "parent": "root",
                            "children": ["user-1"],
                            "message": {
                                "id": "system-1",
                                "author": {"role": "system"},
                                "content": {"content_type": "text", "parts": [""]},
                                "metadata": {
                                    "is_user_system_message": False,
                                },
                            },
                        },
                        "user-1": {
                            "id": "user-1",
                            "parent": "system-1",
                            "children": ["assistant-1", "assistant-interrupted"],
                            "message": {
                                "id": "user-1",
                                "author": {"role": "user"},
                                "content": {
                                    "content_type": "text",
                                    "parts": ["How should I think about this?"],
                                },
                                "create_time": 1700000000,
                                "metadata": {},
                            },
                        },
                        "assistant-interrupted": {
                            "id": "assistant-interrupted",
                            "parent": "user-1",
                            "children": [],
                            "message": {
                                "id": "assistant-interrupted",
                                "author": {"role": "assistant"},
                                "content": {"content_type": "text", "parts": [""]},
                                "create_time": 1700000001,
                                "metadata": {"model_slug": "gpt-4"},
                            },
                        },
                        "assistant-1": {
                            "id": "assistant-1",
                            "parent": "user-1",
                            "children": [],
                            "message": {
                                "id": "assistant-1",
                                "author": {"role": "assistant"},
                                "content": {
                                    "content_type": "text",
                                    "parts": [
                                        "It's not a failure, it's feedback. Actually, it's useful."
                                    ],
                                },
                                "create_time": 1700000002,
                                "metadata": {"model_slug": "gpt-4"},
                            },
                        },
                    },
                }
            ]
            input_path.write_text(
                "<html><body><script>var jsonData = "
                + json.dumps(conversations)
                + ";</script></body></html>",
                encoding="utf-8",
            )

            exit_code = export_detector.main(
                [
                    "--input-path",
                    str(input_path),
                    "--output-sqlite",
                    str(output_sqlite),
                    "--output-summary",
                    str(output_summary),
                    "--overwrite",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_sqlite.exists())
            self.assertTrue(output_summary.exists())

            connection = sqlite3.connect(output_sqlite)
            try:
                row = connection.execute(
                    "SELECT pattern_id, model, preceding_user_text_normalized "
                    "FROM correction_frame_instances"
                ).fetchone()
                source_kind = connection.execute(
                    "SELECT input_source_kind FROM run_metadata"
                ).fetchone()[0]
                marker_count = connection.execute(
                    "SELECT SUM(occurrence_count) FROM discourse_marker_counts"
                ).fetchone()[0]
            finally:
                connection.close()

            self.assertEqual(row[0], "pronoun_not_pronoun")
            self.assertEqual(row[1], "gpt-4")
            self.assertEqual(row[2], "How should I think about this?")
            self.assertEqual(source_kind, "chatgpt-html")
            self.assertEqual(marker_count, 1)

