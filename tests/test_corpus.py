from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import corpus


class CorpusTests(unittest.TestCase):
    def test_iter_messages_reads_claude_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "conversations.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "uuid": "conv-1",
                            "chat_messages": [
                                {
                                    "sender": "human",
                                    "text": "Hello",
                                    "created_at": "2024-01-02T00:00:00Z",
                                },
                                {
                                    "sender": "assistant",
                                    "text": "*smiles*\nHi there",
                                    "created_at": "2024-01-02T00:00:01Z",
                                },
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            rows = list(corpus.iter_messages(input_path))
            self.assertEqual(
                rows,
                [
                    ("2024-01", "human", "Hello", "conv-1"),
                    ("2024-01", "assistant", "*smiles*\nHi there", "conv-1"),
                ],
            )

    def test_iter_messages_reads_chatgpt_html_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "chat.html"
            conversations = [
                {
                    "conversation_id": "conv-2",
                    "current_node": "assistant-1",
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
                                "metadata": {"is_user_system_message": False},
                            },
                        },
                        "user-1": {
                            "id": "user-1",
                            "parent": "system-1",
                            "children": ["assistant-1"],
                            "message": {
                                "id": "user-1",
                                "author": {"role": "user"},
                                "content": {
                                    "content_type": "text",
                                    "parts": ["Hello there"],
                                },
                                "create_time": 1700000000,
                                "metadata": {},
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
                                    "parts": ["*waves*\nHi"],
                                },
                                "create_time": 1700000001,
                                "metadata": {},
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

            rows = list(corpus.iter_messages(input_path))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][1], "user")
            self.assertEqual(rows[0][2], "Hello there")
            self.assertEqual(rows[1][1], "assistant")
            self.assertEqual(rows[1][2], "*waves*\nHi")
            self.assertEqual(rows[1][3], "conv-2")

    def test_iter_messages_reads_wildchat_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "wildchat.sqlite"
            connection = sqlite3.connect(str(input_path))
            try:
                connection.executescript(
                    """
                    CREATE TABLE assistant_messages (
                        id INTEGER PRIMARY KEY,
                        source_conversation_id TEXT,
                        message_timestamp_utc TEXT,
                        conversation_timestamp_utc TEXT,
                        assistant_text_raw TEXT
                    );
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO assistant_messages (
                        source_conversation_id,
                        message_timestamp_utc,
                        conversation_timestamp_utc,
                        assistant_text_raw
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            "conv-3",
                            "2024-02-03T01:02:03+00:00",
                            "2024-02-03T01:00:00+00:00",
                            "*smiles*\nHello",
                        ),
                        (
                            "conv-4",
                            None,
                            "2024-03-04T00:00:00+00:00",
                            "Second row",
                        ),
                    ],
                )
                connection.commit()
            finally:
                connection.close()

            rows = list(corpus.iter_messages(input_path))
            self.assertEqual(
                rows,
                [
                    ("2024-02", "assistant", "*smiles*\nHello", "conv-3"),
                    ("2024-03", "assistant", "Second row", "conv-4"),
                ],
            )
