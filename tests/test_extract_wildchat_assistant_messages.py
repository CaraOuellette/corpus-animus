from __future__ import annotations

import json
import os
import sqlite3
import unittest
from datetime import datetime, timezone
from pathlib import Path

import extract_wildchat_assistant_messages as phase1


TEST_OUTPUT_DIR = phase1.ROOT / "_test_outputs"


def utc_datetime(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


class MessageNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = phase1.MessageNormalizer()

    def test_normalize_collapses_whitespace(self) -> None:
        raw = "  hello\t\tworld\r\n\r\nnext line  "
        self.assertEqual(self.normalizer.normalize(raw), "hello world next line")

    def test_normalize_folds_quotes_and_dashes(self) -> None:
        raw = "“quoted” – and — more"
        self.assertEqual(self.normalizer.normalize(raw), '"quoted" - and - more')

    def test_normalize_removes_zero_width_controls_and_raw_is_unchanged(self) -> None:
        raw = "A\u200bB\x00C"
        normalized = self.normalizer.normalize(raw)
        self.assertEqual(normalized, "ABC")
        self.assertEqual(raw, "A\u200bB\x00C")


class ExtractionUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = phase1.MessageNormalizer()

    def test_single_turn_user_assistant(self) -> None:
        row = {
            "conversation_hash": "conv-1",
            "model": "gpt-4-0314",
            "timestamp": utc_datetime(2024, 1, 1, 12, 0, 0),
            "language": "English",
            "country": "United States",
            "state": "NY",
            "redacted": False,
            "toxic": False,
            "conversation": [
                {
                    "role": "user",
                    "content": "Question",
                    "timestamp": utc_datetime(2024, 1, 1, 12, 0, 1),
                },
                {
                    "role": "assistant",
                    "content": "Answer",
                    "timestamp": utc_datetime(2024, 1, 1, 12, 0, 2),
                    "language": "English",
                    "country": "United States",
                    "state": "NY",
                    "redacted": False,
                    "toxic": False,
                    "turn_identifier": 7,
                },
            ],
        }

        extraction = phase1.extract_assistant_messages(row, self.normalizer)

        self.assertEqual(extraction.user_messages_observed, 1)
        self.assertEqual(extraction.assistant_messages_observed, 1)
        self.assertEqual(extraction.assistant_messages_skipped_empty, 0)
        self.assertEqual(extraction.system_messages_observed, 0)
        self.assertEqual(len(extraction.records), 1)

        record = extraction.records[0]
        self.assertEqual(record.source_conversation_id, "conv-1")
        self.assertEqual(record.conversation_message_index, 1)
        self.assertEqual(record.assistant_turn_number, 1)
        self.assertEqual(record.assistant_text_raw, "Answer")
        self.assertEqual(record.preceding_user_text_raw, "Question")
        self.assertEqual(
            record.preceding_user_timestamp_utc,
            "2024-01-01T12:00:01+00:00",
        )
        self.assertIsNone(record.system_prompt_text)

    def test_multi_turn_extraction_with_system_prompt(self) -> None:
        row = {
            "conversation_hash": "conv-2",
            "model": "gpt-4-0125-preview",
            "timestamp": utc_datetime(2024, 2, 1, 8, 0, 0),
            "language": "French",
            "country": "France",
            "state": None,
            "redacted": True,
            "toxic": False,
            "conversation": [
                {
                    "role": "system",
                    "content": "You are helpful.",
                    "timestamp": utc_datetime(2024, 2, 1, 8, 0, 0),
                },
                {
                    "role": "system",
                    "content": "Answer briefly.",
                    "timestamp": utc_datetime(2024, 2, 1, 8, 0, 0),
                },
                {
                    "role": "user",
                    "content": "First question",
                    "timestamp": utc_datetime(2024, 2, 1, 8, 0, 1),
                },
                {
                    "role": "assistant",
                    "content": "First answer",
                    "timestamp": utc_datetime(2024, 2, 1, 8, 0, 2),
                },
                {
                    "role": "user",
                    "content": "Second question",
                    "timestamp": utc_datetime(2024, 2, 1, 8, 0, 3),
                },
                {
                    "role": "assistant",
                    "content": "Second answer",
                    "timestamp": utc_datetime(2024, 2, 1, 8, 0, 4),
                },
            ],
        }

        extraction = phase1.extract_assistant_messages(row, self.normalizer)

        self.assertEqual(extraction.user_messages_observed, 2)
        self.assertEqual(extraction.assistant_messages_observed, 2)
        self.assertEqual(extraction.assistant_messages_skipped_empty, 0)
        self.assertEqual(extraction.system_messages_observed, 2)
        self.assertEqual(len(extraction.records), 2)

        first, second = extraction.records
        self.assertEqual(
            first.system_prompt_text,
            "You are helpful.\n\nAnswer briefly.",
        )
        self.assertEqual(first.preceding_user_text_raw, "First question")
        self.assertEqual(second.preceding_user_text_raw, "Second question")
        self.assertEqual(second.assistant_turn_number, 2)
        self.assertEqual(second.conversation_redacted, 1)
        self.assertEqual(second.conversation_toxic, 0)

    def test_assistant_without_prior_user_sets_null_context(self) -> None:
        row = {
            "conversation_hash": "conv-3",
            "conversation": [
                {
                    "role": "assistant",
                    "content": "Opening answer",
                    "timestamp": utc_datetime(2024, 3, 1, 9, 0, 0),
                }
            ],
        }

        extraction = phase1.extract_assistant_messages(row, self.normalizer)

        self.assertEqual(len(extraction.records), 1)
        self.assertEqual(extraction.assistant_messages_observed, 1)
        self.assertEqual(extraction.assistant_messages_skipped_empty, 0)
        record = extraction.records[0]
        self.assertIsNone(record.preceding_user_text_raw)
        self.assertIsNone(record.preceding_user_text_normalized)
        self.assertIsNone(record.preceding_user_timestamp_utc)

    def test_blank_assistant_content_is_skipped(self) -> None:
        row = {
            "conversation_hash": "conv-4",
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "   \n\t  "},
                {"role": "assistant", "content": "Real answer"},
            ],
        }

        extraction = phase1.extract_assistant_messages(row, self.normalizer)

        self.assertEqual(len(extraction.records), 1)
        self.assertEqual(extraction.assistant_messages_observed, 2)
        self.assertEqual(extraction.assistant_messages_skipped_empty, 1)
        self.assertEqual(extraction.records[0].assistant_text_raw, "Real answer")
        self.assertEqual(extraction.records[0].assistant_turn_number, 2)


@unittest.skipUnless(
    (phase1.ROOT / "WildChat-1M").exists(),
    "WildChat dataset is required for integration tests.",
)
class SQLiteIntegrationTests(unittest.TestCase):
    def test_bounded_real_data_run_creates_expected_schema_and_counts(self) -> None:
        dataset_root = phase1.ROOT / "WildChat-1M"
        normalizer = phase1.MessageNormalizer()
        expected_rows = 0
        reader = phase1.WildChatReader(dataset_root)
        for row in reader.iter_conversations(limit_conversations=100):
            expected_rows += len(
                phase1.extract_assistant_messages(row, normalizer).records
            )

        TEST_OUTPUT_DIR.mkdir(exist_ok=True)
        sqlite_path = TEST_OUTPUT_DIR / "phase1_integration.sqlite"
        profile_path = TEST_OUTPUT_DIR / "phase1_integration_profile.json"

        try:
            exit_code = phase1.main(
                [
                    "--input-root",
                    str(dataset_root),
                    "--output-sqlite",
                    str(sqlite_path),
                    "--output-profile",
                    str(profile_path),
                    "--limit-conversations",
                    "100",
                    "--overwrite",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(sqlite_path.exists())
            self.assertTrue(profile_path.exists())

            with sqlite3.connect(sqlite_path) as connection:
                row_count = connection.execute(
                    "SELECT COUNT(*) FROM assistant_messages"
                ).fetchone()[0]
                self.assertEqual(row_count, expected_rows)

                distinct_conversations = connection.execute(
                    "SELECT COUNT(DISTINCT source_conversation_id) "
                    "FROM assistant_messages"
                ).fetchone()[0]
                self.assertGreater(distinct_conversations, 0)
                self.assertLessEqual(distinct_conversations, 100)

                non_null_preceding = connection.execute(
                    "SELECT COUNT(*) FROM assistant_messages "
                    "WHERE preceding_user_text_raw IS NOT NULL"
                ).fetchone()[0]
                self.assertGreater(non_null_preceding, 0)

                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                self.assertIn("assistant_messages", tables)
                self.assertIn("run_metadata", tables)
                self.assertIn("dataset_profile", tables)

                indexes = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index'"
                    )
                }
                self.assertIn(
                    "idx_assistant_messages_source_conversation_id",
                    indexes,
                )
                self.assertIn("idx_assistant_messages_model", indexes)
                self.assertIn(
                    "idx_assistant_messages_assistant_turn_number",
                    indexes,
                )
                self.assertIn(
                    "idx_assistant_messages_message_timestamp_utc",
                    indexes,
                )
                self.assertIn(
                    "idx_assistant_messages_conversation_language",
                    indexes,
                )

            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertEqual(profile["conversations_processed"], 100)
            self.assertEqual(profile["assistant_messages_emitted"], expected_rows)
            self.assertGreaterEqual(
                profile["assistant_messages_observed"],
                profile["assistant_messages_emitted"],
            )
            self.assertGreater(profile["rows_with_preceding_user_context"], 0)
            self.assertTrue(profile["counts_by_model"])
        finally:
            for path in (sqlite_path, profile_path):
                if path.exists():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass


@unittest.skipUnless(
    os.environ.get("RUN_FULL_WILDCHAT_PHASE1") == "1"
    and (phase1.ROOT / "WildChat-1M").exists(),
    "Set RUN_FULL_WILDCHAT_PHASE1=1 to run the full-dataset acceptance test.",
)
class FullDatasetAcceptanceTests(unittest.TestCase):
    def test_full_dataset_counts_match_current_local_copy(self) -> None:
        dataset_root = phase1.ROOT / "WildChat-1M"
        TEST_OUTPUT_DIR.mkdir(exist_ok=True)
        sqlite_path = TEST_OUTPUT_DIR / "phase1_full.sqlite"
        profile_path = TEST_OUTPUT_DIR / "phase1_full_profile.json"

        try:
            exit_code = phase1.main(
                [
                    "--input-root",
                    str(dataset_root),
                    "--output-sqlite",
                    str(sqlite_path),
                    "--output-profile",
                    str(profile_path),
                    "--overwrite",
                ]
            )

            self.assertEqual(exit_code, 0)

            with sqlite3.connect(sqlite_path) as connection:
                row_count = connection.execute(
                    "SELECT COUNT(*) FROM assistant_messages"
                ).fetchone()[0]
                non_null_system_prompts = connection.execute(
                    "SELECT COUNT(*) FROM assistant_messages "
                    "WHERE system_prompt_text IS NOT NULL"
                ).fetchone()[0]

            profile = json.loads(profile_path.read_text(encoding="utf-8"))

            self.assertEqual(profile["conversations_processed"], 837_989)
            self.assertEqual(profile["assistant_messages_observed"], 1_960_074)
            self.assertEqual(profile["assistant_messages_emitted"], 1_960_073)
            self.assertEqual(profile["assistant_messages_skipped_empty"], 1)
            self.assertEqual(profile["system_messages_observed"], 0)
            self.assertEqual(row_count, 1_960_073)
            self.assertEqual(non_null_system_prompts, 0)
        finally:
            for path in (sqlite_path, profile_path):
                if path.exists():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass


if __name__ == "__main__":
    unittest.main()
