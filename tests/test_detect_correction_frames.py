from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

import detect_correction_frames as phase23


TEST_OUTPUT_DIR = phase23.ROOT / "_test_outputs"


class CorrectionFrameDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = phase23.CorrectionFrameDetector()

    def assert_single_match(
        self,
        text: str,
        expected_x: str | None,
        expected_y: str | None,
    ) -> phase23.CorrectionMatch:
        matches = self.detector.find(text)
        self.assertEqual(len(matches), 1, matches)
        match = matches[0]
        self.assertEqual(match.x_span, expected_x)
        self.assertEqual(match.y_span, expected_y)
        return match

    def test_its_not_x_its_y(self) -> None:
        match = self.assert_single_match(
            "It's not a liquid, it's a gas.",
            "a liquid",
            "a gas",
        )
        self.assertEqual(match.pattern_id, "pronoun_not_pronoun")

    def test_its_not_just_x_its_y(self) -> None:
        self.assert_single_match(
            "It's not just speed, it's reliability.",
            "speed",
            "reliability",
        )

    def test_its_not_x_its_just_y(self) -> None:
        self.assert_single_match(
            "It's not luck, it's just probability.",
            "luck",
            "probability",
        )

    def test_period_separated_thats_not_x_thats_y(self) -> None:
        self.assert_single_match(
            "That's not a bug. That's a feature.",
            "a bug",
            "a feature",
        )

    def test_em_dash_without_spaces(self) -> None:
        self.assert_single_match(
            "It's not a shortcut\u2014it's a workflow.",
            "a shortcut",
            "a workflow",
        )

    def test_em_dash_with_spaces(self) -> None:
        self.assert_single_match(
            "That's not failure \u2014 that's feedback.",
            "failure",
            "feedback",
        )

    def test_not_because_but_because_allows_just(self) -> None:
        match = self.assert_single_match(
            "Not just because users are lazy, but because the interface is unclear.",
            "users are lazy",
            "the interface is unclear",
        )
        self.assertEqual(match.pattern_id, "not_because_but_because")

    def test_less_about_more_about_allows_just(self) -> None:
        match = self.assert_single_match(
            "It's less about just raw talent and more about consistent practice.",
            "raw talent",
            "consistent practice",
        )
        self.assertEqual(match.pattern_id, "less_about_more_about")

    def test_multiple_non_overlapping_matches(self) -> None:
        matches = self.detector.find(
            "It's not a liquid, it's a gas. That's not a bug. That's a feature."
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual([match.x_span for match in matches], ["a liquid", "a bug"])

    def test_not_only_but_is_not_tier1_correction(self) -> None:
        matches = self.detector.find(
            "Not only does Alex learn quickly, but he also teaches others."
        )
        self.assertEqual(matches, [])

    def test_not_but_does_not_cross_unrelated_sentence_boundary(self) -> None:
        matches = self.detector.find(
            "It was not easy. Gendry: But now, we're here together."
        )
        self.assertEqual(matches, [])

    def test_generic_not_but_does_not_match_period_separator(self) -> None:
        matches = self.detector.find(
            "It's not having enough money. But hey, you can still try."
        )
        self.assertEqual(matches, [])


class DiscourseMarkerCounterTests(unittest.TestCase):
    def test_counts_markers(self) -> None:
        counter = phase23.DiscourseMarkerCounter()
        counts = {
            marker.marker_id: marker.occurrence_count
            for marker in counter.count(
                "Actually, this is common. In fact, strictly speaking, "
                "the real issue is trust. However, there is more."
            )
        }

        self.assertEqual(counts["actually"], 1)
        self.assertEqual(counts["in_fact"], 1)
        self.assertEqual(counts["strictly_speaking"], 1)
        self.assertEqual(counts["the_real_x_is"], 1)
        self.assertEqual(counts["sentence_initial_however"], 1)


class DetectionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_OUTPUT_DIR.mkdir(exist_ok=True)
        self.input_sqlite = TEST_OUTPUT_DIR / "phase23_input.sqlite"
        self.output_sqlite = TEST_OUTPUT_DIR / "phase23_output.sqlite"
        self.output_summary = TEST_OUTPUT_DIR / "phase23_summary.json"
        for path in (self.input_sqlite, self.output_sqlite, self.output_summary):
            if path.exists():
                try:
                    path.unlink()
                except PermissionError:
                    pass

    def tearDown(self) -> None:
        for path in (self.input_sqlite, self.output_sqlite, self.output_summary):
            if path.exists():
                try:
                    path.unlink()
                except PermissionError:
                    pass

    def create_input_db(self) -> None:
        connection = sqlite3.connect(self.input_sqlite)
        try:
            connection.executescript(
                """
                CREATE TABLE assistant_messages (
                    id INTEGER PRIMARY KEY,
                    source_conversation_id TEXT NOT NULL,
                    conversation_message_index INTEGER NOT NULL,
                    assistant_turn_number INTEGER NOT NULL,
                    assistant_text_normalized TEXT NOT NULL,
                    preceding_user_text_normalized TEXT,
                    model TEXT,
                    conversation_language TEXT,
                    message_timestamp_utc TEXT
                );
                """
            )
            connection.executemany(
                """
                INSERT INTO assistant_messages (
                    id,
                    source_conversation_id,
                    conversation_message_index,
                    assistant_turn_number,
                    assistant_text_normalized,
                    preceding_user_text_normalized,
                    model,
                    conversation_language,
                    message_timestamp_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        1,
                        "conv-a",
                        1,
                        1,
                        "It's not a liquid, it's a gas. Actually, this matters.",
                        "What is steam?",
                        "gpt-4-0125-preview",
                        "English",
                        "2024-01-01T00:00:00+00:00",
                    ),
                    (
                        2,
                        "conv-b",
                        1,
                        1,
                        "This is a plain answer with no special markers.",
                        "Say hello",
                        "gpt-3.5-turbo-0613",
                        "English",
                        "2024-01-01T00:01:00+00:00",
                    ),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def test_cli_writes_detection_outputs(self) -> None:
        self.create_input_db()

        exit_code = phase23.main(
            [
                "--input-sqlite",
                str(self.input_sqlite),
                "--output-sqlite",
                str(self.output_sqlite),
                "--output-summary",
                str(self.output_summary),
                "--overwrite",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(self.output_sqlite.exists())
        self.assertTrue(self.output_summary.exists())

        connection = sqlite3.connect(self.output_sqlite)
        try:
            correction_count = connection.execute(
                "SELECT COUNT(*) FROM correction_frame_instances"
            ).fetchone()[0]
            marker_count = connection.execute(
                "SELECT SUM(occurrence_count) FROM discourse_marker_counts"
            ).fetchone()[0]
            flag_count = connection.execute(
                "SELECT COUNT(*) FROM message_detection_flags"
            ).fetchone()[0]
            pattern_id = connection.execute(
                "SELECT pattern_id FROM correction_frame_instances"
            ).fetchone()[0]
            metadata_count = connection.execute(
                "SELECT COUNT(*) FROM run_metadata"
            ).fetchone()[0]
        finally:
            connection.close()

        self.assertEqual(correction_count, 1)
        self.assertEqual(marker_count, 1)
        self.assertEqual(flag_count, 1)
        self.assertEqual(pattern_id, "pronoun_not_pronoun")
        self.assertEqual(metadata_count, 1)

        summary = json.loads(self.output_summary.read_text(encoding="utf-8"))
        self.assertEqual(summary["messages_scanned"], 2)
        self.assertEqual(summary["correction_instances"], 1)
        self.assertEqual(summary["discourse_marker_occurrences"], 1)


@unittest.skipUnless(
    (phase23.ROOT / "wildchat_phase1_assistant_messages.sqlite").exists(),
    "Phase 1 SQLite artifact is required for bounded real-data detection.",
)
class BoundedRealDataDetectionTests(unittest.TestCase):
    def test_bounded_real_data_detection_run(self) -> None:
        output_sqlite = TEST_OUTPUT_DIR / "phase23_real_output.sqlite"
        output_summary = TEST_OUTPUT_DIR / "phase23_real_summary.json"
        for path in (output_sqlite, output_summary):
            if path.exists():
                try:
                    path.unlink()
                except PermissionError:
                    pass

        try:
            exit_code = phase23.main(
                [
                    "--input-sqlite",
                    str(phase23.ROOT / "wildchat_phase1_assistant_messages.sqlite"),
                    "--output-sqlite",
                    str(output_sqlite),
                    "--output-summary",
                    str(output_summary),
                    "--limit-messages",
                    "1000",
                    "--overwrite",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads(output_summary.read_text(encoding="utf-8"))
            self.assertEqual(summary["messages_scanned"], 1000)

            connection = sqlite3.connect(output_sqlite)
            try:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                indexes = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index'"
                    )
                }
            finally:
                connection.close()

            self.assertIn("correction_frame_instances", tables)
            self.assertIn("discourse_marker_counts", tables)
            self.assertIn("message_detection_flags", tables)
            self.assertIn("run_metadata", tables)
            self.assertIn("idx_correction_instances_pattern", indexes)
            self.assertIn("idx_marker_counts_marker", indexes)
        finally:
            for path in (output_sqlite, output_summary):
                if path.exists():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass


if __name__ == "__main__":
    unittest.main()
