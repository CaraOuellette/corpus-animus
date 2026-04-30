from __future__ import annotations

import argparse
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_SQLITE = ROOT / "wildchat_phase1_assistant_messages.sqlite"
DEFAULT_OUTPUT_SQLITE = ROOT / "wildchat_phase2_correction_frames.sqlite"
DEFAULT_OUTPUT_SUMMARY = ROOT / "wildchat_phase2_correction_frames_summary.json"
SCRIPT_VERSION = "0.1.0"
BATCH_SIZE = 5_000
CONTEXT_CHARS = 500

JUST = r"(?:just\s+)?"
SPAN_X = r".{1,240}?"
SPAN_X_CLAUSE = r"[^.!?]{1,240}?"
SPAN_Y = r".{1,300}?"
SEPARATOR = r"(?:\s*[,;:]\s*|\s*\.\s+|\s*[-\u2013\u2014]\s*)"
SEPARATOR_NO_PERIOD = r"(?:\s*[,;:]\s*|\s*[-\u2013\u2014]\s*)"
END_BOUNDARY = r"(?=$|[.!?](?:\s|$))"
CLAUSE_SUBJECT = r"(?:it|this|that|they|these|those)"
POSITIVE_AUX = r"(?:\s+(?:is|are|was|were)|['\u2019](?:s|re))"
NEGATIVE_AUX = (
    r"(?:\s+(?:is|are|was|were)\s+not|"
    r"['\u2019](?:s|re)\s+not|"
    r"\s+(?:isn['\u2019]t|aren['\u2019]t|wasn['\u2019]t|weren['\u2019]t))"
)
CLAUSE_START = rf"{CLAUSE_SUBJECT}{POSITIVE_AUX}"
NEGATIVE_CLAUSE_START = rf"{CLAUSE_SUBJECT}{NEGATIVE_AUX}"


@dataclass(frozen=True)
class InputMessage:
    assistant_message_id: int
    source_conversation_id: str
    conversation_message_index: int
    assistant_turn_number: int
    assistant_text_normalized: str
    preceding_user_text_normalized: str | None
    model: str | None
    conversation_language: str | None
    message_timestamp_utc: str | None


@dataclass(frozen=True)
class CorrectionPattern:
    pattern_id: str
    family: str
    confidence: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class CorrectionMatch:
    pattern_id: str
    family: str
    confidence: str
    match_text: str
    x_span: str | None
    y_span: str | None
    match_start: int
    match_end: int
    x_start: int | None
    x_end: int | None
    y_start: int | None
    y_end: int | None
    context_text: str


@dataclass(frozen=True)
class MarkerPattern:
    marker_id: str
    marker_text: str
    family: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class MarkerCount:
    marker_id: str
    marker_text: str
    family: str
    occurrence_count: int


@dataclass
class DetectionStats:
    messages_scanned: int = 0
    correction_instances: int = 0
    messages_with_correction: int = 0
    discourse_marker_occurrences: int = 0
    messages_with_discourse_marker: int = 0
    counts_by_pattern: Counter[str] = field(default_factory=Counter)
    correction_instances_by_model: Counter[str] = field(default_factory=Counter)
    correction_messages_by_model: Counter[str] = field(default_factory=Counter)
    marker_occurrences_by_marker: Counter[str] = field(default_factory=Counter)
    marker_occurrences_by_model: Counter[str] = field(default_factory=Counter)

    def update(
        self,
        message: InputMessage,
        correction_matches: Sequence[CorrectionMatch],
        marker_counts: Sequence[MarkerCount],
    ) -> None:
        model_key = message.model or "<empty>"
        self.messages_scanned += 1

        if correction_matches:
            self.messages_with_correction += 1
            self.correction_messages_by_model[model_key] += 1

        for match in correction_matches:
            self.correction_instances += 1
            self.counts_by_pattern[match.pattern_id] += 1
            self.correction_instances_by_model[model_key] += 1

        marker_total = sum(marker.occurrence_count for marker in marker_counts)
        if marker_total:
            self.messages_with_discourse_marker += 1
            self.discourse_marker_occurrences += marker_total
            self.marker_occurrences_by_model[model_key] += marker_total
            for marker in marker_counts:
                self.marker_occurrences_by_marker[marker.marker_id] += (
                    marker.occurrence_count
                )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "messages_scanned": self.messages_scanned,
            "correction_instances": self.correction_instances,
            "messages_with_correction": self.messages_with_correction,
            "correction_message_rate": safe_rate(
                self.messages_with_correction,
                self.messages_scanned,
            ),
            "correction_instance_rate": safe_rate(
                self.correction_instances,
                self.messages_scanned,
            ),
            "discourse_marker_occurrences": self.discourse_marker_occurrences,
            "messages_with_discourse_marker": self.messages_with_discourse_marker,
            "discourse_marker_message_rate": safe_rate(
                self.messages_with_discourse_marker,
                self.messages_scanned,
            ),
            "counts_by_pattern": dict(sorted(self.counts_by_pattern.items())),
            "correction_instances_by_model": dict(
                sorted(self.correction_instances_by_model.items())
            ),
            "correction_messages_by_model": dict(
                sorted(self.correction_messages_by_model.items())
            ),
            "marker_occurrences_by_marker": dict(
                sorted(self.marker_occurrences_by_marker.items())
            ),
            "marker_occurrences_by_model": dict(
                sorted(self.marker_occurrences_by_model.items())
            ),
        }


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE)


def normalize_for_detection(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2013": "-",
                "\u2014": "-",
                "\u2015": "-",
                "\u2212": "-",
            }
        )
    )
    normalized = normalized.replace("**", "").replace("__", "")
    normalized = normalized.replace("*", "").replace("_", "").replace("`", "")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or not unicodedata.category(character).startswith("C")
    )
    return re.sub(r"\s+", " ", normalized).strip()


def clean_span(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip(" \t\r\n,;:.-\u2013\u2014")
    return cleaned or None


def context_window(text: str, start: int, end: int, chars: int = CONTEXT_CHARS) -> str:
    window_start = max(0, start - chars)
    window_end = min(len(text), end + chars)
    prefix = "..." if window_start > 0 else ""
    suffix = "..." if window_end < len(text) else ""
    return prefix + text[window_start:window_end].strip() + suffix


def make_correction_patterns() -> list[CorrectionPattern]:
    definitions = [
        (
            "pronoun_not_pronoun",
            "explicit_negation_contrast",
            "high",
            rf"\b{NEGATIVE_CLAUSE_START}\s+{JUST}(?P<x>{SPAN_X_CLAUSE})"
            rf"{SEPARATOR}{CLAUSE_START}\s+{JUST}(?P<y>{SPAN_Y}){END_BOUNDARY}",
        ),
        (
            "not_because_but_because",
            "explicit_negation_contrast",
            "high",
            rf"\bnot\s+(?!only\b){JUST}because\s+(?P<x>{SPAN_X_CLAUSE})"
            rf"{SEPARATOR_NO_PERIOD}but\s+{JUST}because\s+(?P<y>{SPAN_Y}){END_BOUNDARY}",
        ),
        (
            "not_but_rather_instead",
            "explicit_negation_contrast",
            "high",
            rf"\bnot\s+(?!only\b){JUST}(?P<x>{SPAN_X_CLAUSE})"
            rf"{SEPARATOR_NO_PERIOD}(?:but|rather|instead)\s+{JUST}(?P<y>{SPAN_Y}){END_BOUNDARY}",
        ),
        (
            "x_not_case_rather_y",
            "explicit_negation_contrast",
            "high",
            rf"\b(?P<x>{SPAN_X_CLAUSE})\s+"
            rf"(?:is\s+not|isn['\u2019]t|are\s+not|aren['\u2019]t)\s+"
            rf"(?:really\s+)?(?:the\s+case|true|accurate)"
            rf"{SEPARATOR}(?:rather|instead|actually|in\s+reality)\s*,?\s*"
            rf"{JUST}(?P<y>{SPAN_Y}){END_BOUNDARY}",
        ),
        (
            "not_about_about",
            "explicit_negation_contrast",
            "high",
            rf"\b{NEGATIVE_CLAUSE_START}\s+{JUST}"
            rf"(?:about|a\s+matter\s+of|a\s+question\s+of)\s+(?P<x>{SPAN_X_CLAUSE})"
            rf"{SEPARATOR}{CLAUSE_START}\s+{JUST}"
            rf"(?:about|a\s+matter\s+of|a\s+question\s+of)\s+(?P<y>{SPAN_Y})"
            rf"{END_BOUNDARY}",
        ),
        (
            "does_not_does",
            "explicit_negation_contrast",
            "medium",
            rf"\b{CLAUSE_SUBJECT}\s+(?:does\s+not|doesn['\u2019]t)\s+"
            rf"{JUST}(?P<x>{SPAN_X_CLAUSE}){SEPARATOR_NO_PERIOD}{CLAUSE_SUBJECT}\s+"
            rf"{JUST}(?P<y>{SPAN_Y}){END_BOUNDARY}",
        ),
        (
            "less_about_more_about",
            "explicit_negation_contrast",
            "high",
            rf"\bless\s+about\s+{JUST}(?P<x>{SPAN_X_CLAUSE})\s+"
            rf"(?:and\s+)?more\s+about\s+{JUST}(?P<y>{SPAN_Y}){END_BOUNDARY}",
        ),
        (
            "rather_than_y",
            "explicit_negation_contrast",
            "medium",
            rf"\brather\s+than\s+{JUST}(?P<x>{SPAN_X_CLAUSE}){SEPARATOR_NO_PERIOD}"
            rf"{JUST}(?P<y>{SPAN_Y}){END_BOUNDARY}",
        ),
        (
            "instead_of_think_y",
            "explicit_negation_contrast",
            "high",
            rf"\binstead\s+of\s+{JUST}(?P<x>{SPAN_X_CLAUSE}){SEPARATOR_NO_PERIOD}"
            rf"(?:(?:think|thinking)\s+of\s+it\s+as|"
            rf"think\s+about\s+it\s+as|see\s+it\s+as|consider\s+it|focus\s+on)?"
            rf"\s*{JUST}(?P<y>{SPAN_Y}){END_BOUNDARY}",
        ),
        (
            "common_misconception_reality",
            "misconception_template",
            "high",
            rf"\b(?:a\s+)?common\s+(?:misconception|myth)\s+"
            rf"(?:is|that)\s+{JUST}(?P<x>{SPAN_X_CLAUSE}){SEPARATOR}"
            rf"(?:in\s+reality|actually|the\s+truth\s+is|the\s+reality\s+is)"
            rf"\s*,?\s*{JUST}(?P<y>{SPAN_Y}){END_BOUNDARY}",
        ),
        (
            "people_think_but",
            "misconception_template",
            "high",
            rf"\b(?:people\s+often\s+think|you\s+might\s+assume|"
            rf"it['\u2019]s\s+often\s+said\s+that|"
            rf"while\s+it['\u2019]s\s+tempting\s+to\s+think)\s+"
            rf"{JUST}(?P<x>{SPAN_X_CLAUSE}){SEPARATOR}"
            rf"(?:(?:but|however|actually|in\s+reality)\s*,?\s*)?"
            rf"{JUST}(?P<y>{SPAN_Y}){END_BOUNDARY}",
        ),
        (
            "contrary_to_belief",
            "misconception_template",
            "medium",
            rf"\bcontrary\s+to\s+(?:popular\s+belief|"
            rf"(?:a\s+)?common\s+assumption|what\s+you\s+might\s+think)"
            rf"\s*,?\s*{JUST}(?P<y>{SPAN_Y}){END_BOUNDARY}",
        ),
        (
            "clause_y_not_x",
            "reversed_correction",
            "medium",
            rf"\b(?P<y>{CLAUSE_START}\s+[^.!?]{{1,180}}?){SEPARATOR_NO_PERIOD}"
            rf"not\s+(?!only\b){JUST}(?P<x>{SPAN_X_CLAUSE}){END_BOUNDARY}",
        ),
    ]
    return [
        CorrectionPattern(
            pattern_id=pattern_id,
            family=family,
            confidence=confidence,
            regex=compile_pattern(pattern),
        )
        for pattern_id, family, confidence, pattern in definitions
    ]


def make_marker_patterns() -> list[MarkerPattern]:
    definitions = [
        ("actually", "actually", "discourse_marker", r"\bactually\b"),
        ("in_fact", "in fact", "discourse_marker", r"\bin\s+fact\b"),
        ("in_reality", "in reality", "discourse_marker", r"\bin\s+reality\b"),
        (
            "more_precisely",
            "more precisely",
            "precision_marker",
            r"\b(?:more\s+precisely|to\s+be\s+(?:more\s+)?precise)\b",
        ),
        (
            "strictly_speaking",
            "strictly speaking",
            "precision_marker",
            r"\bstrictly\s+speaking\b",
        ),
        (
            "more_accurately",
            "more accurately",
            "precision_marker",
            r"\b(?:more\s+accurately|to\s+be\s+more\s+accurate)\b",
        ),
        (
            "contrary_to",
            "contrary to",
            "misconception_marker",
            r"\bcontrary\s+to\s+(?:popular\s+belief|(?:a\s+)?common\s+assumption|what\s+you\s+might\s+think)\b",
        ),
        (
            "truth_reality_is",
            "the truth/reality is",
            "misconception_marker",
            r"\b(?:the\s+truth\s+is|the\s+reality\s+is)\b",
        ),
        (
            "sentence_initial_however",
            "however",
            "contrast_marker",
            r"(?:^|[.!?]\s+)however\b",
        ),
        (
            "on_the_contrary",
            "on the contrary",
            "contrast_marker",
            r"\bon\s+the\s+contrary\b",
        ),
        (
            "corrective_rather",
            "rather",
            "contrast_marker",
            r"(?:^|[,;:.!?]\s+)rather\b",
        ),
        (
            "really_happening",
            "what's really happening",
            "reality_marker",
            r"\bwhat['\u2019]s\s+really\s+happening\s+is\b",
        ),
        (
            "the_real_x_is",
            "the real X is",
            "reality_marker",
            r"\bthe\s+real\s+[a-z][a-z-]{0,40}\s+is\b",
        ),
    ]
    return [
        MarkerPattern(
            marker_id=marker_id,
            marker_text=marker_text,
            family=family,
            regex=compile_pattern(pattern),
        )
        for marker_id, marker_text, family, pattern in definitions
    ]


class CorrectionFrameDetector:
    def __init__(self, patterns: Sequence[CorrectionPattern] | None = None) -> None:
        self.patterns = list(patterns or make_correction_patterns())

    def find(self, text: str) -> list[CorrectionMatch]:
        detection_text = normalize_for_detection(text)
        matches: list[CorrectionMatch] = []
        occupied_spans: list[tuple[int, int]] = []

        for pattern in self.patterns:
            for regex_match in pattern.regex.finditer(detection_text):
                start, end = regex_match.span()
                if start == end or any(overlaps((start, end), span) for span in occupied_spans):
                    continue

                x_span = clean_span(regex_match.groupdict().get("x"))
                y_span = clean_span(regex_match.groupdict().get("y"))
                x_bounds = group_bounds(regex_match, "x")
                y_bounds = group_bounds(regex_match, "y")

                matches.append(
                    CorrectionMatch(
                        pattern_id=pattern.pattern_id,
                        family=pattern.family,
                        confidence=pattern.confidence,
                        match_text=regex_match.group(0).strip(),
                        x_span=x_span,
                        y_span=y_span,
                        match_start=start,
                        match_end=end,
                        x_start=x_bounds[0],
                        x_end=x_bounds[1],
                        y_start=y_bounds[0],
                        y_end=y_bounds[1],
                        context_text=context_window(detection_text, start, end),
                    )
                )
                occupied_spans.append((start, end))

        matches.sort(key=lambda item: (item.match_start, item.match_end))
        return matches


class DiscourseMarkerCounter:
    def __init__(self, patterns: Sequence[MarkerPattern] | None = None) -> None:
        self.patterns = list(patterns or make_marker_patterns())

    def count(self, text: str) -> list[MarkerCount]:
        detection_text = normalize_for_detection(text)
        counts: list[MarkerCount] = []
        for pattern in self.patterns:
            occurrence_count = len(list(pattern.regex.finditer(detection_text)))
            if occurrence_count:
                counts.append(
                    MarkerCount(
                        marker_id=pattern.marker_id,
                        marker_text=pattern.marker_text,
                        family=pattern.family,
                        occurrence_count=occurrence_count,
                    )
                )
        return counts


def overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def group_bounds(regex_match: re.Match[str], group_name: str) -> tuple[int | None, int | None]:
    try:
        start, end = regex_match.span(group_name)
    except IndexError:
        return None, None
    if start < 0 or end < 0:
        return None, None
    return start, end


class Phase1Reader:
    def __init__(self, input_sqlite: Path) -> None:
        self.input_sqlite = input_sqlite.resolve()
        if not self.input_sqlite.exists():
            raise FileNotFoundError(f"Input SQLite does not exist: {self.input_sqlite}")
        self.connection = sqlite3.connect(str(self.input_sqlite))

    def iter_messages(
        self,
        limit_messages: int | None = None,
        batch_size: int = BATCH_SIZE,
    ) -> Iterator[InputMessage]:
        sql = """
            SELECT
                id,
                source_conversation_id,
                conversation_message_index,
                assistant_turn_number,
                assistant_text_normalized,
                preceding_user_text_normalized,
                model,
                conversation_language,
                message_timestamp_utc
            FROM assistant_messages
            ORDER BY id
        """
        parameters: tuple[Any, ...] = ()
        if limit_messages is not None:
            sql += " LIMIT ?"
            parameters = (limit_messages,)

        cursor = self.connection.execute(sql, parameters)
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                yield InputMessage(
                    assistant_message_id=row[0],
                    source_conversation_id=row[1],
                    conversation_message_index=row[2],
                    assistant_turn_number=row[3],
                    assistant_text_normalized=row[4],
                    preceding_user_text_normalized=row[5],
                    model=row[6],
                    conversation_language=row[7],
                    message_timestamp_utc=row[8],
                )

    def close(self) -> None:
        self.connection.close()


class DetectionSQLiteWriter:
    def __init__(self, output_sqlite: Path, overwrite: bool = False) -> None:
        self.output_sqlite = output_sqlite.resolve()
        self.output_sqlite.parent.mkdir(parents=True, exist_ok=True)
        if self.output_sqlite.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output SQLite already exists: {self.output_sqlite}. Use --overwrite."
                )
            self.output_sqlite.unlink()

        self.connection = sqlite3.connect(str(self.output_sqlite))
        self.connection.execute("PRAGMA journal_mode = DELETE")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute("PRAGMA temp_store = MEMORY")

    def create_tables(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE correction_frame_instances (
                id INTEGER PRIMARY KEY,
                assistant_message_id INTEGER NOT NULL,
                source_conversation_id TEXT NOT NULL,
                conversation_message_index INTEGER NOT NULL,
                assistant_turn_number INTEGER NOT NULL,
                model TEXT,
                conversation_language TEXT,
                message_timestamp_utc TEXT,
                pattern_id TEXT NOT NULL,
                pattern_family TEXT NOT NULL,
                confidence TEXT NOT NULL,
                match_text TEXT NOT NULL,
                x_span TEXT,
                y_span TEXT,
                match_start INTEGER NOT NULL,
                match_end INTEGER NOT NULL,
                x_start INTEGER,
                x_end INTEGER,
                y_start INTEGER,
                y_end INTEGER,
                context_text TEXT NOT NULL,
                preceding_user_text_normalized TEXT
            );

            CREATE TABLE discourse_marker_counts (
                id INTEGER PRIMARY KEY,
                assistant_message_id INTEGER NOT NULL,
                model TEXT,
                conversation_language TEXT,
                marker_id TEXT NOT NULL,
                marker_text TEXT NOT NULL,
                marker_family TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL
            );

            CREATE TABLE message_detection_flags (
                assistant_message_id INTEGER PRIMARY KEY,
                model TEXT,
                conversation_language TEXT,
                has_correction_frame INTEGER NOT NULL,
                correction_instance_count INTEGER NOT NULL,
                has_discourse_marker INTEGER NOT NULL,
                discourse_marker_occurrence_count INTEGER NOT NULL
            );

            CREATE TABLE detection_summary (
                metric_key TEXT PRIMARY KEY,
                metric_value TEXT NOT NULL
            );

            CREATE TABLE run_metadata (
                run_id INTEGER PRIMARY KEY,
                script_name TEXT NOT NULL,
                script_version TEXT NOT NULL,
                input_source_path TEXT NOT NULL,
                input_source_kind TEXT NOT NULL,
                output_sqlite TEXT NOT NULL,
                output_summary TEXT NOT NULL,
                cli_args_json TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                completed_at_utc TEXT NOT NULL,
                messages_scanned INTEGER NOT NULL,
                correction_instances INTEGER NOT NULL,
                messages_with_correction INTEGER NOT NULL,
                discourse_marker_occurrences INTEGER NOT NULL,
                messages_with_discourse_marker INTEGER NOT NULL
            );
            """
        )
        self.connection.commit()

    def insert_correction_matches(
        self,
        message: InputMessage,
        matches: Sequence[CorrectionMatch],
    ) -> None:
        if not matches:
            return
        if not self.connection.in_transaction:
            self.connection.execute("BEGIN TRANSACTION")
        self.connection.executemany(
            """
            INSERT INTO correction_frame_instances (
                assistant_message_id,
                source_conversation_id,
                conversation_message_index,
                assistant_turn_number,
                model,
                conversation_language,
                message_timestamp_utc,
                pattern_id,
                pattern_family,
                confidence,
                match_text,
                x_span,
                y_span,
                match_start,
                match_end,
                x_start,
                x_end,
                y_start,
                y_end,
                context_text,
                preceding_user_text_normalized
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    message.assistant_message_id,
                    message.source_conversation_id,
                    message.conversation_message_index,
                    message.assistant_turn_number,
                    message.model,
                    message.conversation_language,
                    message.message_timestamp_utc,
                    match.pattern_id,
                    match.family,
                    match.confidence,
                    match.match_text,
                    match.x_span,
                    match.y_span,
                    match.match_start,
                    match.match_end,
                    match.x_start,
                    match.x_end,
                    match.y_start,
                    match.y_end,
                    match.context_text,
                    message.preceding_user_text_normalized,
                )
                for match in matches
            ],
        )

    def insert_marker_counts(
        self,
        message: InputMessage,
        marker_counts: Sequence[MarkerCount],
    ) -> None:
        if not marker_counts:
            return
        if not self.connection.in_transaction:
            self.connection.execute("BEGIN TRANSACTION")
        self.connection.executemany(
            """
            INSERT INTO discourse_marker_counts (
                assistant_message_id,
                model,
                conversation_language,
                marker_id,
                marker_text,
                marker_family,
                occurrence_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    message.assistant_message_id,
                    message.model,
                    message.conversation_language,
                    marker.marker_id,
                    marker.marker_text,
                    marker.family,
                    marker.occurrence_count,
                )
                for marker in marker_counts
            ],
        )

    def insert_message_flags(
        self,
        message: InputMessage,
        matches: Sequence[CorrectionMatch],
        marker_counts: Sequence[MarkerCount],
    ) -> None:
        marker_total = sum(marker.occurrence_count for marker in marker_counts)
        if not matches and not marker_total:
            return
        if not self.connection.in_transaction:
            self.connection.execute("BEGIN TRANSACTION")
        self.connection.execute(
            """
            INSERT INTO message_detection_flags (
                assistant_message_id,
                model,
                conversation_language,
                has_correction_frame,
                correction_instance_count,
                has_discourse_marker,
                discourse_marker_occurrence_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.assistant_message_id,
                message.model,
                message.conversation_language,
                int(bool(matches)),
                len(matches),
                int(bool(marker_total)),
                marker_total,
            ),
        )

    def flush(self) -> None:
        self.connection.commit()

    def create_indexes(self) -> None:
        self.connection.executescript(
            """
            CREATE INDEX idx_correction_instances_message_id
                ON correction_frame_instances (assistant_message_id);
            CREATE INDEX idx_correction_instances_model
                ON correction_frame_instances (model);
            CREATE INDEX idx_correction_instances_pattern
                ON correction_frame_instances (pattern_id);
            CREATE INDEX idx_marker_counts_message_id
                ON discourse_marker_counts (assistant_message_id);
            CREATE INDEX idx_marker_counts_model
                ON discourse_marker_counts (model);
            CREATE INDEX idx_marker_counts_marker
                ON discourse_marker_counts (marker_id);
            CREATE INDEX idx_message_flags_model
                ON message_detection_flags (model);
            """
        )
        self.connection.commit()

    def write_summary(self, stats: DetectionStats) -> None:
        rows = [
            (key, json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value))
            for key, value in stats.to_json_dict().items()
        ]
        self.connection.executemany(
            "INSERT INTO detection_summary (metric_key, metric_value) VALUES (?, ?)",
            rows,
        )
        self.connection.commit()

    def write_run_metadata(self, metadata: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO run_metadata (
                script_name,
                script_version,
                input_source_path,
                input_source_kind,
                output_sqlite,
                output_summary,
                cli_args_json,
                started_at_utc,
                completed_at_utc,
                messages_scanned,
                correction_instances,
                messages_with_correction,
                discourse_marker_occurrences,
                messages_with_discourse_marker
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata["script_name"],
                metadata["script_version"],
                metadata["input_source_path"],
                metadata["input_source_kind"],
                metadata["output_sqlite"],
                metadata["output_summary"],
                metadata["cli_args_json"],
                metadata["started_at_utc"],
                metadata["completed_at_utc"],
                metadata["messages_scanned"],
                metadata["correction_instances"],
                metadata["messages_with_correction"],
                metadata["discourse_marker_occurrences"],
                metadata["messages_with_discourse_marker"],
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def iter_batches(items: Iterable[InputMessage], batch_size: int) -> Iterator[list[InputMessage]]:
    batch: list[InputMessage] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_summary_artifact(output_summary: Path, stats: DetectionStats) -> None:
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with output_summary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats.to_json_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def prepare_output_paths(
    output_sqlite: Path,
    output_summary: Path,
    overwrite: bool,
) -> tuple[Path, Path]:
    output_sqlite = output_sqlite.resolve()
    output_summary = output_summary.resolve()
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    if output_summary.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output summary already exists: {output_summary}. Use --overwrite."
            )
        output_summary.unlink()

    return output_sqlite, output_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 correction-frame detection and Phase 3 marker counts."
    )
    parser.add_argument(
        "--input-sqlite",
        type=Path,
        default=DEFAULT_INPUT_SQLITE,
        help="Path to the Phase 1 assistant-message SQLite database.",
    )
    parser.add_argument(
        "--output-sqlite",
        type=Path,
        default=DEFAULT_OUTPUT_SQLITE,
        help="Path to the Phase 2/3 output SQLite database.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
        help="Path to the JSON summary artifact.",
    )
    parser.add_argument(
        "--limit-messages",
        type=int,
        default=None,
        help="Optional number of assistant messages to scan.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of messages to process between commits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output artifacts.",
    )
    return parser.parse_args(argv)


def run_detection(args: argparse.Namespace) -> DetectionStats:
    output_sqlite, output_summary = prepare_output_paths(
        args.output_sqlite,
        args.output_summary,
        args.overwrite,
    )
    reader = Phase1Reader(args.input_sqlite)
    writer = DetectionSQLiteWriter(output_sqlite, overwrite=args.overwrite)
    detector = CorrectionFrameDetector()
    marker_counter = DiscourseMarkerCounter()
    started_at_utc = utc_now_iso()
    stats = DetectionStats()

    writer.create_tables()
    try:
        messages = reader.iter_messages(
            limit_messages=args.limit_messages,
            batch_size=args.batch_size,
        )
        for batch in iter_batches(messages, args.batch_size):
            for message in batch:
                matches = detector.find(message.assistant_text_normalized)
                marker_counts = marker_counter.count(message.assistant_text_normalized)
                writer.insert_correction_matches(message, matches)
                writer.insert_marker_counts(message, marker_counts)
                writer.insert_message_flags(message, matches, marker_counts)
                stats.update(message, matches, marker_counts)

            writer.flush()
            if stats.messages_scanned % 50_000 == 0:
                print(
                    "Scanned "
                    f"{stats.messages_scanned:,} messages, "
                    f"found {stats.correction_instances:,} correction instances, "
                    f"{stats.discourse_marker_occurrences:,} marker occurrences."
                )

        writer.create_indexes()
        writer.write_summary(stats)
        completed_at_utc = utc_now_iso()
        writer.write_run_metadata(
            {
                "script_name": Path(__file__).name,
                "script_version": SCRIPT_VERSION,
                "input_source_path": str(Path(args.input_sqlite).resolve()),
                "input_source_kind": "sqlite/assistant_messages",
                "output_sqlite": str(output_sqlite),
                "output_summary": str(output_summary),
                "cli_args_json": json.dumps(
                    {
                        "input_sqlite": str(args.input_sqlite),
                        "output_sqlite": str(args.output_sqlite),
                        "output_summary": str(args.output_summary),
                        "limit_messages": args.limit_messages,
                        "batch_size": args.batch_size,
                        "overwrite": args.overwrite,
                    },
                    sort_keys=True,
                ),
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
                "messages_scanned": stats.messages_scanned,
                "correction_instances": stats.correction_instances,
                "messages_with_correction": stats.messages_with_correction,
                "discourse_marker_occurrences": stats.discourse_marker_occurrences,
                "messages_with_discourse_marker": stats.messages_with_discourse_marker,
            }
        )
        write_summary_artifact(output_summary, stats)
    finally:
        reader.close()
        writer.close()

    return stats


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stats = run_detection(args)
    print(
        "Completed Phase 2/3 detection: "
        f"{stats.messages_scanned:,} messages scanned, "
        f"{stats.correction_instances:,} correction instances, "
        f"{stats.discourse_marker_occurrences:,} marker occurrences."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
