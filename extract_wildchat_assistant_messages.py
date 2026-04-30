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

import pyarrow as pa
import pyarrow.ipc as ipc


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = ROOT / "WildChat-1M"
DEFAULT_OUTPUT_SQLITE = ROOT / "wildchat_phase1_assistant_messages.sqlite"
DEFAULT_OUTPUT_PROFILE = ROOT / "wildchat_phase1_assistant_messages_profile.json"
SCRIPT_VERSION = "0.1.0"
BATCH_SIZE = 5_000
PROFILE_EMPTY_VALUE = "<empty>"

SMART_PUNCTUATION_TRANSLATION = str.maketrans(
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

ASSISTANT_MESSAGE_COLUMNS = [
    "source_dataset",
    "source_conversation_id",
    "conversation_message_index",
    "assistant_turn_number",
    "assistant_text_raw",
    "assistant_text_normalized",
    "preceding_user_text_raw",
    "preceding_user_text_normalized",
    "preceding_user_timestamp_utc",
    "system_prompt_text",
    "model",
    "conversation_timestamp_utc",
    "conversation_language",
    "conversation_country",
    "conversation_state",
    "conversation_redacted",
    "conversation_toxic",
    "message_timestamp_utc",
    "message_language",
    "message_country",
    "message_state",
    "message_redacted",
    "message_toxic",
    "message_turn_identifier",
]


@dataclass(frozen=True)
class AssistantMessageRecord:
    source_dataset: str
    source_conversation_id: str
    conversation_message_index: int
    assistant_turn_number: int
    assistant_text_raw: str
    assistant_text_normalized: str
    preceding_user_text_raw: str | None
    preceding_user_text_normalized: str | None
    preceding_user_timestamp_utc: str | None
    system_prompt_text: str | None
    model: str | None
    conversation_timestamp_utc: str | None
    conversation_language: str | None
    conversation_country: str | None
    conversation_state: str | None
    conversation_redacted: int | None
    conversation_toxic: int | None
    message_timestamp_utc: str | None
    message_language: str | None
    message_country: str | None
    message_state: str | None
    message_redacted: int | None
    message_toxic: int | None
    message_turn_identifier: int | None

    def as_tuple(self) -> tuple[Any, ...]:
        return tuple(getattr(self, column) for column in ASSISTANT_MESSAGE_COLUMNS)


@dataclass
class ConversationExtraction:
    records: list[AssistantMessageRecord]
    assistant_messages_observed: int = 0
    assistant_messages_skipped_empty: int = 0
    user_messages_observed: int = 0
    system_messages_observed: int = 0


@dataclass
class ProfileStats:
    conversations_processed: int = 0
    assistant_messages_emitted: int = 0
    assistant_messages_observed: int = 0
    assistant_messages_skipped_empty: int = 0
    user_messages_observed: int = 0
    system_messages_observed: int = 0
    rows_with_preceding_user_context: int = 0
    counts_by_model: Counter[str] = field(default_factory=Counter)
    counts_by_conversation_language: Counter[str] = field(default_factory=Counter)

    def update(self, extraction: ConversationExtraction) -> None:
        self.conversations_processed += 1
        self.assistant_messages_observed += extraction.assistant_messages_observed
        self.assistant_messages_skipped_empty += (
            extraction.assistant_messages_skipped_empty
        )
        self.user_messages_observed += extraction.user_messages_observed
        self.system_messages_observed += extraction.system_messages_observed

        for record in extraction.records:
            self.assistant_messages_emitted += 1
            if record.preceding_user_text_raw is not None:
                self.rows_with_preceding_user_context += 1
            self.counts_by_model[record.model or PROFILE_EMPTY_VALUE] += 1
            self.counts_by_conversation_language[
                record.conversation_language or PROFILE_EMPTY_VALUE
            ] += 1

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "conversations_processed": self.conversations_processed,
            "assistant_messages_emitted": self.assistant_messages_emitted,
            "assistant_messages_observed": self.assistant_messages_observed,
            "assistant_messages_skipped_empty": self.assistant_messages_skipped_empty,
            "user_messages_observed": self.user_messages_observed,
            "system_messages_observed": self.system_messages_observed,
            "rows_with_preceding_user_context": self.rows_with_preceding_user_context,
            "counts_by_model": dict(sorted(self.counts_by_model.items())),
            "counts_by_conversation_language": dict(
                sorted(self.counts_by_conversation_language.items())
            ),
        }

    def dataset_profile_rows(self) -> list[tuple[str, str]]:
        rows = [
            ("conversations_processed", str(self.conversations_processed)),
            ("assistant_messages_emitted", str(self.assistant_messages_emitted)),
            ("assistant_messages_observed", str(self.assistant_messages_observed)),
            (
                "assistant_messages_skipped_empty",
                str(self.assistant_messages_skipped_empty),
            ),
            ("user_messages_observed", str(self.user_messages_observed)),
            ("system_messages_observed", str(self.system_messages_observed)),
            (
                "rows_with_preceding_user_context",
                str(self.rows_with_preceding_user_context),
            ),
        ]
        rows.extend(
            (f"count_by_model:{key}", str(value))
            for key, value in sorted(self.counts_by_model.items())
        )
        rows.extend(
            (f"count_by_conversation_language:{key}", str(value))
            for key, value in sorted(self.counts_by_conversation_language.items())
        )
        return rows


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def format_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat()
    return str(value)


def bool_to_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def resolve_train_dir(input_root: Path) -> Path:
    input_root = input_root.resolve()
    if input_root.is_dir() and list(input_root.glob("data-*.arrow")):
        return input_root

    train_dir = input_root / "train"
    if train_dir.is_dir() and list(train_dir.glob("data-*.arrow")):
        return train_dir

    raise FileNotFoundError(
        f"Could not find WildChat Arrow shards under {input_root} or {train_dir}."
    )


def iter_arrow_rows(files: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in sorted(files):
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                for row in batch.to_pylist():
                    yield row


class WildChatReader:
    def __init__(self, input_root: Path) -> None:
        self.input_root = input_root.resolve()
        self.train_dir = resolve_train_dir(self.input_root)

    def iter_conversations(
        self, limit_conversations: int | None = None
    ) -> Iterator[dict[str, Any]]:
        for index, row in enumerate(iter_arrow_rows(self.train_dir.glob("data-*.arrow"))):
            if limit_conversations is not None and index >= limit_conversations:
                break
            yield row


class MessageNormalizer:
    def normalize(self, text: str | None) -> str | None:
        if text is None:
            return None

        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.translate(SMART_PUNCTUATION_TRANSLATION)
        normalized = "".join(
            character
            for character in normalized
            if self._keep_character(character)
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _keep_character(character: str) -> bool:
        if character in {"\n", "\t"}:
            return True
        if unicodedata.category(character).startswith("C"):
            return False
        return True


class SQLiteWriter:
    def __init__(self, output_path: Path, overwrite: bool = False) -> None:
        self.output_path = output_path.resolve()
        self.overwrite = overwrite
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output SQLite already exists: {self.output_path}. Use --overwrite."
                )
            self.output_path.unlink()

        self.connection = sqlite3.connect(str(self.output_path))
        self.connection.execute("PRAGMA journal_mode = DELETE")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute("PRAGMA temp_store = MEMORY")

    def create_tables(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE assistant_messages (
                id INTEGER PRIMARY KEY,
                source_dataset TEXT NOT NULL,
                source_conversation_id TEXT NOT NULL,
                conversation_message_index INTEGER NOT NULL,
                assistant_turn_number INTEGER NOT NULL,
                assistant_text_raw TEXT NOT NULL,
                assistant_text_normalized TEXT NOT NULL,
                preceding_user_text_raw TEXT,
                preceding_user_text_normalized TEXT,
                preceding_user_timestamp_utc TEXT,
                system_prompt_text TEXT,
                model TEXT,
                conversation_timestamp_utc TEXT,
                conversation_language TEXT,
                conversation_country TEXT,
                conversation_state TEXT,
                conversation_redacted INTEGER,
                conversation_toxic INTEGER,
                message_timestamp_utc TEXT,
                message_language TEXT,
                message_country TEXT,
                message_state TEXT,
                message_redacted INTEGER,
                message_toxic INTEGER,
                message_turn_identifier INTEGER
            );

            CREATE TABLE run_metadata (
                run_id INTEGER PRIMARY KEY,
                script_name TEXT NOT NULL,
                script_version TEXT NOT NULL,
                input_root TEXT NOT NULL,
                input_train_dir TEXT NOT NULL,
                output_sqlite TEXT NOT NULL,
                output_profile TEXT NOT NULL,
                cli_args_json TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                completed_at_utc TEXT NOT NULL,
                conversations_processed INTEGER NOT NULL,
                assistant_messages_emitted INTEGER NOT NULL,
                assistant_messages_observed INTEGER NOT NULL,
                assistant_messages_skipped_empty INTEGER NOT NULL,
                user_messages_observed INTEGER NOT NULL,
                system_messages_observed INTEGER NOT NULL,
                rows_with_preceding_user_context INTEGER NOT NULL
            );

            CREATE TABLE dataset_profile (
                metric_key TEXT PRIMARY KEY,
                metric_value TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def insert_records(self, records: Sequence[AssistantMessageRecord]) -> None:
        if not records:
            return

        placeholders = ", ".join("?" for _ in ASSISTANT_MESSAGE_COLUMNS)
        columns = ", ".join(ASSISTANT_MESSAGE_COLUMNS)
        sql = (
            f"INSERT INTO assistant_messages ({columns}) "
            f"VALUES ({placeholders})"
        )
        if not self.connection.in_transaction:
            self.connection.execute("BEGIN TRANSACTION")
        self.connection.executemany(sql, [record.as_tuple() for record in records])

    def flush(self) -> None:
        self.connection.commit()

    def create_indexes(self) -> None:
        self.connection.executescript(
            """
            CREATE INDEX idx_assistant_messages_source_conversation_id
                ON assistant_messages (source_conversation_id);
            CREATE INDEX idx_assistant_messages_model
                ON assistant_messages (model);
            CREATE INDEX idx_assistant_messages_assistant_turn_number
                ON assistant_messages (assistant_turn_number);
            CREATE INDEX idx_assistant_messages_message_timestamp_utc
                ON assistant_messages (message_timestamp_utc);
            CREATE INDEX idx_assistant_messages_conversation_language
                ON assistant_messages (conversation_language);
            """
        )
        self.connection.commit()

    def write_dataset_profile(self, stats: ProfileStats) -> None:
        self.connection.executemany(
            "INSERT INTO dataset_profile (metric_key, metric_value) VALUES (?, ?)",
            stats.dataset_profile_rows(),
        )
        self.connection.commit()

    def write_run_metadata(self, metadata: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO run_metadata (
                script_name,
                script_version,
                input_root,
                input_train_dir,
                output_sqlite,
                output_profile,
                cli_args_json,
                started_at_utc,
                completed_at_utc,
                conversations_processed,
                assistant_messages_emitted,
                assistant_messages_observed,
                assistant_messages_skipped_empty,
                user_messages_observed,
                system_messages_observed,
                rows_with_preceding_user_context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata["script_name"],
                metadata["script_version"],
                metadata["input_root"],
                metadata["input_train_dir"],
                metadata["output_sqlite"],
                metadata["output_profile"],
                metadata["cli_args_json"],
                metadata["started_at_utc"],
                metadata["completed_at_utc"],
                metadata["conversations_processed"],
                metadata["assistant_messages_emitted"],
                metadata["assistant_messages_observed"],
                metadata["assistant_messages_skipped_empty"],
                metadata["user_messages_observed"],
                metadata["system_messages_observed"],
                metadata["rows_with_preceding_user_context"],
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def extract_assistant_messages(
    row: dict[str, Any],
    normalizer: MessageNormalizer,
    max_records: int | None = None,
) -> ConversationExtraction:
    conversation = row.get("conversation") or []
    records: list[AssistantMessageRecord] = []
    prior_system_messages: list[str] = []
    previous_user_message: dict[str, Any] | None = None
    assistant_turn_number = 0
    assistant_messages_observed = 0
    assistant_messages_skipped_empty = 0
    user_messages_observed = 0
    system_messages_observed = 0

    for index, message in enumerate(conversation):
        role = str(message.get("role") or "").lower()
        content = coerce_text(message.get("content"))

        if role == "system":
            system_messages_observed += 1
            if content is not None:
                prior_system_messages.append(content)
            continue

        if role == "user":
            user_messages_observed += 1
            previous_user_message = message
            continue

        if role != "assistant":
            continue

        assistant_messages_observed += 1
        assistant_turn_number += 1
        if content is None or not content.strip():
            assistant_messages_skipped_empty += 1
            continue

        preceding_user_text_raw = (
            coerce_text(previous_user_message.get("content"))
            if previous_user_message is not None
            else None
        )
        record = AssistantMessageRecord(
            source_dataset="WildChat",
            source_conversation_id=str(row.get("conversation_hash") or ""),
            conversation_message_index=index,
            assistant_turn_number=assistant_turn_number,
            assistant_text_raw=content,
            assistant_text_normalized=normalizer.normalize(content) or "",
            preceding_user_text_raw=preceding_user_text_raw,
            preceding_user_text_normalized=normalizer.normalize(preceding_user_text_raw),
            preceding_user_timestamp_utc=(
                format_timestamp(previous_user_message.get("timestamp"))
                if previous_user_message is not None
                else None
            ),
            system_prompt_text=(
                "\n\n".join(prior_system_messages) if prior_system_messages else None
            ),
            model=coerce_text(row.get("model")),
            conversation_timestamp_utc=format_timestamp(row.get("timestamp")),
            conversation_language=coerce_text(row.get("language")),
            conversation_country=coerce_text(row.get("country")),
            conversation_state=coerce_text(row.get("state")),
            conversation_redacted=bool_to_int(row.get("redacted")),
            conversation_toxic=bool_to_int(row.get("toxic")),
            message_timestamp_utc=format_timestamp(message.get("timestamp")),
            message_language=coerce_text(message.get("language")),
            message_country=coerce_text(message.get("country")),
            message_state=coerce_text(message.get("state")),
            message_redacted=bool_to_int(message.get("redacted")),
            message_toxic=bool_to_int(message.get("toxic")),
            message_turn_identifier=message.get("turn_identifier"),
        )
        records.append(record)

        if max_records is not None and len(records) >= max_records:
            break

    return ConversationExtraction(
        records=records,
        assistant_messages_observed=assistant_messages_observed,
        assistant_messages_skipped_empty=assistant_messages_skipped_empty,
        user_messages_observed=user_messages_observed,
        system_messages_observed=system_messages_observed,
    )


def write_profile_artifact(output_path: Path, stats: ProfileStats) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats.to_json_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def prepare_output_paths(
    output_sqlite: Path, output_profile: Path, overwrite: bool
) -> tuple[Path, Path]:
    output_sqlite = output_sqlite.resolve()
    output_profile = output_profile.resolve()
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    output_profile.parent.mkdir(parents=True, exist_ok=True)

    if output_profile.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output profile already exists: {output_profile}. Use --overwrite."
            )
        output_profile.unlink()

    return output_sqlite, output_profile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one normalized row per WildChat assistant message into SQLite."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Path to the WildChat dataset root or its train directory.",
    )
    parser.add_argument(
        "--output-sqlite",
        type=Path,
        default=DEFAULT_OUTPUT_SQLITE,
        help="Path to the output SQLite database.",
    )
    parser.add_argument(
        "--output-profile",
        type=Path,
        default=DEFAULT_OUTPUT_PROFILE,
        help="Path to the output JSON profile artifact.",
    )
    parser.add_argument(
        "--limit-conversations",
        type=int,
        default=None,
        help="Optional limit for the number of conversations to process.",
    )
    parser.add_argument(
        "--limit-messages",
        type=int,
        default=None,
        help="Optional limit for the number of assistant rows to emit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args(argv)


def run_extraction(args: argparse.Namespace) -> ProfileStats:
    output_sqlite, output_profile = prepare_output_paths(
        args.output_sqlite, args.output_profile, args.overwrite
    )
    reader = WildChatReader(args.input_root)
    normalizer = MessageNormalizer()
    writer = SQLiteWriter(output_sqlite, overwrite=args.overwrite)
    writer.create_tables()

    started_at_utc = utc_now_iso()
    stats = ProfileStats()
    batch: list[AssistantMessageRecord] = []

    try:
        for conversation_index, row in enumerate(
            reader.iter_conversations(limit_conversations=args.limit_conversations),
            start=1,
        ):
            remaining_messages = None
            if args.limit_messages is not None:
                remaining_messages = args.limit_messages - stats.assistant_messages_emitted
                if remaining_messages <= 0:
                    break

            extraction = extract_assistant_messages(
                row,
                normalizer,
                max_records=remaining_messages,
            )
            stats.update(extraction)
            batch.extend(extraction.records)

            if len(batch) >= BATCH_SIZE:
                writer.insert_records(batch)
                writer.flush()
                batch.clear()

            if conversation_index % 25_000 == 0:
                print(
                    "Processed "
                    f"{conversation_index:,} conversations, "
                    f"emitted {stats.assistant_messages_emitted:,} assistant rows."
                )

            if (
                args.limit_messages is not None
                and stats.assistant_messages_emitted >= args.limit_messages
            ):
                break

        if batch:
            writer.insert_records(batch)
            writer.flush()

        writer.create_indexes()
        writer.write_dataset_profile(stats)

        completed_at_utc = utc_now_iso()
        writer.write_run_metadata(
            {
                "script_name": Path(__file__).name,
                "script_version": SCRIPT_VERSION,
                "input_root": str(Path(args.input_root).resolve()),
                "input_train_dir": str(reader.train_dir),
                "output_sqlite": str(output_sqlite),
                "output_profile": str(output_profile),
                "cli_args_json": json.dumps(
                    {
                        "input_root": str(args.input_root),
                        "output_sqlite": str(args.output_sqlite),
                        "output_profile": str(args.output_profile),
                        "limit_conversations": args.limit_conversations,
                        "limit_messages": args.limit_messages,
                        "overwrite": args.overwrite,
                    },
                    sort_keys=True,
                ),
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
                "conversations_processed": stats.conversations_processed,
                "assistant_messages_emitted": stats.assistant_messages_emitted,
                "assistant_messages_observed": stats.assistant_messages_observed,
                "assistant_messages_skipped_empty": (
                    stats.assistant_messages_skipped_empty
                ),
                "user_messages_observed": stats.user_messages_observed,
                "system_messages_observed": stats.system_messages_observed,
                "rows_with_preceding_user_context": stats.rows_with_preceding_user_context,
            }
        )
        write_profile_artifact(output_profile, stats)
    finally:
        writer.close()

    return stats


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stats = run_extraction(args)
    print(
        "Completed WildChat Phase 1 extraction: "
        f"{stats.conversations_processed:,} conversations, "
        f"{stats.assistant_messages_emitted:,} assistant rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
