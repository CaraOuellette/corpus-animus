from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from extract_wildchat_assistant_messages import (
    AssistantMessageRecord,
    ConversationExtraction,
    MessageNormalizer,
    ProfileStats,
    SQLiteWriter,
    coerce_text,
    prepare_output_paths,
    utc_now_iso,
    write_profile_artifact,
)
from prepare_chat_dataset_samples import iter_arrow_rows, iter_sharegpt_objects


ROOT = Path(__file__).resolve().parent
SHAREGPT_PATH = (
    ROOT
    / "ShareGPT_Vicuna_unfiltered_raw"
    / "ShareGPT_V3_unfiltered_cleaned_split.json"
)
ULTRACHAT_ROOT = ROOT / "UltraChat"
SCRIPT_VERSION = "0.1.0"
BATCH_SIZE = 5_000

DEFAULT_OUTPUTS = {
    "sharegpt": (
        ROOT / "sharegpt_phase1_assistant_messages.sqlite",
        ROOT / "sharegpt_phase1_assistant_messages_profile.json",
    ),
    "ultrachat": (
        ROOT / "ultrachat_phase1_assistant_messages.sqlite",
        ROOT / "ultrachat_phase1_assistant_messages_profile.json",
    ),
}

SHAREGPT_USER_ROLES = {"human", "user"}
SHAREGPT_ASSISTANT_ROLES = {"gpt", "chatgpt", "bing", "bard"}


def sharegpt_message_text(message: dict[str, Any] | None) -> str | None:
    if not message:
        return None

    for key in ("value", "text"):
        value = coerce_text(message.get(key))
        if value is not None:
            return value

    markdown = message.get("markdown")
    if isinstance(markdown, dict):
        value = coerce_text(markdown.get("answer"))
        if value is not None:
            return value

    return None


class ShareGPTReader:
    def __init__(self, input_path: Path) -> None:
        self.input_path = input_path.resolve()
        if not self.input_path.exists():
            raise FileNotFoundError(f"ShareGPT input file does not exist: {self.input_path}")

    def iter_conversations(
        self,
        limit_conversations: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        for index, row in enumerate(iter_sharegpt_objects(self.input_path)):
            if limit_conversations is not None and index >= limit_conversations:
                break
            yield row


def resolve_ultrachat_train_dir(input_root: Path) -> Path:
    input_root = input_root.resolve()
    if input_root.is_dir() and list(input_root.glob("data-*.arrow")):
        return input_root

    train_dir = input_root / "train"
    if train_dir.is_dir() and list(train_dir.glob("data-*.arrow")):
        return train_dir

    raise FileNotFoundError(
        f"Could not find UltraChat Arrow shards under {input_root} or {train_dir}."
    )


class UltraChatReader:
    def __init__(self, input_root: Path) -> None:
        self.input_root = input_root.resolve()
        self.train_dir = resolve_ultrachat_train_dir(self.input_root)

    def iter_conversations(
        self,
        limit_conversations: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        for index, row in enumerate(iter_arrow_rows(self.train_dir.glob("data-*.arrow"))):
            if limit_conversations is not None and index >= limit_conversations:
                break
            yield row


def extract_sharegpt_assistant_messages(
    row: dict[str, Any],
    normalizer: MessageNormalizer,
    max_records: int | None = None,
) -> ConversationExtraction:
    conversation = row.get("conversations") or []
    records: list[AssistantMessageRecord] = []
    prior_system_messages: list[str] = []
    previous_user_message: dict[str, Any] | None = None
    assistant_turn_number = 0
    assistant_messages_observed = 0
    assistant_messages_skipped_empty = 0
    user_messages_observed = 0
    system_messages_observed = 0

    for index, message in enumerate(conversation):
        role = str(message.get("from") or "").lower()
        content = sharegpt_message_text(message)

        if role == "system":
            system_messages_observed += 1
            if content is not None:
                prior_system_messages.append(content)
            continue

        if role in SHAREGPT_USER_ROLES:
            user_messages_observed += 1
            previous_user_message = message
            continue

        if role not in SHAREGPT_ASSISTANT_ROLES:
            continue

        assistant_messages_observed += 1
        assistant_turn_number += 1
        if content is None or not content.strip():
            assistant_messages_skipped_empty += 1
            continue

        preceding_user_text_raw = sharegpt_message_text(previous_user_message)
        record = AssistantMessageRecord(
            source_dataset="ShareGPT",
            source_conversation_id=str(row.get("id") or ""),
            conversation_message_index=index,
            assistant_turn_number=assistant_turn_number,
            assistant_text_raw=content,
            assistant_text_normalized=normalizer.normalize(content) or "",
            preceding_user_text_raw=preceding_user_text_raw,
            preceding_user_text_normalized=normalizer.normalize(preceding_user_text_raw),
            preceding_user_timestamp_utc=None,
            system_prompt_text=(
                "\n\n".join(prior_system_messages) if prior_system_messages else None
            ),
            model=None,
            conversation_timestamp_utc=None,
            conversation_language=None,
            conversation_country=None,
            conversation_state=None,
            conversation_redacted=None,
            conversation_toxic=None,
            message_timestamp_utc=None,
            message_language=None,
            message_country=None,
            message_state=None,
            message_redacted=None,
            message_toxic=None,
            message_turn_identifier=index,
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


def extract_ultrachat_assistant_messages(
    row: dict[str, Any],
    normalizer: MessageNormalizer,
    max_records: int | None = None,
) -> ConversationExtraction:
    turns = row.get("data") or []
    records: list[AssistantMessageRecord] = []
    previous_user_text_raw: str | None = None
    assistant_turn_number = 0
    assistant_messages_observed = 0
    assistant_messages_skipped_empty = 0
    user_messages_observed = 0

    for index, turn in enumerate(turns):
        content = coerce_text(turn)

        if index % 2 == 0:
            user_messages_observed += 1
            if content is not None and content.strip():
                previous_user_text_raw = content
            else:
                previous_user_text_raw = None
            continue

        assistant_messages_observed += 1
        assistant_turn_number += 1
        if content is None or not content.strip():
            assistant_messages_skipped_empty += 1
            continue

        record = AssistantMessageRecord(
            source_dataset="UltraChat",
            source_conversation_id=str(row.get("id") or ""),
            conversation_message_index=index,
            assistant_turn_number=assistant_turn_number,
            assistant_text_raw=content,
            assistant_text_normalized=normalizer.normalize(content) or "",
            preceding_user_text_raw=previous_user_text_raw,
            preceding_user_text_normalized=normalizer.normalize(previous_user_text_raw),
            preceding_user_timestamp_utc=None,
            system_prompt_text=None,
            model=None,
            conversation_timestamp_utc=None,
            conversation_language=None,
            conversation_country=None,
            conversation_state=None,
            conversation_redacted=None,
            conversation_toxic=None,
            message_timestamp_utc=None,
            message_language=None,
            message_country=None,
            message_state=None,
            message_redacted=None,
            message_toxic=None,
            message_turn_identifier=index,
        )
        records.append(record)

        if max_records is not None and len(records) >= max_records:
            break

    return ConversationExtraction(
        records=records,
        assistant_messages_observed=assistant_messages_observed,
        assistant_messages_skipped_empty=assistant_messages_skipped_empty,
        user_messages_observed=user_messages_observed,
        system_messages_observed=0,
    )


def default_input_path(dataset: str) -> Path:
    if dataset == "sharegpt":
        return SHAREGPT_PATH
    if dataset == "ultrachat":
        return ULTRACHAT_ROOT
    raise ValueError(f"Unsupported dataset: {dataset}")


def default_output_paths(dataset: str) -> tuple[Path, Path]:
    try:
        return DEFAULT_OUTPUTS[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset}") from error


def dataset_reader_and_extractor(
    dataset: str,
    input_path: Path,
) -> tuple[Any, Callable[[dict[str, Any], MessageNormalizer, int | None], ConversationExtraction], str]:
    if dataset == "sharegpt":
        return ShareGPTReader(input_path), extract_sharegpt_assistant_messages, str(
            input_path.resolve()
        )
    if dataset == "ultrachat":
        reader = UltraChatReader(input_path)
        return reader, extract_ultrachat_assistant_messages, str(reader.train_dir)
    raise ValueError(f"Unsupported dataset: {dataset}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize ShareGPT or UltraChat into one SQLite row per assistant turn."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("sharegpt", "ultrachat"),
        required=True,
        help="Which dataset to normalize.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Path to the ShareGPT JSON file or the UltraChat dataset root/train dir.",
    )
    parser.add_argument(
        "--output-sqlite",
        type=Path,
        default=None,
        help="Path to the output SQLite database.",
    )
    parser.add_argument(
        "--output-profile",
        type=Path,
        default=None,
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
    dataset = args.dataset
    input_path = (args.input_path or default_input_path(dataset)).resolve()
    default_sqlite, default_profile = default_output_paths(dataset)
    output_sqlite, output_profile = prepare_output_paths(
        args.output_sqlite or default_sqlite,
        args.output_profile or default_profile,
        args.overwrite,
    )

    reader, extractor, input_train_dir = dataset_reader_and_extractor(dataset, input_path)
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

            extraction = extractor(
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
                "input_root": str(input_path),
                "input_train_dir": input_train_dir,
                "output_sqlite": str(output_sqlite),
                "output_profile": str(output_profile),
                "cli_args_json": json.dumps(
                    {
                        "dataset": dataset,
                        "input_path": str(input_path),
                        "output_sqlite": str(args.output_sqlite or default_sqlite),
                        "output_profile": str(args.output_profile or default_profile),
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
        "Completed extraction for "
        f"{args.dataset}: {stats.conversations_processed:,} conversations, "
        f"{stats.assistant_messages_emitted:,} assistant rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
