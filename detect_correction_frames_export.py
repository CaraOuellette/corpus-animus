from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import detect_correction_frames as base


ROOT = Path(__file__).resolve().parent
DEFAULT_CLAUDE_INPUT = ROOT / "conversations.json"
DEFAULT_CHATGPT_INPUT = ROOT / "chat.html"
SCRIPT_VERSION = "0.1.0"
CHATGPT_HTML_PREFIX = "var jsonData = "


def isoformat_utc(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    return str(value)


def extract_claude_text(message: dict[str, Any]) -> str | None:
    direct_text = message.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    fragments: list[str] = []
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            fragments.append(text)

    if not fragments:
        return None
    return "\n\n".join(fragments)


def extract_chatgpt_text(message: dict[str, Any]) -> str | None:
    content = message.get("content") or {}
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None

    fragments: list[str] = []
    for part in parts:
        if isinstance(part, str) and part.strip():
            fragments.append(part)
            continue
        if not isinstance(part, dict):
            continue
        if part.get("content_type") == "audio_transcription":
            transcript = part.get("text")
            if isinstance(transcript, str) and transcript.strip():
                fragments.append(transcript)
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            fragments.append(text)

    if not fragments:
        return None
    return "\n\n".join(fragments)


class ClaudeExportReader:
    def __init__(self, input_path: Path) -> None:
        self.input_path = input_path.resolve()
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input export does not exist: {self.input_path}")
        with self.input_path.open("r", encoding="utf-8") as handle:
            self.conversations = json.load(handle)
        if not isinstance(self.conversations, list):
            raise ValueError(
                f"Expected a top-level list in Claude export: {self.input_path}"
            )

    def iter_messages(
        self,
        limit_messages: int | None = None,
        batch_size: int = base.BATCH_SIZE,
    ) -> Iterator[base.InputMessage]:
        del batch_size

        emitted = 0
        next_message_id = 1
        for conversation in self.conversations:
            conversation_id = str(
                conversation.get("uuid")
                or conversation.get("id")
                or f"conversation-{next_message_id}"
            )
            assistant_turn_number = 0
            preceding_user_text: str | None = None

            for index, message in enumerate(conversation.get("chat_messages") or []):
                sender = str(message.get("sender") or "").lower()
                text = extract_claude_text(message)
                normalized = base.normalize_for_detection(text) if text else None

                if sender == "human":
                    if normalized:
                        preceding_user_text = normalized
                    continue

                if sender != "assistant" or not normalized:
                    continue

                assistant_turn_number += 1
                yield base.InputMessage(
                    assistant_message_id=next_message_id,
                    source_conversation_id=conversation_id,
                    conversation_message_index=index,
                    assistant_turn_number=assistant_turn_number,
                    assistant_text_normalized=normalized,
                    preceding_user_text_normalized=preceding_user_text,
                    model=None,
                    conversation_language=None,
                    message_timestamp_utc=isoformat_utc(message.get("created_at")),
                )

                emitted += 1
                next_message_id += 1
                if limit_messages is not None and emitted >= limit_messages:
                    return

    def close(self) -> None:
        return None


class ChatGPTHTMLExportReader:
    def __init__(self, input_path: Path) -> None:
        self.input_path = input_path.resolve()
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input export does not exist: {self.input_path}")
        self.conversations = self._load_conversations()

    def _load_conversations(self) -> list[dict[str, Any]]:
        html_text = self.input_path.read_text(encoding="utf-8")
        prefix_index = html_text.find(CHATGPT_HTML_PREFIX)
        if prefix_index < 0:
            raise ValueError(
                f"Could not find embedded jsonData array in ChatGPT export: {self.input_path}"
            )

        payload = html_text[prefix_index + len(CHATGPT_HTML_PREFIX) :]
        conversations, _ = json.JSONDecoder().raw_decode(payload)
        if not isinstance(conversations, list):
            raise ValueError(
                f"Expected embedded jsonData to decode to a list: {self.input_path}"
            )
        return conversations

    def iter_visible_messages(
        self, conversation: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        mapping = conversation.get("mapping") or {}
        current_node = conversation.get("current_node")
        seen_nodes: set[str] = set()
        messages: list[dict[str, Any]] = []

        while isinstance(current_node, str) and current_node not in seen_nodes:
            seen_nodes.add(current_node)
            node = mapping.get(current_node) or {}
            message = node.get("message")
            if (
                isinstance(message, dict)
                and message.get("id")
                and isinstance(message.get("content"), dict)
                and isinstance(message["content"].get("parts"), list)
                and message["content"]["parts"]
                and (
                    message.get("author", {}).get("role") != "system"
                    or message.get("metadata", {}).get("is_user_system_message")
                )
            ):
                messages.append(message)
            current_node = node.get("parent")

        yield from reversed(messages)

    def iter_messages(
        self,
        limit_messages: int | None = None,
        batch_size: int = base.BATCH_SIZE,
    ) -> Iterator[base.InputMessage]:
        del batch_size

        emitted = 0
        next_message_id = 1
        for conversation in self.conversations:
            conversation_id = str(
                conversation.get("conversation_id")
                or conversation.get("id")
                or f"conversation-{next_message_id}"
            )
            assistant_turn_number = 0
            preceding_user_text: str | None = None

            for index, message in enumerate(self.iter_visible_messages(conversation)):
                author = str(message.get("author", {}).get("role") or "").lower()
                text = extract_chatgpt_text(message)
                normalized = base.normalize_for_detection(text) if text else None

                if author == "user":
                    if normalized:
                        preceding_user_text = normalized
                    continue

                if author != "assistant" or not normalized:
                    continue

                assistant_turn_number += 1
                metadata = message.get("metadata") or {}
                model = metadata.get("model_slug") or conversation.get("default_model_slug")
                yield base.InputMessage(
                    assistant_message_id=next_message_id,
                    source_conversation_id=conversation_id,
                    conversation_message_index=index,
                    assistant_turn_number=assistant_turn_number,
                    assistant_text_normalized=normalized,
                    preceding_user_text_normalized=preceding_user_text,
                    model=model if isinstance(model, str) else None,
                    conversation_language=None,
                    message_timestamp_utc=isoformat_utc(message.get("create_time")),
                )

                emitted += 1
                next_message_id += 1
                if limit_messages is not None and emitted >= limit_messages:
                    return

    def close(self) -> None:
        return None


def detect_input_format(input_path: Path) -> str:
    suffix = input_path.suffix.lower()
    if suffix == ".html":
        return "chatgpt-html"

    sample = input_path.read_text(encoding="utf-8", errors="ignore")[:4096]
    if CHATGPT_HTML_PREFIX in sample:
        return "chatgpt-html"
    if '"chat_messages"' in sample:
        return "claude-json"
    raise ValueError(f"Could not auto-detect input format for {input_path}")


def build_reader(input_path: Path, input_format: str) -> ClaudeExportReader | ChatGPTHTMLExportReader:
    if input_format == "claude-json":
        return ClaudeExportReader(input_path)
    if input_format == "chatgpt-html":
        return ChatGPTHTMLExportReader(input_path)
    raise ValueError(f"Unsupported input format: {input_format}")


def default_output_paths(input_path: Path) -> tuple[Path, Path]:
    stem = input_path.stem
    return (
        input_path.with_name(f"{stem}_correction_frames.sqlite"),
        input_path.with_name(f"{stem}_correction_frames_summary.json"),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run correction-frame detection directly on exported conversation archives."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_CLAUDE_INPUT if DEFAULT_CLAUDE_INPUT.exists() else DEFAULT_CHATGPT_INPUT,
        help="Path to a supported export file such as conversations.json or chat.html.",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "claude-json", "chatgpt-html"),
        default="auto",
        help="Explicit export format. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--output-sqlite",
        type=Path,
        default=None,
        help="Path to the output SQLite database. Defaults beside the input file.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Path to the JSON summary artifact. Defaults beside the input file.",
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
        default=base.BATCH_SIZE,
        help="Number of messages to process between commits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output artifacts.",
    )
    return parser.parse_args(argv)


def run_detection(args: argparse.Namespace) -> base.DetectionStats:
    input_path = args.input_path.resolve()
    input_format = (
        detect_input_format(input_path)
        if args.input_format == "auto"
        else args.input_format
    )
    default_sqlite, default_summary = default_output_paths(input_path)
    output_sqlite, output_summary = base.prepare_output_paths(
        args.output_sqlite or default_sqlite,
        args.output_summary or default_summary,
        args.overwrite,
    )

    reader = build_reader(input_path, input_format)
    writer = base.DetectionSQLiteWriter(output_sqlite, overwrite=args.overwrite)
    detector = base.CorrectionFrameDetector()
    marker_counter = base.DiscourseMarkerCounter()
    started_at_utc = base.utc_now_iso()
    stats = base.DetectionStats()

    writer.create_tables()
    try:
        messages = reader.iter_messages(
            limit_messages=args.limit_messages,
            batch_size=args.batch_size,
        )
        for batch in base.iter_batches(messages, args.batch_size):
            for message in batch:
                matches = detector.find(message.assistant_text_normalized)
                marker_counts = marker_counter.count(message.assistant_text_normalized)
                writer.insert_correction_matches(message, matches)
                writer.insert_marker_counts(message, marker_counts)
                writer.insert_message_flags(message, matches, marker_counts)
                stats.update(message, matches, marker_counts)

            writer.flush()

        writer.create_indexes()
        writer.write_summary(stats)
        writer.write_run_metadata(
            {
                "script_name": Path(__file__).name,
                "script_version": SCRIPT_VERSION,
                "input_source_path": str(input_path),
                "input_source_kind": input_format,
                "output_sqlite": str(output_sqlite),
                "output_summary": str(output_summary),
                "cli_args_json": json.dumps(
                    {
                        "input_path": str(args.input_path),
                        "input_format": args.input_format,
                        "output_sqlite": str(args.output_sqlite)
                        if args.output_sqlite
                        else None,
                        "output_summary": str(args.output_summary)
                        if args.output_summary
                        else None,
                        "limit_messages": args.limit_messages,
                        "batch_size": args.batch_size,
                        "overwrite": args.overwrite,
                    },
                    sort_keys=True,
                ),
                "started_at_utc": started_at_utc,
                "completed_at_utc": base.utc_now_iso(),
                "messages_scanned": stats.messages_scanned,
                "correction_instances": stats.correction_instances,
                "messages_with_correction": stats.messages_with_correction,
                "discourse_marker_occurrences": stats.discourse_marker_occurrences,
                "messages_with_discourse_marker": stats.messages_with_discourse_marker,
            }
        )
        base.write_summary_artifact(output_summary, stats)
    finally:
        reader.close()
        writer.close()

    return stats


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stats = run_detection(args)
    print(
        "Completed export detection: "
        f"{stats.messages_scanned:,} messages scanned, "
        f"{stats.correction_instances:,} correction instances, "
        f"{stats.discourse_marker_occurrences:,} marker occurrences."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
