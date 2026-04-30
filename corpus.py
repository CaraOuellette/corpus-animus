from __future__ import annotations

"""
Lightweight corpus adapter for local conversation export files.

This provides the narrow interface expected by `italic_lines.py` and
`embodied_clusters.py`:

- `DEFAULT_DB_PATH`
- `iter_messages(input_path=None)`

Despite the legacy name, the default input is not a SQLite database here.
It is usually an export file such as `conversations.json` or `chat.html`,
but this adapter also supports local Phase 1 assistant-message SQLite files.
"""

import sqlite3
from pathlib import Path
from typing import Iterator

from detect_correction_frames_export import (
    CHATGPT_HTML_PREFIX,
    ChatGPTHTMLExportReader,
    ClaudeExportReader,
    detect_input_format,
    extract_chatgpt_text,
    extract_claude_text,
    isoformat_utc,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = (
    ROOT / "conversations.json"
    if (ROOT / "conversations.json").exists()
    else ROOT / "chat.html"
)


def month_key(timestamp: str | None) -> str:
    if not timestamp:
        return ""
    return timestamp[:7]


def iter_claude_messages(input_path: Path) -> Iterator[tuple[str, str, str, str]]:
    reader = ClaudeExportReader(input_path)
    try:
        for conversation in reader.conversations:
            conversation_id = str(
                conversation.get("uuid")
                or conversation.get("id")
                or ""
            )
            for message in conversation.get("chat_messages") or []:
                sender = str(message.get("sender") or "").lower()
                text = extract_claude_text(message)
                if not isinstance(text, str) or not text.strip():
                    continue
                timestamp = isoformat_utc(message.get("created_at"))
                yield (
                    month_key(timestamp),
                    sender,
                    text,
                    conversation_id,
                )
    finally:
        reader.close()


def iter_chatgpt_messages(input_path: Path) -> Iterator[tuple[str, str, str, str]]:
    reader = ChatGPTHTMLExportReader(input_path)
    try:
        for conversation in reader.conversations:
            conversation_id = str(
                conversation.get("conversation_id")
                or conversation.get("id")
                or ""
            )
            for message in reader.iter_visible_messages(conversation):
                sender = str(message.get("author", {}).get("role") or "").lower()
                text = extract_chatgpt_text(message)
                if not isinstance(text, str) or not text.strip():
                    continue
                timestamp = isoformat_utc(message.get("create_time"))
                yield (
                    month_key(timestamp),
                    sender,
                    text,
                    conversation_id,
                )
    finally:
        reader.close()


def iter_phase1_sqlite_messages(
    input_path: Path,
    batch_size: int = 10_000,
) -> Iterator[tuple[str, str, str, str]]:
    connection = sqlite3.connect(str(input_path))
    try:
        cursor = connection.execute(
            """
            SELECT
                source_conversation_id,
                message_timestamp_utc,
                conversation_timestamp_utc,
                assistant_text_raw
            FROM assistant_messages
            ORDER BY id
            """
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for source_conversation_id, message_timestamp_utc, conversation_timestamp_utc, assistant_text_raw in rows:
                if not isinstance(assistant_text_raw, str) or not assistant_text_raw.strip():
                    continue
                timestamp = message_timestamp_utc or conversation_timestamp_utc
                yield (
                    month_key(timestamp),
                    "assistant",
                    assistant_text_raw,
                    str(source_conversation_id or ""),
                )
    finally:
        connection.close()


def iter_messages(input_path: Path | str | None = None) -> Iterator[tuple[str, str, str, str]]:
    """
    Yield `(month_key, sender, content, conversation_id)` rows from a supported
    export file.

    Supported formats:
    - Claude JSON export: `conversations.json`
    - ChatGPT HTML export: `chat.html`
    - Phase 1 assistant-message SQLite:
      `wildchat_phase1_assistant_messages.sqlite`
      `sharegpt_phase1_assistant_messages.sqlite`
      `ultrachat_phase1_assistant_messages.sqlite`
    """
    if input_path is None:
        input_path = DEFAULT_DB_PATH

    path = Path(input_path).resolve()
    if path.suffix.lower() == ".sqlite":
        yield from iter_phase1_sqlite_messages(path)
        return

    input_format = detect_input_format(path)
    if input_format == "claude-json":
        yield from iter_claude_messages(path)
        return
    if input_format == "chatgpt-html":
        yield from iter_chatgpt_messages(path)
        return
    raise ValueError(f"Unsupported input format for corpus iteration: {path}")
