from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.ipc as ipc


ROOT = Path(__file__).resolve().parent
SAMPLE_SIZE = 1000

ULTRACHAT_TRAIN_DIR = ROOT / "UltraChat" / "train"
WILDCHAT_TRAIN_DIR = ROOT / "WildChat-1M" / "train"
SHAREGPT_PATH = (
    ROOT
    / "ShareGPT_Vicuna_unfiltered_raw"
    / "ShareGPT_V3_unfiltered_cleaned_split.json"
)

SAMPLE_OUTPUTS = {
    "UltraChat": ROOT / "ultrachat_sample_1000.csv",
    "WildChat": ROOT / "wildchat_sample_1000.csv",
    "ShareGPT": ROOT / "sharegpt_sample_1000.csv",
}
REPORT_OUTPUT = ROOT / "dataset_schema_report.md"


def is_id_like(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered == "id"
        or lowered.endswith("_id")
        or lowered == "conversation_hash"
        or lowered == "hashed_ip"
        or lowered == "turn_identifier"
        or lowered.endswith("_hash")
    )


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def iter_arrow_rows(files: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in sorted(files):
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                for row in batch.to_pylist():
                    yield row


def iter_sharegpt_objects(path: Path, chunk_size: int = 1 << 20) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        index = 0

        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError("ShareGPT file is empty.")
            buffer += chunk
            while index < len(buffer) and buffer[index].isspace():
                index += 1
            if index < len(buffer):
                break

        if buffer[index] != "[":
            raise ValueError("Expected ShareGPT JSON to be a top-level array.")
        index += 1

        while True:
            while True:
                while index < len(buffer) and buffer[index] in " \t\r\n,":
                    index += 1
                if index < len(buffer):
                    break
                chunk = handle.read(chunk_size)
                if not chunk:
                    return
                buffer = buffer[index:] + chunk
                index = 0

            if buffer[index] == "]":
                return

            while True:
                try:
                    item, end = decoder.raw_decode(buffer, index)
                    yield item
                    index = end
                    if index > chunk_size:
                        buffer = buffer[index:]
                        index = 0
                    break
                except json.JSONDecodeError:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        raise
                    buffer = buffer[index:] + chunk
                    index = 0


def extract_ultrachat_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    turns = row.get("data") or []
    if len(turns) < 2:
        return None
    prompt = turns[0]
    response = turns[1]
    if not prompt or not response:
        return None
    return prompt, response


def extract_wildchat_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    conversation = row.get("conversation") or []
    for index, message in enumerate(conversation):
        if message.get("role") != "user":
            continue
        prompt = message.get("content")
        if not prompt:
            continue
        for reply in conversation[index + 1 :]:
            if reply.get("role") == "assistant" and reply.get("content"):
                return prompt, reply["content"]
    return None


def extract_sharegpt_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    conversation = row.get("conversations") or []
    for index, message in enumerate(conversation):
        if message.get("from") not in {"human", "user"}:
            continue
        prompt = message.get("value") or message.get("text")
        if not prompt:
            continue
        for reply in conversation[index + 1 :]:
            if reply.get("from") in {"gpt", "chatgpt", "bing", "bard"}:
                response = reply.get("value") or reply.get("text")
                if response:
                    return prompt, response
    return None


def write_samples() -> None:
    ultra_rows: list[dict[str, str]] = []
    for row in iter_arrow_rows(ULTRACHAT_TRAIN_DIR.glob("data-*.arrow")):
        pair = extract_ultrachat_pair(row)
        if not pair:
            continue
        ultra_rows.append(
            {
                "source_dataset": "UltraChat",
                "source_id": format_value(row.get("id")),
                "model": "",
                "human_prompt": pair[0],
                "ai_response": pair[1],
            }
        )
        if len(ultra_rows) >= SAMPLE_SIZE:
            break

    wild_rows: list[dict[str, str]] = []
    for row in iter_arrow_rows(WILDCHAT_TRAIN_DIR.glob("data-*.arrow")):
        pair = extract_wildchat_pair(row)
        if not pair:
            continue
        wild_rows.append(
            {
                "source_dataset": "WildChat",
                "source_id": format_value(row.get("conversation_hash")),
                "model": format_value(row.get("model")),
                "human_prompt": pair[0],
                "ai_response": pair[1],
            }
        )
        if len(wild_rows) >= SAMPLE_SIZE:
            break

    share_rows: list[dict[str, str]] = []
    for row in iter_sharegpt_objects(SHAREGPT_PATH):
        pair = extract_sharegpt_pair(row)
        if not pair:
            continue
        share_rows.append(
            {
                "source_dataset": "ShareGPT",
                "source_id": format_value(row.get("id")),
                "model": "",
                "human_prompt": pair[0],
                "ai_response": pair[1],
            }
        )
        if len(share_rows) >= SAMPLE_SIZE:
            break

    for dataset_name, rows in [
        ("UltraChat", ultra_rows),
        ("WildChat", wild_rows),
        ("ShareGPT", share_rows),
    ]:
        output_path = SAMPLE_OUTPUTS[dataset_name]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source_dataset",
                    "source_id",
                    "model",
                    "human_prompt",
                    "ai_response",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)


def collect_wildchat_value_sets() -> dict[str, set[Any]]:
    values = {
        "model": set(),
        "language": set(),
        "toxic": set(),
        "redacted": set(),
        "conversation[].role": set(),
        "conversation[].language": set(),
        "conversation[].redacted": set(),
        "conversation[].toxic": set(),
    }

    for row in iter_arrow_rows(WILDCHAT_TRAIN_DIR.glob("data-*.arrow")):
        for key in ("model", "language", "toxic", "redacted"):
            value = row.get(key)
            if value is not None:
                values[key].add(value)

        for message in row.get("conversation") or []:
            for raw_key, output_key in (
                ("role", "conversation[].role"),
                ("language", "conversation[].language"),
                ("redacted", "conversation[].redacted"),
                ("toxic", "conversation[].toxic"),
            ):
                value = message.get(raw_key)
                if value is not None:
                    values[output_key].add(value)

    return values


def collect_sharegpt_summary() -> dict[str, Any]:
    top_level_keys: set[str] = set()
    conversation_keys: set[str] = set()
    markdown_keys: set[str] = set()
    from_values: set[str] = set()
    row_count = 0

    for row in iter_sharegpt_objects(SHAREGPT_PATH):
        row_count += 1
        top_level_keys.update(row.keys())
        for message in row.get("conversations") or []:
            conversation_keys.update(message.keys())
            from_value = message.get("from")
            if from_value is not None:
                from_values.add(from_value)
            markdown = message.get("markdown")
            if isinstance(markdown, dict):
                markdown_keys.update(markdown.keys())

    return {
        "row_count": row_count,
        "top_level_keys": sorted(top_level_keys),
        "conversation_keys": sorted(conversation_keys),
        "markdown_keys": sorted(markdown_keys),
        "from_values": sorted(from_values),
    }


def arrow_schema_lines(schema: pa.Schema, prefix: str = "") -> list[str]:
    lines: list[str] = []
    for field in schema:
        name = field.name
        field_path = f"{prefix}{name}"
        if is_id_like(name):
            continue

        field_type = field.type
        if pa.types.is_struct(field_type):
            nested = arrow_schema_lines(field_type, f"{field_path}.")
            lines.extend(nested)
        elif pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
            value_field = field_type.value_field
            if pa.types.is_struct(value_field.type):
                nested = arrow_schema_lines(value_field.type, f"{field_path}[].")
                lines.extend(nested)
            else:
                lines.append(f"`{field_path}[]` ({value_field.type})")
        else:
            lines.append(f"`{field_path}` ({field_type})")
    return lines


def load_arrow_schema(path: Path) -> pa.Schema:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_stream(source)
        return reader.schema


def bullet_lines(items: Iterable[str]) -> list[str]:
    return [f"- {item}" for item in items]


def render_values(values: Iterable[Any]) -> str:
    rendered = [format_value(value) for value in values]
    return ", ".join(f"`{value}`" if value else "`<empty>`" for value in rendered)


def write_report() -> None:
    ultrachat_schema = load_arrow_schema(
        sorted(ULTRACHAT_TRAIN_DIR.glob("data-*.arrow"))[0]
    )
    wildchat_schema = load_arrow_schema(
        sorted(WILDCHAT_TRAIN_DIR.glob("data-*.arrow"))[0]
    )
    wildchat_values = collect_wildchat_value_sets()
    sharegpt_summary = collect_sharegpt_summary()

    report_lines: list[str] = [
        "# Dataset Sample and Schema Report",
        "",
        "The existing `chat_data_sample.csv` appears to be ShareGPT-derived.",
        "Its first row matches the first ShareGPT record after flattening the first human/GPT pair.",
        "",
        "ID-like fields were omitted below, per request.",
        "",
        "## UltraChat",
        "",
        f"Sample file: `{SAMPLE_OUTPUTS['UltraChat'].name}`",
        "",
        "Columns:",
    ]
    report_lines.extend(bullet_lines(arrow_schema_lines(ultrachat_schema)))
    report_lines.extend(
        [
            "",
            "Limited-value fields:",
            "- No non-ID scalar fields with a small fixed set of values were exposed at the top level.",
            "- The main non-ID field is `data[]`, a list of utterance strings.",
            "",
            "## WildChat",
            "",
            f"Sample file: `{SAMPLE_OUTPUTS['WildChat'].name}`",
            "",
            "Columns:",
        ]
    )
    report_lines.extend(bullet_lines(arrow_schema_lines(wildchat_schema)))
    report_lines.extend(
        [
            "",
            "Limited-value fields:",
            f"- `model`: {render_values(sorted(wildchat_values['model'], key=str))}",
            f"- `toxic`: {render_values(sorted(wildchat_values['toxic'], key=str))}",
            f"- `redacted`: {render_values(sorted(wildchat_values['redacted'], key=str))}",
            f"- `conversation[].role`: {render_values(sorted(wildchat_values['conversation[].role'], key=str))}",
            f"- `conversation[].toxic`: {render_values(sorted(wildchat_values['conversation[].toxic'], key=str))}",
            f"- `conversation[].redacted`: {render_values(sorted(wildchat_values['conversation[].redacted'], key=str))}",
            "",
            "Higher-cardinality but still potentially useful:",
            f"- `language`: {len(wildchat_values['language'])} values",
            f"- `conversation[].language`: {len(wildchat_values['conversation[].language'])} values",
            "",
            "## ShareGPT",
            "",
            f"Sample file: `{SAMPLE_OUTPUTS['ShareGPT'].name}`",
            "",
            "Columns:",
            "- `conversations[]` (list)",
            "- `conversations[].from` (string)",
            "- `conversations[].value` (string)",
            "- `conversations[].text` (string, present in some rows)",
            "- `conversations[].markdown` (object, present in some rows)",
        ]
    )
    if sharegpt_summary["markdown_keys"]:
        report_lines.append(
            "- `conversations[].markdown.{"
            + ", ".join(sharegpt_summary["markdown_keys"])
            + "}`"
        )
    report_lines.extend(
        [
            "",
            "Limited-value fields:",
            f"- `conversations[].from`: {render_values(sharegpt_summary['from_values'])}",
        ]
    )

    with REPORT_OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(report_lines) + "\n")


def main() -> None:
    write_samples()
    write_report()

    for dataset_name, path in SAMPLE_OUTPUTS.items():
        print(f"{dataset_name}: wrote {path.name}")
    print(f"Report: wrote {REPORT_OUTPUT.name}")


if __name__ == "__main__":
    main()
