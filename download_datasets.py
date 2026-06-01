"""Download chat corpora from Hugging Face into local paths matching what the
extract / analysis scripts in this repo expect.

Three storage shapes are used:

1. **Arrow shards** (`data-*.arrow` via `datasets.save_to_disk`) — matches the
   `pa.ipc.open_stream` reader in `extract_wildchat_assistant_messages.py` and
   `extract_sharegpt_ultrachat_assistant_messages.py`. Used for the original
   WildChat-1M and UltraChat slices.
2. **Single JSON file** — ShareGPT's distribution shape.
3. **Snapshot of native files** (parquet / jsonl, kept as-published via
   `huggingface_hub.snapshot_download`) — used for newer datasets where we
   prefer to keep the Hugging Face layout intact and read parquet directly
   from per-dataset readers (no Arrow IPC conversion).

Gated datasets (`allenai/WildChat-4.8M-Full`, `lmsys/lmsys-chat-1m`,
`lmsys/chatbot_arena_conversations`) require accepting the terms on the HF
dataset page (and, for WildChat-4.8M-Full, manual approval). Authenticate
with `huggingface-cli login` or set `HF_TOKEN` before pulling them.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent


ARROW_DATASETS = {
    "wildchat": {
        "hf_repo": "allenai/WildChat-1M",
        "target": ROOT / "WildChat-1M" / "train",
        "split": "train",
    },
    "ultrachat": {
        "hf_repo": "stingning/ultrachat",
        "target": ROOT / "UltraChat" / "train",
        "split": "train",
    },
}

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_TARGET_DIR = ROOT / "ShareGPT_Vicuna_unfiltered_raw"

# Datasets pulled as native HF snapshots (parquet / jsonl), not converted to
# Arrow IPC. Keys are the CLI names; values describe where to put the files
# and which file globs to fetch.
SNAPSHOT_DATASETS = {
    "wildchat-4.8m": {
        "hf_repo": "allenai/WildChat-4.8M-Full",
        "target": ROOT / "WildChat-4.8M-Full",
        "allow_patterns": ["data/*.parquet", "*.md", "LICENSE*"],
        "gated": True,
        "note": "Manual approval required on the HF dataset page.",
    },
    "lmsys-chat-1m": {
        "hf_repo": "lmsys/lmsys-chat-1m",
        "target": ROOT / "lmsys-chat-1m",
        "allow_patterns": ["data/*.parquet", "*.md"],
        "gated": True,
        "note": "Accept the terms on the HF dataset page before downloading.",
    },
    "prism": {
        "hf_repo": "HannahRoseKirk/prism-alignment",
        "target": ROOT / "prism-alignment",
        "allow_patterns": ["*.jsonl", "*.md"],
        "gated": False,
        "note": None,
    },
    "chatbot-arena-33k": {
        "hf_repo": "lmsys/chatbot_arena_conversations",
        "target": ROOT / "chatbot_arena_conversations",
        "allow_patterns": ["data/*.parquet", "*.md"],
        "gated": True,
        "note": "Accept the terms on the HF dataset page before downloading.",
    },
    "arena-140k": {
        "hf_repo": "lmarena-ai/arena-human-preference-140k",
        "target": ROOT / "arena-human-preference-140k",
        "allow_patterns": ["data/*.parquet", "*.md"],
        "gated": False,
        "note": None,
    },
}

ALL_DATASETS = (
    "wildchat",
    "sharegpt",
    "ultrachat",
    *SNAPSHOT_DATASETS.keys(),
)


def download_arrow_dataset(
    name: str,
    hf_repo: str,
    target: Path,
    split: str,
    split_slice: str | None,
    overwrite: bool,
    hf_revision: str | None,
) -> None:
    from datasets import load_dataset

    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)

    split_arg = f"{split}[:{split_slice}]" if split_slice else split
    print(f"[{name}] loading {hf_repo} ({split_arg})...")
    dataset = load_dataset(hf_repo, split=split_arg, revision=hf_revision)
    print(f"[{name}] saving Arrow shards to {target}")
    dataset.save_to_disk(str(target))
    arrow_files = sorted(target.glob("data-*.arrow"))
    print(f"[{name}] wrote {len(arrow_files)} Arrow shard(s).")


def download_snapshot(
    name: str,
    hf_repo: str,
    target: Path,
    allow_patterns: Sequence[str],
    overwrite: bool,
    hf_revision: str | None,
) -> None:
    from huggingface_hub import snapshot_download

    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] snapshotting {hf_repo} -> {target}")
    snapshot_download(
        repo_id=hf_repo,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=list(allow_patterns),
        revision=hf_revision,
    )
    pulled = sorted(p.name for p in target.rglob("*") if p.is_file())
    print(f"[{name}] pulled {len(pulled)} file(s).")


def download_sharegpt(overwrite: bool, hf_revision: str | None) -> None:
    from huggingface_hub import hf_hub_download

    SHAREGPT_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    target_file = SHAREGPT_TARGET_DIR / SHAREGPT_FILENAME

    if target_file.exists() and not overwrite:
        raise FileExistsError(
            f"{target_file} already exists. Use --overwrite to replace it."
        )

    print(f"[sharegpt] downloading {SHAREGPT_FILENAME} from {SHAREGPT_REPO}...")
    cached_path = hf_hub_download(
        repo_id=SHAREGPT_REPO,
        filename=SHAREGPT_FILENAME,
        repo_type="dataset",
        revision=hf_revision,
    )
    if target_file.exists():
        target_file.unlink()
    shutil.copy2(cached_path, target_file)
    print(f"[sharegpt] wrote {target_file}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=ALL_DATASETS,
        default=list(ALL_DATASETS),
        help="Datasets to download. Defaults to all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing target directories/files.",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default=None,
        help=(
            "Optional split slice for arrow datasets (passed to load_dataset, "
            "e.g. '1000' loads the first 1000 rows). Useful for smoke tests. "
            "Ignored for ShareGPT."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision (branch, tag, or commit) for all datasets.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name in args.datasets:
        if name == "sharegpt":
            download_sharegpt(overwrite=args.overwrite, hf_revision=args.revision)
        elif name in ARROW_DATASETS:
            spec = ARROW_DATASETS[name]
            download_arrow_dataset(
                name=name,
                hf_repo=spec["hf_repo"],
                target=spec["target"],
                split=spec["split"],
                split_slice=args.limit,
                overwrite=args.overwrite,
                hf_revision=args.revision,
            )
        elif name in SNAPSHOT_DATASETS:
            spec = SNAPSHOT_DATASETS[name]
            if spec.get("note"):
                print(f"[{name}] note: {spec['note']}")
            download_snapshot(
                name=name,
                hf_repo=spec["hf_repo"],
                target=spec["target"],
                allow_patterns=spec["allow_patterns"],
                overwrite=args.overwrite,
                hf_revision=args.revision,
            )
        else:  # pragma: no cover - argparse choices guard against this
            raise ValueError(f"Unknown dataset: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
