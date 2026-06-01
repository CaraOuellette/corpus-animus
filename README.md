# corpus-animus

Small analysis toolkit for studying stylistic features in chat corpora.

Included here:

- correction-frame detection scripts
- italic-line extraction
- embodied/action clustering
- normalization scripts for WildChat, ShareGPT, and UltraChat
- focused unit tests for the published scripts

This repo is intentionally source-only. Local datasets, SQLite outputs, reports,
and other workspace artifacts are ignored by default.

## Requirements

- Python 3.11+
- `pip install -r requirements.txt`

## Downloading datasets

`download_datasets.py` fetches the chat corpora used by the analysis scripts
from Hugging Face into local paths under this directory.

Available dataset names (positional args):

- `wildchat` — `allenai/WildChat-1M` (ODC-BY, public)
- `sharegpt` — `anon8231489123/ShareGPT_Vicuna_unfiltered` (public)
- `ultrachat` — `stingning/ultrachat` (public)
- `wildchat-4.8m` — `allenai/WildChat-4.8M-Full` (gated, manual approval)
- `lmsys-chat-1m` — `lmsys/lmsys-chat-1m` (gated, accept terms)
- `prism` — `HannahRoseKirk/prism-alignment` (CC-BY, public)
- `chatbot-arena-33k` — `lmsys/chatbot_arena_conversations` (gated, accept terms)
- `arena-140k` — `lmarena-ai/arena-human-preference-140k` (CC-BY-4.0, public)

```
python download_datasets.py                    # everything
python download_datasets.py wildchat sharegpt  # subset
python download_datasets.py wildchat --limit 1000  # smoke test (arrow datasets only)
python download_datasets.py prism arena-140k   # public-only pull
```

For gated datasets, accept the terms on the HF dataset page and authenticate
with `huggingface-cli login` or `HF_TOKEN`. WildChat-1M/UltraChat are stored
as Arrow IPC shards (the format the existing extract scripts read); the
newer datasets are kept as native parquet/jsonl snapshots.

## Main Scripts

- `extract_wildchat_assistant_messages.py`
- `extract_sharegpt_ultrachat_assistant_messages.py`
- `detect_correction_frames.py`
- `detect_correction_frames_export.py`
- `italic_lines.py`
- `embodied_clusters.py`
- `embodied_clusters_expanded.py`
- `corpus.py`

## Notes

- The scripts expect dataset files to be placed locally beside the repo or in
  the relative paths referenced in each script.
- No dataset contents are included in this repository.
