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
