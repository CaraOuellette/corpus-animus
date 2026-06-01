# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Tests: `python -m unittest discover tests`
- Single test file: `python -m unittest tests.test_detect_correction_frames`
- Single test: `python -m unittest tests.test_corpus.CorpusTests.test_iter_messages_reads_claude_export`
- Profanity sub-project tests: `python -m unittest discover profanity/tests` (separate test root)
- Install deps: `pip install -r requirements.txt` (Python 3.11+; deps are just `pyarrow`, `datasets`, `huggingface_hub`)
- Pull a dataset: `python download_datasets.py <name> [--limit N]` (names in [README.md](README.md); gated ones need `huggingface-cli login` or `HF_TOKEN`)

## Research threads

Several analysis threads toward one goal: grounding claims about model "character"/vibes in corpus data, with confounds made legible *first* (see [DATASETS.md](DATASETS.md)). The machinery for the first two is detailed under Architecture.

- **Correction frames** (`detect_correction_frames*.py`) — corrective/contrastive rhetoric in the **assistant's own** output (scans assistant text, not user turns): "X, not Y" negation-contrast frames (`explicit_negation_contrast`), misconception-correction templates ("contrary to popular belief…", `misconception_template`), reversed "Y, not X" (`reversed_correction`), plus counted discourse markers ("actually", "in fact", "more precisely"). Span-extracting `CorrectionPattern`s carry `family`/`confidence` and emit X/Y offsets; `MarkerPattern`s are tallied.
- **Italic lines & embodiment** (`italic_lines.py`, `embodied_clusters*.py`, `objects.py`) — roleplay-style whole-line `*...*` mood/action descriptions and embodied/proprioceptive language. Driving question: where does a personal claude.ai italic-line rate fall inside PRISM's per-participant distribution? Design notes in [italic-lines-prevalence.md](italic-lines-prevalence.md) and [clod-embodiment-classificatons.md](clod-embodiment-classificatons.md).
- **Profanity prevalence** (`profanity/`) — the one thread in its own directory (own README, tests, methodology). Studies **assistant-first** profanity: does a model swear before the user does, and does the rate vary by model? `expressive_profanity.py` is the primary register-aware detector (expressive vs. dysphemistic); `profanity_first.py` is a wordlist baseline kept as a cautionary instrument (it floored on PRISM via clinical-vocabulary contamination). The model-level claim is not yet testable from available data — see `profanity/methodology.md`.

## Architecture

Two parallel ingestion paths feed into the same family of analysis scripts. Knowing which path a script is on determines what input it expects and what fields are available.

### Path A — HF corpora → Phase 1 SQLite → Phase 2 SQLite

For published chat corpora (WildChat, ShareGPT, UltraChat). Files land beside the repo (e.g. `WildChat-1M/train/`).

1. **Phase 1 extract** normalizes assistant turns into a SQLite `assistant_messages` table with the full schema in [extract_wildchat_assistant_messages.py:39](extract_wildchat_assistant_messages.py:39) (`assistant_text_raw`/`_normalized`, preceding user turn, model, timestamps, language/country/toxicity flags, turn indices). `MessageNormalizer` does smart-quote/dash folding via `SMART_PUNCTUATION_TRANSLATION`.
   - WildChat: `extract_wildchat_assistant_messages.py` → `wildchat_phase1_assistant_messages.sqlite`
   - ShareGPT/UltraChat: `extract_sharegpt_ultrachat_assistant_messages.py` (reuses Phase 1 dataclasses + writer from the WildChat module) → `{sharegpt,ultrachat}_phase1_assistant_messages.sqlite`
2. **Phase 2 detection**: `detect_correction_frames.py` reads a Phase 1 SQLite and writes `*_phase2_correction_frames.sqlite` plus a JSON summary. Patterns (`CorrectionPattern`, `MarkerPattern`) are regex with `family`/`confidence` tagging; matches are emitted with X/Y span offsets and ±500 char context.

*Shared readers:* `prepare_chat_dataset_samples.py` exports the dataset row iterators (`iter_arrow_rows`, `iter_sharegpt_objects`) that the ShareGPT/UltraChat Phase 1 extract imports. Run as a script it also writes 1000-row CSV samples of each on-disk Path A dataset for eyeballing.

### Path B — Personal exports → `corpus.iter_messages` → analysis

For Claude JSON exports (`conversations.json`) and ChatGPT HTML exports (`chat.html`), and *also* for Phase 1 SQLite when given a `.sqlite` path. The adapter is in [corpus.py](corpus.py); format dispatch is in `detect_correction_frames_export.detect_input_format`. The corresponding correction-frame entry point on this path is `detect_correction_frames_export.py` (delegates to `detect_correction_frames` internals).

`corpus.iter_messages` is the narrow interface — it yields `(month_key, sender, content, conversation_id)` tuples. Anything that walks assistant text in this repo should consume it: [italic_lines.py](italic_lines.py), [embodied_clusters.py](embodied_clusters.py), [embodied_clusters_expanded.py](embodied_clusters_expanded.py) all chain through `iter_corpus_italic_lines` → `iter_messages`.

### Stylistic-feature scripts (consume Path A or B)

- `italic_lines.py` — extracts whole-line `*...*` (mood/action descriptions), distinct from inline `*emphasis*`. Strips fenced/inline code first. `--ngrams` ranks by # distinct lines containing each n-gram, *not* raw frequency, so high-volume single lines like `*pauses*` don't dominate.
- `embodied_clusters.py` (conservative) and `embodied_clusters_expanded.py` (broad) — multi-membership regex taxonomies over normalized italic lines. A line can match many clusters or land in `uncategorized`. Treat the regex tables as first-pass heuristics meant to be refined against actual data; [objects.py](objects.py) is a stub for an in-progress object-handling cluster.

### Dataset download shapes

[download_datasets.py](download_datasets.py) writes into three different on-disk shapes depending on which reader will consume them:

1. **Arrow IPC shards** (`datasets.save_to_disk` → `data-*.arrow`) — required by Path A extract scripts (`pa.ipc.open_stream`). Used for WildChat-1M and UltraChat.
2. **Single JSON file** — ShareGPT (`ShareGPT_V3_unfiltered_cleaned_split.json`).
3. **Native HF snapshot** (parquet/jsonl, via `huggingface_hub.snapshot_download`) — used for newer datasets (WildChat-4.8M-Full, LMSYS-Chat-1M, PRISM, Chatbot Arena 33k, arena-human-preference-140k). These do **not** have Phase 1 extract scripts yet; the existing `pa.ipc.open_stream` reader will not work on them. Building one is an open task.

## Repo conventions

- **Source-only checked in.** No datasets or generated artifacts in git — [.gitignore](.gitignore) drops outputs by pattern (`*.sqlite`, `*.arrow`, `*.parquet`, `*.jsonl`, the corpora dirs, `/runs/`, `/_test_outputs/`). A script that emits a new artifact shape needs a matching ignore pattern.
- **Confound-aware analysis.** [DATASETS.md](DATASETS.md) catalogs each corpus's population, model coverage, collection UI, and filter pipeline. Any cross-dataset comparison should reference its confound checklist before attributing differences to a model — UltraChat has no humans, ShareGPT model labels are unreliable, WildChat is OpenAI-only and PII-redacted, etc.
- **Backlog files** like [italic-lines-prevalence.md](italic-lines-prevalence.md) and [clod-embodiment-classificatons.md](clod-embodiment-classificatons.md) hold in-progress design notes / taxonomies that haven't been promoted to code yet. They are tracked deliberately.
