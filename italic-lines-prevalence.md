# italic-lines prevalence

## v1 question

Given Rain's claude.ai italic-line rate R, and PRISM's distribution of per-participant rates D restricted to Anthropic-model conversations: where does R sit inside D?

## Definition

italic-line = `^[ \t]*\*([^*\n]+?)\*[ \t]*$` — applied after `CODE_BLOCK_PATTERN` strip, both as defined in `italic_lines.py`. Do not redefine.

## Units of analysis

- Per-conversation: fraction of conversations with ≥1 italic-line
- Per-assistant-turn: fraction of assistant turns with ≥1 italic-line

## Datasets

- v1 reference corpus: PRISM (`HannahRoseKirk/prism-alignment`), restricted to Anthropic-model conversations
- v1 sample: rain claude.ai corpus (`~/.claude-chats/conversations.db`)
- v2 (out of scope): arena-human-preference-140k (cross-vendor comparison), WildChat-4.8M-Full (scale + non-Anthropic baseline)

## Tasks

extend `corpus.py`:
- [ ] Add a PRISM adapter to `iter_messages` that detects a PRISM directory (or `utterances.jsonl` path) and yields the standard `(month_key, sender, content, conversation_id)` tuple. Tests in `tests/`.

edit `italic_lines.py`:
- [ ] `prevalence` subcommand: emits per-conversation and per-assistant-turn rates, overall and stratified by:
  - model family (Anthropic vs others, by substring match on model name)
  - PRISM `conversation_type` (Unguided / Values-guided / Controversy-guided)
  - PRISM per-utterance rating bucket (terciles of the rating distribution: low / mid / high)
- [ ] Output: markdown report at `reports/italic_lines_prevalence_v1.md` containing the rates, the stratifications, and Rain's claude.ai rate located inside PRISM's per-participant distribution (percentile + ASCII histogram of D with R marked).
- [ ] Validation artifact: write a random 200-row sample of detector matches to `reports/italic_lines_validation_sample.jsonl`. Manual labeling deferred but the artifact must exist.

## Confounds (per DATASETS.md)

- PRISM prompt distribution is study-shaped, not organic.
- claude.ai is n=1 user; PRISM is ~1,500 paid participants. The "R inside D" framing is honest about user-level variance, not about prompt-distribution differences.
- Languages are not filtered in v1; treated as a confound, deferred.
- "Anthropic family" coarsens across Claude versions.

## Out of scope for v1

- Cross-vendor comparison (Arena-140k → v2)
- Per-user-turn stratification (ambiguous; revisit with a clearer question)
- Hand-labeling the validation sample (artifact written, labeling deferred)
- Language stratification
