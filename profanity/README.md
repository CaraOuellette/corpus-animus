# Profanity-prevalence project

Studies **assistant-first profanity** in chat corpora: does a model swear
before the user does, and does the rate vary by model? One thread of the
broader "ground model-character vibes in data" effort.

## Files

- `expressive_profanity.py` — primary detector. Register-aware *expressive*
  (cathartic, not dysphemistic) profanity: sentence-opener "oh fuck you're
  right", all-caps, "fuck yes", holy-compounds, asterisk-action + expletive.
  Short positional templates, not a wordlist.
- `profanity_first.py` — wordlist baseline (locked LDNOOBW + a curated vulgar
  subset). Kept as a cautionary instrument: it failed on PRISM via
  clinical-vocabulary contamination.
- `profanity_wordlist.txt`, `profanity_wordlist_vulgar.txt` — the wordlists.
- `phase_a_prism.md` — Phase A writeup (PRISM, public data).
- `methodology.md` — operationalization, the pre-registered prediction,
  confounds, and what data would actually settle the model-level question.
- `tests/` — `python -m unittest discover profanity/tests` (from repo root) or
  `cd profanity && python -m unittest discover tests`.

## Findings so far

- **PRISM is floored** (~0% genuine assistant-first vulgar use): the behavior
  is register-dependent — research-elicited prompts don't produce it.
- **Applied to the personal claude.ai corpus** (in the **`claude-drift`** repo,
  not here — it's personal data): the "Sonnet 4.5 is the sweariest Claude" vibe
  **did not hold**; an earlier era (Opus 4.1) ran ~2× higher. See
  `claude-drift/projects/profanity/`.

## Status

Instrument validated on real Claude chat data. The model-level claim is **not
yet testable** from available data — it needs within-period model variation
(see `methodology.md`).
