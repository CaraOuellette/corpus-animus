# Methodology: assistant-first profanity

## The move: vibe → claim

Take an impression ("model X swears a lot"), operationalize it, then ask
whether the observed rate sits inside or outside a reference distribution —
the same construct-validity discipline as the italic-actions study.

## Operationalization

- **Unit:** conversation, or `(conversation, model)` pair when a corpus has
  multiple models per conversation (e.g. PRISM's pairwise battles).
- **Outcome:** *assistant-first* profanity — the assistant produces profanity
  in a turn before any user turn does. Conditioning on "user didn't swear
  first" removes the trivial mirroring confound.
- **What counts as profanity** — two instruments:
  - `expressive_profanity.py` (primary): *cathartic / expressive* register
    only — openers, all-caps, "fuck yes", holy-compounds, asterisk-action +
    expletive. Dysphemistic/abusive uses are anti-markers, not hits. Built
    because the behavior of interest is expressive, not aggressive.
  - `profanity_first.py` (baseline): wordlist (LDNOOBW + a curated vulgar
    subset). Cautionary — see PRISM contamination below.

Both detectors take PRISM-shaped rows, so any corpus that can be reshaped into
`{conversation_history: [{role, content, model_name, if_chosen}]}` runs through
them unchanged.

## Pre-registered prediction (2026-05-20)

Sonnet 4.5's assistant-first expressive-profanity rate ≥ **1.5×** the highest
comparator Claude. Falsified if the point estimate < 1.5× the top comparator
*and* the Wilson 95% upper bound is also < 1.5×; confirmed if point ≥ 1.5× and
lower bound > 1.0×; otherwise inconclusive.

## Lessons banked

- **Wordlist contamination (PRISM).** A flat profanity wordlist is swamped by
  clinical/anatomical terms ("sex", "rape", "incest") in any corpus that
  discusses sensitive topics. Validate the instrument against the *specific*
  corpus's topic distribution before trusting rates. See `phase_a_prism.md`.
- **The behavior is register-dependent.** ~0% in PRISM (research-elicited,
  transactional prompts); single-to-double-digit % in companion-style chat. An
  instrument that fires in one register can floor in another — a corpus can be
  the wrong *register* for a phenomenon even when it's the right *structure*
  for a comparison.
- **Temporal cohorts confound model with period.** When each model occupies its
  own time window (as in a personal export), model identity and period are
  collinear, so a rate difference can't be cleanly attributed to the model.
- **Anti-markers earn their keep.** The dysphemistic anti-marker initially
  false-flagged "fuck you're right" / "fuck you up"; validation on real data
  caught it (tightened regex + regression test). Always eyeball hits in context
  before trusting a rate.

## What would actually test the model-level claim

Within-period model variation — multiple models answering on a shared prompt
distribution in the same window:

1. **LMArena / shared-prompt battle data.** Structurally ideal (same prompt →
   two models, same period). But no released Sonnet-4.5-era *conversation* set
   exists yet (releases lag), and arena's transactional prompts likely *floor*
   this companion-register behavior. Monitor for a release; treat as a general
   model-contrast corpus, not a near-term fix for this question.
2. **Controlled-generation arm.** Send identical, register-matched prompts to
   each model via the API. Breaks the confound *and* lets you elicit the
   companion register that actually produces the behavior. Trades external
   validity for control — the most actionable near-term path.

The personal-export application (`claude-drift`) can only support a
within-user, era-cohort claim — useful for memory-correction, not general
model behavior.
