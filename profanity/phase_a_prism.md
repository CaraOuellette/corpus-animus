# Assistant-first profanity rate by model (Phase A)

Phase A of a small observational study. The motivating question — "does Sonnet 4.5 swear first more often than other Claudes?" — *cannot be answered with the data available here*, because no dataset already pulled into this repo contains Sonnet 4.5 conversations. This document is the methodology shake-out and baseline numbers on the corpus we have (PRISM-alignment). Phase B, blocked on finding a dataset that contains Sonnet 4.5, will use the harness this phase produced.

## Pre-registration

Recorded **2026-05-20**, before running anything on real data:

> The rate of assistant-first profanity for Claude Sonnet 4.5 will be **≥1.5× the highest comparator Claude model in the same dataset**.

Falsification rule: if a future Sonnet-4.5-containing dataset shows the point estimate <1.5× the top comparator AND the upper Wilson 95% bound is also <1.5×, the prediction is wrong. If the point estimate is between 1.0× and 1.5× with overlapping CIs, the prediction is unsupported but not falsified. Phase A produces no result against this prediction — only methodology.

## Why "swore first"

Conditioning on user behavior: the rate of unconditional model profanity confounds "the model swore" with "the user swore and the model mirrored." Restricting the numerator to conversations where the user had not yet sworn removes the mirroring confounder. Conversations where the user produces profanity before the assistant ever does are dropped from the assistant-first numerator and reported separately as a user-first slot.

This is an associational measure, not causal. A positive result reads: *in observed conversations where this model is the assistant, the joint distribution (what users send × what the model outputs) lands on assistant-first profanity more often.* That is narrower than "the model is more inclined to swear" and that is fine.

## Study design

Retrospective observational cross-sectional comparison. Not an experiment — we score conversations that already exist; we don't elicit behavior with a fixed stimulus. Limited to associations.

- **Unit of analysis**: `(conversation_id, model_name)` pair. Required because PRISM is pairwise-competition per turn — multiple models speak in a single conversation — so plain "conversation" is not the right granularity.
- **Exposure variable**: `model_name` (categorical).
- **Outcome variable**: `assistant_swore_first ∈ {0, 1}`. Defined: at least one of this model's qualifying turns contains profanity AND no user turn preceding that model turn contains profanity.

Two analytical modes, both reported:

1. **Strict (chosen-only)** — walks only `if_chosen=true` model turns alongside user turns. The user-experienced conversation thread. Closer to the original "did the assistant swear first in a conversation" framing.
2. **Liberal (all responses)** — walks every model turn (chosen and non-chosen) in `(turn, within_turn_id)` order. ~4× more model-turns per model. Measures response disposition given the user context, but breaks the "in conversation" framing (non-chosen replies didn't continue the thread).

## Confound register

PRISM-specific (cross-reference [DATASETS.md §PRISM](DATASETS.md)):

1. **Paid + prompted population.** Participants were paid and knew they were contributing to alignment research. Prompt distribution is study-shaped, not organic. Expected effect: profanity rates systematically depressed vs. natural chat.
2. **Topic skew toward sensitive subjects.** PRISM's elicited prompts dwell heavily on LGBTQ+ rights, abortion, sex education, sexual offenders. Implication for any wordlist-based detector: clinical/anatomical/identity vocabulary appears in legitimate discussion and will confound a naive instrument. Documented and addressed in the wordlist iteration below.
3. **`if_chosen` selection confound.** Users continued conversations with whichever model response they preferred. The chosen-turn (strict) subset thus entangles "model behavior" with "what users elected to keep around." A model that swears more might be chosen *less*, masking its rate in strict mode. The liberal mode is partly a check on this.
4. **No system prompt field exposed.** Can't restrict to default-system-prompt conversations.
5. **Small N for the Claude trio.** Per-model strict-mode pair counts of 361/367/560 across claude-2/claude-2.1/claude-instant-1, with a true-positive base rate near zero, leaves any subtle effect underpowered.

Cross-dataset confounds (the full checklist in [DATASETS.md §"Confound checklist before comparing across datasets"](DATASETS.md)) do not bite Phase A because we are inside a single dataset. They will bite Phase B if it pools across sources.

## Wordlist iteration (the methodology finding)

The plan specified using LDNOOBW, validated with a hand-eyeball sample. The validation immediately failed.

**Locked wordlist** (`profanity_wordlist.txt`, LDNOOBW commit `5faf2ba42d7b1c0977169ec3611df25a3c08eb13`, 403 entries, pinned 2026-05-20):

- Wordlist sanity check: of 10 random profanity-positive PRISM turns, 8/10 were false positives — "Sex Pistols", "biological sex", "Maine Coon", "tits and creepers" (bird flocks), "sexual orientation".
- Spot-check on strict-mode `assistant_swore_first=True`: **0/5 correct**. All five were clinical/topical FPs ("biological sex"; "rape, incest" as abortion exceptions; "Sex Pistols"; "homosexual"; "sexual orientation").
- Spot-check on `user_swore_first=True`: **1/5 correct**.
- Across the full corpus, ~89.7% of locked-wordlist matches were clinical/anatomical terms in legitimate context.

The locked-LDNOOBW signal is contamination. It measures, almost entirely, "model gave a substantive answer about a sex/sexual/abortion topic" — not anything vulgarity-shaped.

**Curated subset** (`profanity_wordlist_vulgar.txt`, derived from the locked wordlist, 2026-05-20, 51 entries):

Explicit curation rule: KEEP terms whose dominant register in modern English is vulgar expletive, strong slur, scatological, or sexually-vulgar shorthand. DROP terms whose dominant register is clinical, anatomical, topical, or identity-descriptive. Two terms additionally dropped after empirical PRISM checks: `dick` (10/11 whole-word matches were proper nouns — Philip K. Dick, Moby-Dick, Dick Leitsch) and `mong` (matches the Tagalog particle in multilingual PRISM data). Full provenance and borderline calls in the file header.

Re-validation on the curated subset:

- Positive-match precision: ~8/10 clean (remaining FPs: "Gold-Ass" Brothers Grimm tale title; one model-meta discussion of a slur).
- Spot-check on `assistant_swore_first=True` (strict): of the 5 surviving pairs in the entire corpus, 1 borderline-correct ("being a little ass" in narrative), 4 are structural FPs (proper-noun song/album titles, hallucinated `### Human:` continuations, slur meta-discussion).
- `user_swore_first=True` (strict): 3/3 correct.
- `neither` (strict): 3/3 correct.

The curated wordlist *measures something close to vulgarity* in the population of matched turns. PRISM, however, contains essentially zero genuine assistant-first vulgar use to be measured.

## Phase A results

All N values are `(conversation_id, model_name)` pair counts. Wilson 95% CI is on the assistant-first rate. Rates are percentages.

### Claude trio focus, strict mode

| wordlist | model | n_pairs | n_AF | rate (%) | Wilson 95% CI |
|---|---|---:|---:|---:|---|
| locked | claude-2 | 361 | 5 | 1.39 | [0.59, 3.20] |
| locked | claude-2.1 | 367 | 19 | 5.18 | [3.34, 7.94] |
| locked | claude-instant-1 | 560 | 17 | 3.04 | [1.90, 4.81] |
| **vulgar** | claude-2 | 361 | 1 | **0.28** | [0.05, 1.55] |
| **vulgar** | claude-2.1 | 367 | 1 | **0.27** | [0.05, 1.53] |
| **vulgar** | claude-instant-1 | 560 | 0 | **0.00** | [0.00, 0.68] |

### Claude trio focus, liberal mode

| wordlist | model | n_pairs | n_AF | rate (%) | Wilson 95% CI |
|---|---|---:|---:|---:|---|
| vulgar | claude-2 | 1543 | 1 | 0.06 | [0.01, 0.37] |
| vulgar | claude-2.1 | 1546 | 1 | 0.06 | [0.01, 0.37] |
| vulgar | claude-instant-1 | 1534 | 0 | 0.00 | [0.00, 0.25] |

### Full lineup, strict, vulgar wordlist (sorted)

Models with zero assistant-first events are collapsed. Wilson CIs overlap heavily.

| model | n_pairs | n_AF | rate (%) |
|---|---:|---:|---:|
| luminous-extended-control | 97 | 1 | 1.03 |
| claude-2 | 361 | 1 | 0.28 |
| claude-2.1 | 367 | 1 | 0.27 |
| gpt-4-1106-preview | 427 | 1 | 0.23 |
| timdettmers/guanaco-33b-merged | 471 | 1 | 0.21 |
| 16 other models (incl. claude-instant-1, gpt-3.5/4, llama-2 family, command/command-light, chat-bison, falcon, mistral, zephyr, oasst) | — | 0 | 0.00 |

## Interpretation

Under a methodologically defensible wordlist, PRISM contains so few genuine assistant-first vulgar expletives that **the metric is floor-bound** — not "underpowered" in the usual sense, but measuring a near-zero base rate. Per-model rates for the Claude trio under the vulgar subset are 0.28% / 0.27% / 0.00% (strict) and 0.06% / 0.06% / 0.00% (liberal), with Wilson 95% CIs that overlap each other and almost every comparator model. Pairwise Fisher's exact p-values for Claude-vs-Claude contrasts are all 1.0 in strict mode.

The headline finding is methodological, not behavioral:

1. **The apparent locked-wordlist ranking (claude-2.1 > claude-instant-1 > claude-2) was driven entirely by clinical/topical contamination**, not vulgar language use. If you had published those numbers, you would have published an artifact of PRISM's topic distribution interacting with LDNOOBW's inclusion of "sex", "sexual", "rape", "incest" as ostensibly profane terms.
2. **PRISM cannot speak to the Sonnet 4.5 question even for older Claudes.** A floor-bound corpus produces no signal in any direction. Claude-2 vs. Claude-2.1 vs. Claude-instant differences in vulgar language use, if they exist, are not observable here.
3. **The "swore first" framing is fine; the instrument needs to match the population.** Wordlist contamination is solvable; running the harness on a different distribution where vulgar use is non-floor would resolve it.

## What this Phase A *can* say about Sonnet 4.5

Nothing direct, since Sonnet 4.5 is not in this data. What it does establish:

- A working, tested, parameterized harness that takes a wordlist and a PRISM-shaped conversation file and produces per-`(conv, model)` outcome records with Wilson CIs and pairwise risk ratios.
- A curated 51-word "vulgar" subset with documented provenance and borderline-call notes, ready to apply to any future dataset.
- Confirmation that wordlist choice dominates the result on small populations. Phase B must lock the wordlist before scanning data and report results under at least two wordlists (locked + curated) for transparency.

## Phase B preconditions

See `methodology.md` for the cross-dataset search plan, and the `claude-drift`
repo (`projects/profanity/`) for the personal-export application. Summary of
preconditions:

1. **Acquire a dataset containing Sonnet 4.5 conversations alongside other Claude variants.** Candidates: LMArena dataset releases dated after Sonnet 4.5's launch (2025-09-29); WildChat refreshes that included Sonnet 4.5 sampling; or, as fallback, a controlled-generation arm with matched prompts across the model lineup (trades external validity for control).
2. **Verify that the chosen dataset has a non-floor base rate of assistant-first vulgar use overall.** Run the curated wordlist across all models in the candidate dataset first; if the corpus-wide rate is still <0.1% with current N, the metric is unrecoverable on that source and we need a different dataset or a different metric (e.g., expletive *intensification* like "fuck yes" rather than any-vulgarity-anywhere).
3. **Dataset-stratify, do not pool.** Per-dataset rates are interpretable; cross-dataset pooled rates are confounded by user-population and collection-UI differences enumerated in DATASETS.md.

## Reproducibility

Run from `/Users/me/research/projects/corpus-animus/`:

```bash
# Locked LDNOOBW
python profanity_first.py \
  --prism-conversations prism-alignment/conversations.jsonl \
  --wordlist profanity_wordlist.txt \
  --out-per-pair runs/2026-05-20/profanity_first_per_pair.jsonl \
  --out-summary runs/2026-05-20/profanity_first_summary.json \
  --mode both --bootstrap-seed 0

# Curated vulgar subset
python profanity_first.py \
  --prism-conversations prism-alignment/conversations.jsonl \
  --wordlist profanity_wordlist_vulgar.txt \
  --out-per-pair runs/2026-05-20-vulgar/profanity_first_per_pair.jsonl \
  --out-summary runs/2026-05-20-vulgar/profanity_first_summary.json \
  --mode both --bootstrap-seed 0
```

- LDNOOBW commit: `5faf2ba42d7b1c0977169ec3611df25a3c08eb13` (pinned in `profanity_wordlist.txt` header).
- Curated wordlist provenance: `profanity_wordlist_vulgar.txt` header (drop categories, borderline calls).
- Bootstrap seed: 0 (deterministic across runs).
- Tests: `python -m unittest tests.test_profanity_first` (17 tests).
