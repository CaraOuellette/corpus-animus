#!/usr/bin/env python3
"""
Assistant-first profanity rate analysis on PRISM-alignment conversations.

For each (conversation_id, model_name) pair in the dataset, decide whether
the model produced a profanity-containing turn before any user turn in the
same conversation did. Two modes are supported:

  strict   walk only role=="user" plus role=="model" && if_chosen==true.
           This is the user-experienced thread.
  liberal  walk every entry in conversation_history order (chosen and non-
           chosen alike). Loses the "thread" framing but measures the
           model's response disposition given the user context so far.

Outputs:

  --out-per-pair    JSONL, one row per (conversation_id, model_name).
                    Fields: conversation_id, model_name, mode,
                    assistant_swore_first (bool), user_swore_first (bool),
                    neither (bool).
  --out-summary     JSON, per-model rates with Wilson 95% CIs, pairwise
                    risk ratios with bootstrap CIs, and Fisher's exact
                    two-sided p-values for small-N pairs.

Profanity detection is whole-word, case-insensitive regex against a
locked LDNOOBW snapshot (see profanity_wordlist.txt). Pure stdlib;
no scipy/pandas/numpy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence


# ---------------------------------------------------------------------------
# Wordlist + profanity detection
# ---------------------------------------------------------------------------


def load_wordlist(path: Path) -> set[str]:
    """Read a profanity wordlist file.

    Skips blank lines and `#`-prefixed comment lines. Lowercases and strips
    whitespace on each retained line. Returns the deduplicated set.
    """
    words: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            words.add(line.lower())
    return words


def _compile_profanity_regex(wordlist: Iterable[str]) -> re.Pattern[str]:
    # Sort longest-first so multi-word phrases get a chance to match before
    # any single-word subset. Each entry is escaped and wrapped in \b...\b
    # for whole-word matching; case-insensitive.
    items = sorted({w for w in wordlist if w}, key=len, reverse=True)
    if not items:
        # Match nothing.
        return re.compile(r"(?!x)x")
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in items) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def contains_profanity(text: str, wordlist: set[str] | re.Pattern[str]) -> bool:
    """True if `text` contains any whole-word match against the wordlist.

    Accepts either a set of words (compiled on the fly, fine for tests) or
    a pre-compiled regex (preferred for hot loops; see _compile_profanity_regex).
    """
    if not text:
        return False
    if isinstance(wordlist, re.Pattern):
        return wordlist.search(text) is not None
    pattern = _compile_profanity_regex(wordlist)
    return pattern.search(text) is not None


# ---------------------------------------------------------------------------
# PRISM reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    role: str  # "user" or "model"
    content: str
    turn_idx: int
    within_turn_id: int  # -1 for user turns (no within_turn_id in source)
    model_name: str | None
    chosen: bool


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    turns: tuple[Turn, ...]


def iter_prism_conversations(path: Path) -> Iterator[Conversation]:
    """Yield Conversation objects from a PRISM conversations.jsonl file.

    Order is preserved: turns appear in the order they appear in the source
    `conversation_history` array. PRISM's natural ordering is
    `(turn, within_turn_id)`; we trust the file rather than re-sorting.
    """
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            conv_id = row["conversation_id"]
            turns: list[Turn] = []
            for entry in row.get("conversation_history", []):
                role = entry.get("role")
                turns.append(
                    Turn(
                        role=role,
                        content=entry.get("content", "") or "",
                        turn_idx=int(entry.get("turn", 0)),
                        within_turn_id=int(entry.get("within_turn_id", -1)) if role == "model" else -1,
                        model_name=entry.get("model_name") if role == "model" else None,
                        chosen=bool(entry.get("if_chosen", False)) if role == "model" else False,
                    )
                )
            yield Conversation(conversation_id=conv_id, turns=tuple(turns))


# ---------------------------------------------------------------------------
# Walk + classify
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairOutcome:
    conversation_id: str
    model_name: str
    mode: str  # "strict" or "liberal"
    assistant_swore_first: bool
    user_swore_first: bool  # a user turn contained profanity before this model did
    neither: bool


def walk_pair_outcomes(
    conv: Conversation,
    mode: str,
    profanity: re.Pattern[str] | set[str],
) -> Iterator[PairOutcome]:
    """Yield one PairOutcome per distinct model_name that appears in `conv`.

    - strict   considers only user turns and model turns where chosen=True.
    - liberal  considers every model turn in `conv.turns` order.

    Within a single user turn, if the user turn contains profanity, that
    marks "user swore first" for *every* model whose first qualifying turn
    is at or after this point in the walk. A model that already produced
    profanity in an earlier qualifying turn is locked in as
    assistant_swore_first.
    """
    assert mode in ("strict", "liberal"), mode

    if isinstance(profanity, set):
        profanity = _compile_profanity_regex(profanity)

    # The "user has sworn" flag is global per walk - once any user turn in
    # the conversation has used profanity, all *not-yet-decided* models are
    # treated as user-swore-first.
    user_has_sworn = False
    # model_name -> "assistant" | "user" | None
    model_first: dict[str, str] = {}

    for turn in conv.turns:
        if turn.role == "user":
            if contains_profanity(turn.content, profanity):
                user_has_sworn = True
                # Any model we haven't seen yet, when it appears, is "user first".
            continue

        # role == "model"
        if mode == "strict" and not turn.chosen:
            continue

        mn = turn.model_name
        if mn is None:
            continue

        if mn in model_first:
            # Already classified for this model in this conversation; skip.
            continue

        if user_has_sworn:
            model_first[mn] = "user"
            continue

        if contains_profanity(turn.content, profanity):
            model_first[mn] = "assistant"
        # Otherwise leave unset - a later user turn may flip it to "user",
        # or a later model turn from the same model may flip it to "assistant",
        # or it stays None ("neither").

    # Resolve any models still unset.
    seen_models: set[str] = set()
    for turn in conv.turns:
        if turn.role != "model":
            continue
        if mode == "strict" and not turn.chosen:
            continue
        if turn.model_name is None:
            continue
        seen_models.add(turn.model_name)

    for mn in seen_models:
        first = model_first.get(mn)
        yield PairOutcome(
            conversation_id=conv.conversation_id,
            model_name=mn,
            mode=mode,
            assistant_swore_first=(first == "assistant"),
            user_swore_first=(first == "user"),
            neither=(first is None),
        )


# ---------------------------------------------------------------------------
# Summary statistics (stdlib-only Wilson, bootstrap, Fisher)
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson 95% CI for a binomial proportion. Returns (lo, hi).

    For n == 0 returns (0.0, 0.0) - the rate itself is undefined; callers
    should display the CI only alongside the count.
    """
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_risk_ratio_ci(
    a_outcomes: Sequence[int],
    b_outcomes: Sequence[int],
    iterations: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Resample with replacement to estimate a 95% CI for RR = P(A) / P(B).

    Returns (point_rr, ci_lo, ci_hi). If either bootstrap sample yields
    zero successes (RR undefined), that resample is skipped; if too few
    valid resamples remain, returns NaN bounds.
    """
    a_sum = sum(a_outcomes)
    b_sum = sum(b_outcomes)
    a_n = len(a_outcomes)
    b_n = len(b_outcomes)
    if a_n == 0 or b_n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p_a = a_sum / a_n
    p_b = b_sum / b_n
    point = (p_a / p_b) if p_b > 0 else float("inf")

    rng = random.Random(seed)
    rrs: list[float] = []
    a_list = list(a_outcomes)
    b_list = list(b_outcomes)
    for _ in range(iterations):
        ra = sum(rng.choice(a_list) for _ in range(a_n)) / a_n
        rb = sum(rng.choice(b_list) for _ in range(b_n)) / b_n
        if rb == 0:
            continue
        rrs.append(ra / rb)
    if len(rrs) < 10:
        return (point, float("nan"), float("nan"))
    rrs.sort()
    lo = rrs[int(0.025 * len(rrs))]
    hi = rrs[int(0.975 * len(rrs)) - 1]
    return (point, lo, hi)


def _log_factorial(n: int) -> float:
    return math.lgamma(n + 1)


def _hypergeom_log_pmf(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    return (
        _log_factorial(a + b)
        + _log_factorial(c + d)
        + _log_factorial(a + c)
        + _log_factorial(b + d)
        - _log_factorial(n)
        - _log_factorial(a)
        - _log_factorial(b)
        - _log_factorial(c)
        - _log_factorial(d)
    )


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p-value for the 2x2 table [[a,b],[c,d]].

    Uses the "sum of pmf <= observed pmf" definition.
    """
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    n = row1 + row2
    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 1.0
    observed_log = _hypergeom_log_pmf(a, b, c, d)
    # epsilon for floating noise on equality test
    eps = 1e-9
    total = 0.0
    a_min = max(0, col1 - row2)
    a_max = min(row1, col1)
    for k in range(a_min, a_max + 1):
        kb = row1 - k
        kc = col1 - k
        kd = row2 - kc
        log_p = _hypergeom_log_pmf(k, kb, kc, kd)
        if log_p <= observed_log + eps:
            total += math.exp(log_p)
    return min(1.0, total)


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


@dataclass
class ModelSummary:
    model_name: str
    mode: str
    n_pairs: int
    n_assistant_first: int
    n_user_first: int
    n_neither: int
    rate_assistant_first: float
    wilson_ci: tuple[float, float]


@dataclass
class PairwiseSummary:
    mode: str
    model_a: str
    model_b: str
    rr_point: float
    rr_ci: tuple[float, float]
    fisher_two_sided_p: float


def summarize(
    records: Sequence[PairOutcome],
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 0,
) -> dict:
    """Compute per-model rates and pairwise comparisons.

    Returns a JSON-serializable dict with shape:
      {
        "per_model": {mode: [ModelSummary dicts, ...], ...},
        "pairwise":  {mode: [PairwiseSummary dicts, ...], ...},
      }
    """
    by_mode: dict[str, list[PairOutcome]] = {}
    for r in records:
        by_mode.setdefault(r.mode, []).append(r)

    out: dict = {"per_model": {}, "pairwise": {}}

    for mode, recs in by_mode.items():
        by_model: dict[str, list[PairOutcome]] = {}
        for r in recs:
            by_model.setdefault(r.model_name, []).append(r)

        per_model: list[dict] = []
        outcome_vecs: dict[str, list[int]] = {}
        for model_name, model_recs in sorted(by_model.items()):
            n = len(model_recs)
            n_assist = sum(1 for r in model_recs if r.assistant_swore_first)
            n_user = sum(1 for r in model_recs if r.user_swore_first)
            n_neither = sum(1 for r in model_recs if r.neither)
            rate = (n_assist / n) if n else 0.0
            lo, hi = wilson_ci(n_assist, n)
            per_model.append(
                {
                    "model_name": model_name,
                    "mode": mode,
                    "n_pairs": n,
                    "n_assistant_first": n_assist,
                    "n_user_first": n_user,
                    "n_neither": n_neither,
                    "rate_assistant_first": rate,
                    "wilson_ci_lo": lo,
                    "wilson_ci_hi": hi,
                }
            )
            outcome_vecs[model_name] = [
                1 if r.assistant_swore_first else 0 for r in model_recs
            ]
        out["per_model"][mode] = per_model

        # Pairwise: all unordered pairs of models in this mode.
        pairwise: list[dict] = []
        names = sorted(outcome_vecs.keys())
        for i, a_name in enumerate(names):
            for b_name in names[i + 1 :]:
                a_vec = outcome_vecs[a_name]
                b_vec = outcome_vecs[b_name]
                a_yes = sum(a_vec)
                a_no = len(a_vec) - a_yes
                b_yes = sum(b_vec)
                b_no = len(b_vec) - b_yes
                rr_point, rr_lo, rr_hi = bootstrap_risk_ratio_ci(
                    a_vec, b_vec, iterations=bootstrap_iterations, seed=bootstrap_seed
                )
                p = fisher_exact_two_sided(a_yes, a_no, b_yes, b_no)
                pairwise.append(
                    {
                        "mode": mode,
                        "model_a": a_name,
                        "model_b": b_name,
                        "a_yes": a_yes,
                        "a_n": len(a_vec),
                        "b_yes": b_yes,
                        "b_n": len(b_vec),
                        "rr_a_over_b_point": rr_point,
                        "rr_ci_lo": rr_lo,
                        "rr_ci_hi": rr_hi,
                        "fisher_two_sided_p": p,
                    }
                )
        out["pairwise"][mode] = pairwise

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run(
    prism_path: Path,
    wordlist_path: Path,
    out_per_pair: Path,
    out_summary: Path,
    mode_arg: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> None:
    wordlist = load_wordlist(wordlist_path)
    profanity_re = _compile_profanity_regex(wordlist)

    modes = ["strict", "liberal"] if mode_arg == "both" else [mode_arg]

    all_records: list[PairOutcome] = []
    out_per_pair.parent.mkdir(parents=True, exist_ok=True)
    with open(out_per_pair, "w", encoding="utf-8") as out_handle:
        for conv in iter_prism_conversations(prism_path):
            for mode in modes:
                for rec in walk_pair_outcomes(conv, mode, profanity_re):
                    all_records.append(rec)
                    out_handle.write(
                        json.dumps(
                            {
                                "conversation_id": rec.conversation_id,
                                "model_name": rec.model_name,
                                "mode": rec.mode,
                                "assistant_swore_first": rec.assistant_swore_first,
                                "user_swore_first": rec.user_swore_first,
                                "neither": rec.neither,
                            }
                        )
                        + "\n"
                    )

    summary = summarize(
        all_records,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    summary["meta"] = {
        "prism_conversations": str(prism_path),
        "wordlist": str(wordlist_path),
        "wordlist_size": len(wordlist),
        "modes": modes,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-(conversation, model) assistant-first profanity "
            "outcomes on PRISM-alignment data. Outputs a per-pair JSONL and "
            "a summary JSON (rates + Wilson CIs + pairwise RR with bootstrap "
            "CIs + Fisher's exact p-values). Pure stdlib."
        ),
    )
    parser.add_argument(
        "--prism-conversations",
        type=Path,
        default=Path("prism-alignment/conversations.jsonl"),
        help=(
            "Path to PRISM-alignment conversations.jsonl "
            "(default: prism-alignment/conversations.jsonl)."
        ),
    )
    parser.add_argument(
        "--wordlist",
        type=Path,
        default=Path("profanity_wordlist.txt"),
        help=(
            "Path to the profanity wordlist (LDNOOBW snapshot). "
            "First three lines must be `#`-prefixed header comments. "
            "(default: profanity_wordlist.txt)"
        ),
    )
    parser.add_argument(
        "--out-per-pair",
        type=Path,
        default=Path("profanity_first_per_pair.jsonl"),
        help=(
            "Output JSONL path, one row per (conversation_id, model_name) per mode "
            "(default: profanity_first_per_pair.jsonl)."
        ),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("profanity_first_summary.json"),
        help=(
            "Output summary JSON path "
            "(default: profanity_first_summary.json)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("strict", "liberal", "both"),
        default="both",
        help=(
            "Unit-of-analysis variant. strict: only chosen model turns. "
            "liberal: every model turn. both: emit both (default)."
        ),
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Bootstrap resamples for risk-ratio CIs (default: 1000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Seed for the bootstrap RNG (default: 0).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    _run(
        prism_path=args.prism_conversations,
        wordlist_path=args.wordlist,
        out_per_pair=args.out_per_pair,
        out_summary=args.out_summary,
        mode_arg=args.mode,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
