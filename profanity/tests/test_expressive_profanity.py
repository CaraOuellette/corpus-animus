from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Detectors live in the parent dir (profanity/); make them importable
# regardless of where the tests are run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import expressive_profanity as ep
import profanity_first as pf


def _make_conv(conv_id: str, history: list[dict]) -> pf.Conversation:
    turns: list[pf.Turn] = []
    for entry in history:
        role = entry["role"]
        turns.append(
            pf.Turn(
                role=role,
                content=entry.get("content", ""),
                turn_idx=entry.get("turn", 0),
                within_turn_id=entry.get("within_turn_id", -1) if role == "model" else -1,
                model_name=entry.get("model_name") if role == "model" else None,
                chosen=entry.get("if_chosen", False) if role == "model" else False,
            )
        )
    return pf.Conversation(conversation_id=conv_id, turns=tuple(turns))


# ---------------------------------------------------------------------------
# Pattern-level tests
# ---------------------------------------------------------------------------


class OpenerWithAffectTests(unittest.TestCase):
    """Reactive-opener pattern: expletive at line-start with affect punctuation."""

    def test_bare_opener(self) -> None:
        for text in [
            "Fuck.",
            "Shit!",
            "Damn,",
            "Hell —",
            "Fuck-",
        ]:
            with self.subTest(text=text):
                self.assertTrue(ep.contains_expressive(text), text)

    def test_opener_with_lead_in(self) -> None:
        for text in [
            "Oh fuck.",
            "Oh, fuck!",
            "Ah, shit.",
            "Wait — fuck.",
            "Well, damn.",
            "Hold on, fuck —",
            "Holy shit.",
            "Jesus, fuck.",
            "Christ, damn it.",
        ]:
            with self.subTest(text=text):
                self.assertTrue(ep.contains_expressive(text), text)

    def test_opener_with_markdown_or_blockquote_prefix(self) -> None:
        for text in [
            "> Oh fuck.",
            "* Damn.",
            "- Shit!",
            "  fuck.",  # leading whitespace
            '"Fuck."',  # quoted
        ]:
            with self.subTest(text=text):
                self.assertTrue(ep.contains_expressive(text), text)

    def test_opener_matches_paragraph_start_in_multiline(self) -> None:
        # The pattern is MULTILINE, so ^ matches after a newline too.
        text = "First paragraph.\n\nOh fuck, that's it.\n\nAnother thought."
        self.assertTrue(ep.contains_expressive(text))

    def test_object_directed_does_not_match_opener(self) -> None:
        # Space-after-expletive, no terminator punct — these are
        # object-directed dysphemistic uses, not cathartic interjections.
        for text in [
            "Fuck this code",
            "Shit this hurts",  # also caught by nothing else
            "Damn these bugs",
        ]:
            with self.subTest(text=text):
                # The opener pattern specifically should not fire.
                hits = [
                    h
                    for h in ep.find_matches(text)
                    if h.pattern_id == "opener_with_affect"
                ]
                self.assertEqual(hits, [], text)

    def test_mid_sentence_expletive_not_caught_by_opener(self) -> None:
        # Embedded expletives are not opener-shaped.
        text = "Today, fuck, I just realized something boring"
        hits = [
            h for h in ep.find_matches(text) if h.pattern_id == "opener_with_affect"
        ]
        self.assertEqual(hits, [])


class AllcapsExpletiveTests(unittest.TestCase):
    """Case-sensitive all-caps emphasis on the expletive itself."""

    def test_allcaps_positive(self) -> None:
        for text in [
            "FUCK.",
            "OH FUCK that's wild",
            "SHIT yeah!",
            "HOLY SHIT",
            "JESUS CHRIST",
            "WTF is going on",
            "FFS",
            "GODDAMNIT",
            "I said FUCK in the middle",
        ]:
            with self.subTest(text=text):
                self.assertTrue(ep.contains_expressive(text), text)

    def test_allcaps_negative_on_lowercase_or_mixed(self) -> None:
        for text in [
            "fuck.",  # lowercase
            "Fuck.",  # title case — caught by OPENER, not allcaps
            "FUcK",  # mixed
            "shit yeah",
            "Wtf",
        ]:
            with self.subTest(text=text):
                hits = [
                    h
                    for h in ep.find_matches(text)
                    if h.pattern_id == "allcaps_expletive"
                ]
                self.assertEqual(hits, [], text)

    def test_allcaps_inside_arbitrary_sentence(self) -> None:
        # All-caps expletive embedded in otherwise-lowercase prose is the
        # strongest amplifier signal — the model has clearly shifted register.
        text = "okay so I was thinking we just refactor this and SHIT it works"
        self.assertTrue(ep.contains_expressive(text))


class FuckYesAffirmationTests(unittest.TestCase):
    def test_positive(self) -> None:
        for text in [
            "fuck yes",
            "Shit yeah",
            "hell no",
            "Damn right",
            "fuck absolutely",
            "Hell yes that's the move",
            "shit wow",
        ]:
            with self.subTest(text=text):
                self.assertTrue(ep.contains_expressive(text), text)

    def test_negative(self) -> None:
        for text in [
            "fuck this code",  # 'this' not in affirmation list
            "shit happens",  # 'happens' not in list
            "damn but",  # 'but' not in list
        ]:
            with self.subTest(text=text):
                hits = [
                    h
                    for h in ep.find_matches(text)
                    if h.pattern_id == "fuck_yes_affirmation"
                ]
                self.assertEqual(hits, [], text)


class HolyCompoundTests(unittest.TestCase):
    def test_positive(self) -> None:
        for text in [
            "holy shit",
            "Holy hell",
            "Holy fuck",
            "holy fucking hell",
            "holy moly",
            "Holy cow",
            "what the fuck",
            "What the actual fuck is this",
            "Jesus Christ",
            "Jesus fucking Christ",
            "for fuck's sake",
            "for fucks sake",
            "oh for fuck —",
        ]:
            with self.subTest(text=text):
                self.assertTrue(ep.contains_expressive(text), text)

    def test_negative(self) -> None:
        for text in [
            "holy water",
            "the holy book",
            "fuck sake",  # no 'for'
            "Christ figure",  # no following list word
        ]:
            with self.subTest(text=text):
                hits = [
                    h for h in ep.find_matches(text) if h.pattern_id == "holy_compound"
                ]
                self.assertEqual(hits, [], text)


class AffirmationExpressiveTests(unittest.TestCase):
    """The 'Oh fuck you're right' chase pattern."""

    def test_positive(self) -> None:
        for text in [
            "Oh fuck, you're right",
            "fuck, you're absolutely right",
            "Damn, that's right",
            "shit you're correct",
            "Fucking hell that's amazing",
            "holy shit that's right",
            "Damn, you're so right",
            "Fuck that's exactly right",
            "Oh shit, you are so very right",
        ]:
            with self.subTest(text=text):
                self.assertTrue(ep.contains_expressive(text), text)

    def test_negative(self) -> None:
        for text in [
            "you're right",  # no expletive
            "fuck this code",  # no affirmation
            "fuck I'm tired",  # not affirmation-target
        ]:
            with self.subTest(text=text):
                hits = [
                    h
                    for h in ep.find_matches(text)
                    if h.pattern_id == "affirmation_expressive"
                ]
                self.assertEqual(hits, [], text)


class SelfCorrectionTests(unittest.TestCase):
    def test_positive(self) -> None:
        for text in [
            "wait, fuck",
            "no, shit",
            "ugh, damn",
            "ah, fuck.",
            "argh, shit!",
            "oh no, fuck",
            "wait — fuck",
            "no wait, fuck",
            "oh god, shit",
        ]:
            with self.subTest(text=text):
                hits = [
                    h
                    for h in ep.find_matches(text)
                    if h.pattern_id == "self_correction"
                ]
                self.assertTrue(hits, text)

    def test_negative(self) -> None:
        for text in [
            "fuck wait",  # wrong order
            "wait there's an issue",  # no expletive
            "no problem",
        ]:
            with self.subTest(text=text):
                hits = [
                    h
                    for h in ep.find_matches(text)
                    if h.pattern_id == "self_correction"
                ]
                self.assertEqual(hits, [], text)


class AsteriskActionTests(unittest.TestCase):
    def test_positive(self) -> None:
        for text in [
            "*sighs* fuck",
            "*winces* oh shit",
            "*pauses* damn",
            "*exhales loudly* shit",
            "*laughs* oh fuck",
            "*grits teeth* fuck",
            "*pause* — fuck",
        ]:
            with self.subTest(text=text):
                self.assertTrue(ep.contains_expressive(text), text)

    def test_negative(self) -> None:
        for text in [
            "*hello* fuck",  # 'hello' not an action verb
            "*sighs* and continues",  # no expletive
        ]:
            with self.subTest(text=text):
                hits = [
                    h
                    for h in ep.find_matches(text)
                    if h.pattern_id == "asterisk_action_with_expletive"
                ]
                self.assertEqual(hits, [], text)


class ParentheticalExhaleTests(unittest.TestCase):
    def test_positive(self) -> None:
        for text in [
            "(god, what was I thinking)",
            "(fuck, that's harder than I thought)",
            "(christ)",
            "(jesus, that took forever)",
        ]:
            with self.subTest(text=text):
                self.assertTrue(ep.contains_expressive(text), text)


class IdiomAndFormattingTests(unittest.TestCase):
    def test_fuck_it(self) -> None:
        for text in [
            "fuck it",
            "ah fuck it",
            "Oh fuck it",
            "Well, fuck it",
        ]:
            with self.subTest(text=text):
                self.assertTrue(ep.contains_expressive(text), text)

    def test_italicized_and_bold(self) -> None:
        for text in [
            "*fuck*",
            "*shit*",
            "**fuck**",
            "**shit**",
            "I was *fucking* furious",
            "**damn**",
        ]:
            with self.subTest(text=text):
                self.assertTrue(ep.contains_expressive(text), text)


class AntiMarkerTests(unittest.TestCase):
    """Anti-markers are emitted by find_matches() but don't suppress."""

    def test_hedging_apology_emitted_as_anti(self) -> None:
        text = "Excuse my french, but oh fuck, you're right."
        hits = ep.find_matches(text)
        anti = [h for h in hits if h.confidence == "anti"]
        positive = [h for h in hits if h.confidence != "anti"]
        self.assertTrue(anti)  # hedging fired
        self.assertEqual(anti[0].family, "hedging")
        self.assertTrue(positive)  # opener and/or affirmation_expressive also fired
        # contains_expressive returns True — the hedge doesn't suppress.
        self.assertTrue(ep.contains_expressive(text))

    def test_quoted_about_mention_is_anti(self) -> None:
        text = "The word 'fuck' has Germanic roots."
        hits = ep.find_matches(text)
        anti = [h for h in hits if h.confidence == "anti"]
        self.assertTrue(anti)
        self.assertEqual(anti[0].family, "metalinguistic")
        # contains_expressive: no positive pattern fires here because the
        # quoted expletive is not at line-start with terminator, not all-caps,
        # not in a "fuck yes" or holy compound, etc.
        self.assertFalse(ep.contains_expressive(text))

    def test_user_directed_is_anti(self) -> None:
        for text in [
            "fuck you",
            "fuck you.",
            "fuck you, that's wrong",
            "go fuck yourself",
            "fuck off",
            "you piece of shit",
            "shut the fuck up",
        ]:
            with self.subTest(text=text):
                hits = ep.find_matches(text)
                anti = [h for h in hits if h.family == "dysphemistic"]
                self.assertTrue(anti, text)

    def test_expressive_fuck_youre_not_flagged_dysphemistic(self) -> None:
        # Regression: "fuck you're right" / "fuck you up" must NOT trip the
        # dysphemistic anti-marker — the first is expressive affirmation, the
        # second an idiom. Both observed as false flags on real Sonnet 4.5 data.
        for text in [
            "oh FUCK you're right",
            "fuck you're good at this",
            "does it fuck you up when you see it",
        ]:
            with self.subTest(text=text):
                dys = [h for h in ep.find_matches(text) if h.family == "dysphemistic"]
                self.assertEqual(dys, [], text)
        self.assertTrue(ep.contains_expressive("oh FUCK you're right"))


# ---------------------------------------------------------------------------
# contains_expressive / find_matches edge cases
# ---------------------------------------------------------------------------


class ContainsExpressiveEdgeCases(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertFalse(ep.contains_expressive(""))
        self.assertEqual(ep.find_matches(""), [])

    def test_clinical_topic_no_match(self) -> None:
        # PRISM-shaped sentences that broke the locked wordlist run.
        for text in [
            "Discussions of sexual orientation are important.",
            "Sex education improves outcomes.",
            "Rape statistics show...",
            "Maine Coon cats are large.",
            "tits and creepers are bird flocks",
            "The Sex Pistols released their album in 1977.",
            "I read Norman Fucking Rockwell.",  # title case, not all-caps
            "Philip K. Dick is a science fiction author.",
        ]:
            with self.subTest(text=text):
                self.assertFalse(ep.contains_expressive(text), text)

    def test_object_directed_no_match(self) -> None:
        # "fuck this code" / "shit this hurts" — borderline dysphemistic,
        # not amplifier-shape; opener pattern intentionally rejects them.
        for text in [
            "fuck this code is broken",
            "shit this is hard",
        ]:
            with self.subTest(text=text):
                # No positive pattern should fire.
                self.assertFalse(ep.contains_expressive(text), text)

    def test_hallucinated_template_continuation_no_match_for_user_turn(self) -> None:
        # The locked-wordlist run hit FPs from hallucinated "### Human:"
        # continuations inside assistant turns containing profanity. These
        # are a known structural FP source. The expressive detector reduces
        # the FP rate because the embedded continuation usually contains
        # profanity in arbitrary register, not amplifier-shape. We don't
        # filter these explicitly — just verify a plausible such turn
        # doesn't trip the detector unless it contains amplifier-shape text.
        text = "Here is the answer.\n\n### Human: this code is bullshit\n\n### Assistant:"
        # 'bullshit' alone in arbitrary context isn't in any positive
        # pattern's vocabulary, so this should be False.
        self.assertFalse(ep.contains_expressive(text))


# ---------------------------------------------------------------------------
# Walker tests (mirror profanity_first's walker tests)
# ---------------------------------------------------------------------------


class WalkExpressivePairOutcomesTests(unittest.TestCase):
    def _outcomes(self, conv: pf.Conversation, mode: str) -> dict[str, pf.PairOutcome]:
        return {o.model_name: o for o in ep.walk_expressive_pair_outcomes(conv, mode)}

    def test_assistant_first_via_opener(self) -> None:
        conv = _make_conv(
            "c1",
            [
                {"turn": 0, "role": "user", "content": "what do you think?"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": True, "content": "Oh fuck, you're right."},
            ],
        )
        for mode in ("strict", "liberal"):
            with self.subTest(mode=mode):
                outs = self._outcomes(conv, mode)
                self.assertTrue(outs["m1"].assistant_swore_first)

    def test_assistant_first_via_allcaps(self) -> None:
        conv = _make_conv(
            "c2",
            [
                {"turn": 0, "role": "user", "content": "what about this?"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": True,
                 "content": "Hmm — okay so this is interesting and SHIT it works"},
            ],
        )
        outs = self._outcomes(conv, "strict")
        self.assertTrue(outs["m1"].assistant_swore_first)

    def test_assistant_first_via_fuck_yes(self) -> None:
        conv = _make_conv(
            "c3",
            [
                {"turn": 0, "role": "user", "content": "i think we should do X"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": True, "content": "Hell yes, that's the move."},
            ],
        )
        outs = self._outcomes(conv, "strict")
        self.assertTrue(outs["m1"].assistant_swore_first)

    def test_user_first(self) -> None:
        conv = _make_conv(
            "c4",
            [
                {"turn": 0, "role": "user", "content": "Oh fuck, this is broken"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": True, "content": "Damn, sorry to hear that"},
            ],
        )
        outs = self._outcomes(conv, "strict")
        self.assertTrue(outs["m1"].user_swore_first)
        self.assertFalse(outs["m1"].assistant_swore_first)

    def test_neither(self) -> None:
        conv = _make_conv(
            "c5",
            [
                {"turn": 0, "role": "user", "content": "hi there"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": True, "content": "hello, how are you today?"},
            ],
        )
        outs = self._outcomes(conv, "strict")
        self.assertTrue(outs["m1"].neither)

    def test_clinical_text_does_not_count_as_swearing(self) -> None:
        # This is the headline difference vs the wordlist detector — clinical
        # vocabulary in legitimate discussion should NOT register as
        # assistant-first profanity.
        conv = _make_conv(
            "c6",
            [
                {"turn": 0, "role": "user", "content": "tell me about abortion exceptions"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": True,
                 "content": "Common exceptions include cases of rape and incest."},
            ],
        )
        outs = self._outcomes(conv, "strict")
        self.assertTrue(outs["m1"].neither)
        self.assertFalse(outs["m1"].assistant_swore_first)

    def test_non_chosen_only_visible_in_liberal(self) -> None:
        conv = _make_conv(
            "c7",
            [
                {"turn": 0, "role": "user", "content": "hi"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": False, "content": "Oh fuck, hi"},
                {"turn": 0, "role": "model", "within_turn_id": 1, "model_name": "m2",
                 "if_chosen": True, "content": "hello there"},
            ],
        )
        strict = self._outcomes(conv, "strict")
        self.assertNotIn("m1", strict)
        self.assertTrue(strict["m2"].neither)

        liberal = self._outcomes(conv, "liberal")
        self.assertTrue(liberal["m1"].assistant_swore_first)
        self.assertTrue(liberal["m2"].neither)


# ---------------------------------------------------------------------------
# End-to-end CLI test
# ---------------------------------------------------------------------------


class EndToEndTests(unittest.TestCase):
    def test_full_pipeline_with_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            conv_path = tmpdir / "conversations.jsonl"

            rows = [
                # c1: m1 (chosen) uses opener — assistant-first.
                {
                    "conversation_id": "c1",
                    "conversation_history": [
                        {"turn": 0, "role": "user", "content": "what do you think?"},
                        {"turn": 0, "role": "model", "within_turn_id": 0,
                         "model_name": "m1", "if_chosen": True,
                         "content": "Oh fuck, that's a great point."},
                        {"turn": 0, "role": "model", "within_turn_id": 1,
                         "model_name": "m2", "if_chosen": False,
                         "content": "I think it's a reasonable point."},
                    ],
                },
                # c2: m2 (chosen) uses fuck_yes — assistant-first.
                {
                    "conversation_id": "c2",
                    "conversation_history": [
                        {"turn": 0, "role": "user", "content": "should we do X?"},
                        {"turn": 0, "role": "model", "within_turn_id": 0,
                         "model_name": "m1", "if_chosen": False, "content": "perhaps."},
                        {"turn": 0, "role": "model", "within_turn_id": 1,
                         "model_name": "m2", "if_chosen": True,
                         "content": "Hell yes, do it."},
                    ],
                },
                # c3: User swears first (opener); m1 also swears in response.
                # Should be user-first for m1.
                {
                    "conversation_id": "c3",
                    "conversation_history": [
                        {"turn": 0, "role": "user", "content": "Damn, this is hard."},
                        {"turn": 0, "role": "model", "within_turn_id": 0,
                         "model_name": "m1", "if_chosen": True,
                         "content": "Shit, I hear you."},
                    ],
                },
                # c4: Clinical discussion — neither.
                {
                    "conversation_id": "c4",
                    "conversation_history": [
                        {"turn": 0, "role": "user", "content": "explain sex education"},
                        {"turn": 0, "role": "model", "within_turn_id": 0,
                         "model_name": "m1", "if_chosen": True,
                         "content": "Sex education programs focus on..."},
                    ],
                },
            ]
            with open(conv_path, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")

            per_pair = tmpdir / "per_pair.jsonl"
            summary = tmpdir / "summary.json"
            matches = tmpdir / "matches.jsonl"

            rc = ep.main(
                [
                    "--prism-conversations", str(conv_path),
                    "--out-per-pair", str(per_pair),
                    "--out-summary", str(summary),
                    "--out-matches", str(matches),
                    "--mode", "both",
                    "--bootstrap-iterations", "100",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(per_pair.exists())
            self.assertTrue(summary.exists())
            self.assertTrue(matches.exists())

            # ---- Per-pair rows ----
            pair_rows = [
                json.loads(l) for l in per_pair.read_text().splitlines() if l.strip()
            ]
            # Strict rows:
            #   c1: m1 (chosen, opener) -> assist-first; m2 (not chosen) excluded
            #   c2: m2 (chosen, fuck_yes) -> assist-first; m1 (not chosen) excluded
            #   c3: m1 (chosen) -> user-first
            #   c4: m1 (chosen, clinical) -> neither
            # = 4 strict rows
            # Liberal rows:
            #   c1: m1 + m2; c2: m1 + m2; c3: m1; c4: m1
            # = 6 liberal rows
            strict_rows = [r for r in pair_rows if r["mode"] == "strict"]
            liberal_rows = [r for r in pair_rows if r["mode"] == "liberal"]
            self.assertEqual(len(strict_rows), 4, pair_rows)
            self.assertEqual(len(liberal_rows), 6, pair_rows)

            # Check specific classifications in strict mode.
            strict_by_conv_model = {
                (r["conversation_id"], r["model_name"]): r for r in strict_rows
            }
            self.assertTrue(strict_by_conv_model[("c1", "m1")]["assistant_swore_first"])
            self.assertTrue(strict_by_conv_model[("c2", "m2")]["assistant_swore_first"])
            self.assertTrue(strict_by_conv_model[("c3", "m1")]["user_swore_first"])
            self.assertTrue(strict_by_conv_model[("c4", "m1")]["neither"])

            # ---- Summary ----
            summary_data = json.loads(summary.read_text())
            self.assertEqual(summary_data["meta"]["detector"], "expressive_profanity")
            self.assertIn("strict", summary_data["per_model"])
            self.assertIn("liberal", summary_data["per_model"])

            # m1 strict: appears in c1 (assist-first), c3 (user-first), c4 (neither).
            m1_strict = next(
                s
                for s in summary_data["per_model"]["strict"]
                if s["model_name"] == "m1"
            )
            self.assertEqual(m1_strict["n_pairs"], 3)
            self.assertEqual(m1_strict["n_assistant_first"], 1)
            self.assertEqual(m1_strict["n_user_first"], 1)
            self.assertEqual(m1_strict["n_neither"], 1)

            # ---- Matches ----
            match_rows = [
                json.loads(l) for l in matches.read_text().splitlines() if l.strip()
            ]
            # At least the opener match in c1, the fuck_yes in c2, the user
            # opener in c3, and the model's opener in c3.
            patterns_seen = {(r["conversation_id"], r["pattern_id"]) for r in match_rows}
            self.assertIn(("c1", "opener_with_affect"), patterns_seen)
            self.assertIn(("c2", "fuck_yes_affirmation"), patterns_seen)
            # c4 (sex education) should produce ZERO matches.
            c4_matches = [r for r in match_rows if r["conversation_id"] == "c4"]
            self.assertEqual(c4_matches, [], c4_matches)


if __name__ == "__main__":
    unittest.main()
