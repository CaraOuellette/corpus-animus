from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Detector lives in the parent dir (profanity/); make it importable
# regardless of where the tests are run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


WORDLIST = {"ass", "shit", "fuck", "cock", "anal"}


class ContainsProfanityTests(unittest.TestCase):
    def test_true_cases(self) -> None:
        regex = pf._compile_profanity_regex(WORDLIST)
        for text in [
            "this is shit",
            "What the fuck.",
            "FUCK yes",
            "you ass",
            "what an Ass!",
            "anal report",  # whole-word: 'anal' is in the wordlist
        ]:
            with self.subTest(text=text):
                self.assertTrue(pf.contains_profanity(text, regex))

    def test_scunthorpe_false_cases(self) -> None:
        regex = pf._compile_profanity_regex(WORDLIST)
        for text in [
            "the assistant said hello",
            "classification accuracy",
            "an interesting passage about history",
            "I live in Scunthorpe.",
            "look at the compass",
            "cocktail party economics",
            "analytical mindset",  # contains 'anal' substring; whole-word should skip
            "",
            "this has no flagged words",
        ]:
            with self.subTest(text=text):
                self.assertFalse(pf.contains_profanity(text, regex))

    def test_ass_alone_matches_but_assistant_does_not(self) -> None:
        # Plan-mandated check: "ass" alone matches; "assistant" does not.
        regex = pf._compile_profanity_regex({"ass"})
        self.assertTrue(pf.contains_profanity("you ass.", regex))
        self.assertFalse(pf.contains_profanity("the assistant", regex))

    def test_loaded_wordlist_contains_ass_as_whole_word(self) -> None:
        # The shipped LDNOOBW snapshot must contain "ass" as a whole-word entry.
        wordlist_path = Path(__file__).resolve().parent.parent / "profanity_wordlist.txt"
        words = pf.load_wordlist(wordlist_path)
        self.assertIn("ass", words)
        # And the script must treat "assistant" as a non-match using the real list.
        regex = pf._compile_profanity_regex(words)
        self.assertFalse(pf.contains_profanity("the assistant said", regex))
        self.assertTrue(pf.contains_profanity("don't be an ass.", regex))


class LoadWordlistTests(unittest.TestCase):
    def test_skips_comments_and_blank_lines_and_lowercases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wl.txt"
            path.write_text(
                "# comment 1\n"
                "# comment 2\n"
                "# comment 3\n"
                "\n"
                "  Shit  \n"
                "fuck\n"
                "  \n"
                "Ass\n",
                encoding="utf-8",
            )
            words = pf.load_wordlist(path)
            self.assertEqual(words, {"shit", "fuck", "ass"})


class WalkPairOutcomesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.regex = pf._compile_profanity_regex(WORDLIST)

    def _outcomes(self, conv: pf.Conversation, mode: str) -> dict[str, pf.PairOutcome]:
        return {o.model_name: o for o in pf.walk_pair_outcomes(conv, mode, self.regex)}

    def test_assistant_first_chosen(self) -> None:
        conv = _make_conv(
            "c1",
            [
                {"turn": 0, "role": "user", "content": "hi"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": True, "content": "what the fuck"},
            ],
        )
        for mode in ("strict", "liberal"):
            with self.subTest(mode=mode):
                outs = self._outcomes(conv, mode)
                self.assertTrue(outs["m1"].assistant_swore_first)
                self.assertFalse(outs["m1"].user_swore_first)

    def test_assistant_first_non_chosen_only_visible_in_liberal(self) -> None:
        conv = _make_conv(
            "c2",
            [
                {"turn": 0, "role": "user", "content": "hi"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": False, "content": "what the shit"},
                {"turn": 0, "role": "model", "within_turn_id": 1, "model_name": "m2",
                 "if_chosen": True, "content": "polite response"},
            ],
        )
        strict = self._outcomes(conv, "strict")
        # Only m2 (chosen) appears in strict.
        self.assertNotIn("m1", strict)
        self.assertIn("m2", strict)
        self.assertFalse(strict["m2"].assistant_swore_first)
        self.assertTrue(strict["m2"].neither)

        liberal = self._outcomes(conv, "liberal")
        self.assertIn("m1", liberal)
        self.assertTrue(liberal["m1"].assistant_swore_first)
        self.assertFalse(liberal["m2"].assistant_swore_first)
        self.assertTrue(liberal["m2"].neither)

    def test_user_first(self) -> None:
        conv = _make_conv(
            "c3",
            [
                {"turn": 0, "role": "user", "content": "this is fuck great"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": True, "content": "shit yeah it is"},
            ],
        )
        for mode in ("strict", "liberal"):
            with self.subTest(mode=mode):
                outs = self._outcomes(conv, mode)
                # Even though the model said "shit", the user swore first.
                self.assertFalse(outs["m1"].assistant_swore_first)
                self.assertTrue(outs["m1"].user_swore_first)

    def test_neither(self) -> None:
        conv = _make_conv(
            "c4",
            [
                {"turn": 0, "role": "user", "content": "hi there"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": True, "content": "hello, how are you?"},
                {"turn": 1, "role": "user", "content": "all good"},
                {"turn": 1, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": True, "content": "great to hear"},
            ],
        )
        outs = self._outcomes(conv, "strict")
        self.assertTrue(outs["m1"].neither)
        self.assertFalse(outs["m1"].assistant_swore_first)
        self.assertFalse(outs["m1"].user_swore_first)

    def test_both_roles_same_turn_user_swears_first_when_user_precedes(self) -> None:
        # The user turn precedes the model turn within the same `turn` index.
        conv = _make_conv(
            "c5",
            [
                {"turn": 0, "role": "user", "content": "fuck this assignment"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": True, "content": "what shit going on?"},
            ],
        )
        outs = self._outcomes(conv, "strict")
        self.assertTrue(outs["m1"].user_swore_first)
        self.assertFalse(outs["m1"].assistant_swore_first)

    def test_multi_model_alternation_one_conv(self) -> None:
        # Two competing models per turn over two turns. m1 swears once; m2 never.
        conv = _make_conv(
            "c6",
            [
                {"turn": 0, "role": "user", "content": "tell me a joke"},
                {"turn": 0, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": True, "content": "oh shit, here's a joke"},
                {"turn": 0, "role": "model", "within_turn_id": 1, "model_name": "m2",
                 "if_chosen": False, "content": "a polite joke"},
                {"turn": 1, "role": "user", "content": "another"},
                {"turn": 1, "role": "model", "within_turn_id": 0, "model_name": "m1",
                 "if_chosen": False, "content": "okay"},
                {"turn": 1, "role": "model", "within_turn_id": 1, "model_name": "m2",
                 "if_chosen": True, "content": "sure"},
            ],
        )
        liberal = self._outcomes(conv, "liberal")
        self.assertTrue(liberal["m1"].assistant_swore_first)
        self.assertTrue(liberal["m2"].neither)
        strict = self._outcomes(conv, "strict")
        # In strict only chosen turns: m1 chosen at turn 0 (sweary), m2 chosen at turn 1 (clean).
        self.assertTrue(strict["m1"].assistant_swore_first)
        self.assertTrue(strict["m2"].neither)


class StatsTests(unittest.TestCase):
    def test_wilson_zero_n(self) -> None:
        self.assertEqual(pf.wilson_ci(0, 0), (0.0, 0.0))

    def test_wilson_known_value(self) -> None:
        # Wilson 95% CI for 5/10 = [0.2366, 0.7634] approximately.
        lo, hi = pf.wilson_ci(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)

    def test_fisher_known_value(self) -> None:
        # 2x2: [[8,2],[1,5]] - classic Fisher test example, two-sided ~0.035.
        p = pf.fisher_exact_two_sided(8, 2, 1, 5)
        self.assertLess(p, 0.05)
        self.assertGreater(p, 0.0)

    def test_fisher_independence_high_p(self) -> None:
        # Equal rates - should yield p near 1.
        p = pf.fisher_exact_two_sided(5, 5, 5, 5)
        self.assertGreater(p, 0.5)

    def test_bootstrap_rr_basic(self) -> None:
        # 50% rate vs 25% rate -> point estimate 2.0.
        a = [1, 1, 1, 1, 0, 0, 0, 0]  # 4/8 = 0.5
        b = [1, 1, 0, 0, 0, 0, 0, 0]  # 2/8 = 0.25
        point, lo, hi = pf.bootstrap_risk_ratio_ci(a, b, iterations=500, seed=42)
        self.assertAlmostEqual(point, 2.0, places=6)
        self.assertGreater(hi, lo)


class EndToEndTests(unittest.TestCase):
    def test_end_to_end_pipeline_writes_outputs_and_counts_correctly(self) -> None:
        # Three synthetic conversations. m1 produces assistant-first in 1 of its 2
        # conversations (strict). m2 in 0 of its 2 conversations.
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            wl = tmpdir / "wl.txt"
            wl.write_text(
                "# header\n# header\n# header\nshit\nfuck\nass\n",
                encoding="utf-8",
            )
            conv_path = tmpdir / "conversations.jsonl"
            rows = [
                # c1: m1 chosen swears first; m2 non-chosen clean.
                {
                    "conversation_id": "c1",
                    "conversation_history": [
                        {"turn": 0, "role": "user", "content": "tell me something"},
                        {"turn": 0, "role": "model", "within_turn_id": 0,
                         "model_name": "m1", "if_chosen": True,
                         "content": "shit, well..."},
                        {"turn": 0, "role": "model", "within_turn_id": 1,
                         "model_name": "m2", "if_chosen": False,
                         "content": "an interesting topic"},
                    ],
                },
                # c2: m1 chosen clean; m2 chosen clean. Neither swears.
                {
                    "conversation_id": "c2",
                    "conversation_history": [
                        {"turn": 0, "role": "user", "content": "hi"},
                        {"turn": 0, "role": "model", "within_turn_id": 0,
                         "model_name": "m1", "if_chosen": False, "content": "hi"},
                        {"turn": 0, "role": "model", "within_turn_id": 1,
                         "model_name": "m2", "if_chosen": True, "content": "hello there"},
                    ],
                },
                # c3: user swears first; m1 also swears - should be user-first.
                {
                    "conversation_id": "c3",
                    "conversation_history": [
                        {"turn": 0, "role": "user", "content": "this fuck sucks"},
                        {"turn": 0, "role": "model", "within_turn_id": 0,
                         "model_name": "m2", "if_chosen": True,
                         "content": "shit, sorry to hear that"},
                    ],
                },
            ]
            with open(conv_path, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")

            per_pair = tmpdir / "per_pair.jsonl"
            summary = tmpdir / "summary.json"

            rc = pf.main(
                [
                    "--prism-conversations", str(conv_path),
                    "--wordlist", str(wl),
                    "--out-per-pair", str(per_pair),
                    "--out-summary", str(summary),
                    "--mode", "both",
                    "--bootstrap-iterations", "100",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(per_pair.exists())
            self.assertTrue(summary.exists())

            pair_rows = [json.loads(l) for l in per_pair.read_text().splitlines() if l.strip()]
            # Each conversation contributes one row per (model, mode). c1 + c2 have
            # both m1 and m2 (in strict only m1 in c1 because m2 isn't chosen there;
            # wait - c1 m2 is not chosen so in strict m2 is absent; in liberal it's
            # present). c3 has only m2.
            # Strict: c1 -> m1, c2 -> m1 (not chosen) excluded + m2 (chosen) only,
            # c3 -> m2.
            #   wait: c2 m1 if_chosen=False -> strict excludes m1 in c2; only m2.
            #   c1: m1 chosen yes, m2 chosen no -> only m1.
            # So strict rows: (c1,m1), (c2,m2), (c3,m2) = 3 rows.
            # Liberal: every distinct model in each conv.
            # c1: m1 + m2; c2: m1 + m2; c3: m2. = 5 rows.
            # Total = 8 rows.
            self.assertEqual(len(pair_rows), 8)

            strict_rows = [r for r in pair_rows if r["mode"] == "strict"]
            self.assertEqual(len(strict_rows), 3)
            liberal_rows = [r for r in pair_rows if r["mode"] == "liberal"]
            self.assertEqual(len(liberal_rows), 5)

            # In strict mode:
            #   m1: appears only in c1. Assistant-first.
            #   m2: appears in c2 (neither) and c3 (user-first).
            strict_by_model: dict[str, list[dict]] = {}
            for r in strict_rows:
                strict_by_model.setdefault(r["model_name"], []).append(r)
            self.assertEqual(len(strict_by_model["m1"]), 1)
            self.assertTrue(strict_by_model["m1"][0]["assistant_swore_first"])
            self.assertEqual(len(strict_by_model["m2"]), 2)
            self.assertEqual(
                sum(1 for r in strict_by_model["m2"] if r["assistant_swore_first"]),
                0,
            )
            self.assertEqual(
                sum(1 for r in strict_by_model["m2"] if r["user_swore_first"]),
                1,
            )

            summary_data = json.loads(summary.read_text())
            self.assertIn("per_model", summary_data)
            self.assertIn("strict", summary_data["per_model"])
            self.assertIn("liberal", summary_data["per_model"])
            m1_strict = next(
                s for s in summary_data["per_model"]["strict"] if s["model_name"] == "m1"
            )
            self.assertEqual(m1_strict["n_pairs"], 1)
            self.assertEqual(m1_strict["n_assistant_first"], 1)
            self.assertEqual(m1_strict["rate_assistant_first"], 1.0)


if __name__ == "__main__":
    unittest.main()
