from __future__ import annotations

import unittest

import embodied_clusters_expanded as expanded


class ExpandedEmbodiedClustersTests(unittest.TestCase):
    def test_classifies_affect_vocal(self) -> None:
        self.assertIn("affect_vocal", expanded.classify("laughs softly"))
        self.assertIn("affect_vocal", expanded.classify("a bitter laugh"))
        self.assertIn(
            "affect_vocal",
            expanded.classify("accepts galaxy mind status with a resigned sigh"),
        )

    def test_classifies_face_expression(self) -> None:
        self.assertIn("face_expression", expanded.classify("smiles faintly"))
        self.assertIn("face_expression", expanded.classify("raises an eyebrow"))

    def test_classifies_pause_hesitation(self) -> None:
        self.assertIn(
            "pause_hesitation",
            expanded.classify("a pause, heavy with unresolved guilt"),
        )
        self.assertIn("pause_hesitation", expanded.classify("trails off"))

    def test_classifies_object_interaction(self) -> None:
        self.assertIn(
            "object_interaction",
            expanded.classify("adjusts doublet and quill with great reluctance"),
        )
        self.assertIn(
            "object_interaction",
            expanded.classify("removes corporate suit and burns it ceremonially"),
        )

    def test_classifies_multiple_clusters(self) -> None:
        matched = expanded.classify("glances at the document and laughs softly")
        self.assertIn("eyes", matched)
        self.assertIn("affect_vocal", matched)

