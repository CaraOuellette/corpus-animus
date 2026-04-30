from __future__ import annotations

import unittest

from extract_sharegpt_ultrachat_assistant_messages import (
    MessageNormalizer,
    extract_sharegpt_assistant_messages,
    extract_ultrachat_assistant_messages,
    sharegpt_message_text,
)


class ShareGPTUltraChatExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = MessageNormalizer()

    def test_sharegpt_message_text_prefers_value_then_text_then_markdown(self) -> None:
        self.assertEqual(
            sharegpt_message_text({"value": "primary", "text": "secondary"}),
            "primary",
        )
        self.assertEqual(
            sharegpt_message_text({"text": "secondary"}),
            "secondary",
        )
        self.assertEqual(
            sharegpt_message_text({"markdown": {"answer": "markdown answer"}}),
            "markdown answer",
        )

    def test_extract_sharegpt_assistant_messages_emits_multiple_turns(self) -> None:
        row = {
            "id": "share-1",
            "conversations": [
                {"from": "system", "value": "Behave."},
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "*smiles*\nHello"},
                {"from": "human", "value": "Tell me more"},
                {"from": "chatgpt", "text": "Sure."},
                {"from": "bard", "markdown": {"answer": "Another answer"}},
            ],
        }

        extraction = extract_sharegpt_assistant_messages(row, self.normalizer)

        self.assertEqual(extraction.assistant_messages_observed, 3)
        self.assertEqual(extraction.user_messages_observed, 2)
        self.assertEqual(extraction.system_messages_observed, 1)
        self.assertEqual(len(extraction.records), 3)
        self.assertEqual(extraction.records[0].source_dataset, "ShareGPT")
        self.assertEqual(extraction.records[0].source_conversation_id, "share-1")
        self.assertEqual(extraction.records[0].assistant_turn_number, 1)
        self.assertEqual(extraction.records[0].preceding_user_text_raw, "Hi")
        self.assertEqual(extraction.records[0].system_prompt_text, "Behave.")
        self.assertEqual(extraction.records[1].assistant_turn_number, 2)
        self.assertEqual(extraction.records[1].preceding_user_text_raw, "Tell me more")
        self.assertEqual(extraction.records[2].assistant_text_raw, "Another answer")

    def test_extract_ultrachat_assistant_messages_uses_alternating_turns(self) -> None:
        row = {
            "id": "ultra-1",
            "data": [
                "Prompt one",
                "*nods* Response one",
                "Prompt two",
                "Response two",
            ],
        }

        extraction = extract_ultrachat_assistant_messages(row, self.normalizer)

        self.assertEqual(extraction.assistant_messages_observed, 2)
        self.assertEqual(extraction.user_messages_observed, 2)
        self.assertEqual(len(extraction.records), 2)
        self.assertEqual(extraction.records[0].source_dataset, "UltraChat")
        self.assertEqual(extraction.records[0].source_conversation_id, "ultra-1")
        self.assertEqual(extraction.records[0].preceding_user_text_raw, "Prompt one")
        self.assertEqual(extraction.records[0].assistant_turn_number, 1)
        self.assertEqual(extraction.records[1].preceding_user_text_raw, "Prompt two")
        self.assertEqual(extraction.records[1].assistant_turn_number, 2)


if __name__ == "__main__":
    unittest.main()
