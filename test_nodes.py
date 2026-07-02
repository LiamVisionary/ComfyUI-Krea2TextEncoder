from __future__ import annotations

import json
import unittest

import nodes


class Krea2TextEncoderPromptTests(unittest.TestCase):
    def test_compacts_krea2_photo_json_prompt(self) -> None:
        raw = json.dumps(
            {
                "subject": {
                    "description": "A 42-year-old adult subject in a living room",
                    "age": "42",
                    "mirror_rules": "medium-wide shot, both subjects visible in shot",
                },
                "hair": {"color": "black", "style": "long"},
                "body": {"legs": "feet visible on sofa cushions"},
                "pose": {"position": "seated beside another adult on a sofa"},
                "clothing": {"top": {"type": "sweater"}},
                "accessories": {"prop": "plush sofa"},
                "photography": {
                    "shot_type": "medium-wide shot",
                    "composition": "full sofa visible, background headroom above subject",
                    "crop_control": "eyes, hairline, and crown visible",
                },
                "background": {
                    "setting": "living room with ceiling light",
                    "elements": ["full sofa visible", "visible floor plane"],
                },
                "the_vibe": {"mood": "cinematic realism"},
                "constraints": {
                    "must_keep": ["exact requested age: 42", "feet visible in shot"],
                    "avoid": ["cropped head"],
                },
                "negative_prompt": ["blurry"],
            },
            indent=2,
        )

        compact = nodes._compact_krea2_json_prompt(raw, max_chars=900)
        value = json.loads(compact)

        self.assertLess(len(compact), len(raw))
        self.assertEqual(value["subject"]["age"], "42")
        self.assertIn("visible", compact)
        self.assertIn("full sofa visible", compact)
        self.assertIn('"subject"', compact)
        self.assertIn("\n", compact)
        self.assertLess(compact.count("  "), raw.count("  "))
        # Negation lists are stripped before positive-conditioning encoding.
        self.assertNotIn('"avoid"', compact)
        self.assertNotIn('"negative_prompt"', compact)
        self.assertNotIn("cropped head", compact)

    def test_structured_mode_strips_negation_lists(self) -> None:
        raw = json.dumps(
            {
                "subject": {"description": "An adult woman in a living room", "age": "30"},
                "body": {"legs": "legs visible on sofa cushions"},
                "pose": {"position": "seated"},
                "photography": {"shot_type": "medium-wide shot"},
                "background": {"setting": "living room"},
                "constraints": {
                    "must_keep": ["exact requested age: 30"],
                    "avoid": ["female feet on floor", "cropped head"],
                },
                "negative_prompt": ["blurry", "female leg hanging off sofa"],
            },
            indent=2,
        )

        compact = nodes._compact_krea2_json_prompt(raw, mode="json_structured")

        self.assertNotIn('"avoid"', compact)
        self.assertNotIn('"negative_prompt"', compact)
        self.assertNotIn("female feet on floor", compact)
        self.assertNotIn("female leg hanging off sofa", compact)
        self.assertIn("exact requested age: 30", compact)

    def test_plain_prompt_is_unchanged(self) -> None:
        prompt = "medium-wide living room photo with visible floor plane"
        self.assertEqual(nodes._compact_krea2_json_prompt(prompt), prompt)

    def test_must_keep_survives_tight_budget(self) -> None:
        raw = json.dumps(
            {
                "subject": {
                    "description": "A highly detailed adult subject description " * 30,
                    "age": "42",
                    "mirror_rules": "medium-wide shot",
                },
                "hair": {"color": "black"},
                "body": {"legs": "general lower-body details " * 20},
                "pose": {"position": "general pose details " * 20},
                "clothing": {"top": {"type": "sweater"}},
                "accessories": {"prop": "plush sofa"},
                "photography": {"composition": "full sofa visible, background headroom"},
                "background": {"setting": "living room", "elements": ["ceiling light", "wide sofa"]},
                "the_vibe": {"mood": "cinematic"},
                "constraints": {
                    "must_keep": [
                        "exact requested age: 42",
                        "female legs and feet on sofa cushions",
                        "male eyes, hairline, crown, and facial hair visible",
                    ],
                    "avoid": ["cropped head"],
                },
                "negative_prompt": ["blurry"],
            }
        )

        compact = nodes._compact_krea2_json_prompt(raw, max_chars=700, mode="prose_compact")

        self.assertLessEqual(len(compact), 700)
        self.assertIn("exact requested age: 42", compact)
        self.assertIn("female legs and feet on sofa cushions", compact)
        self.assertIn("male eyes, hairline, crown", compact)

    def test_json_like_malformed_prompt_is_repaired_and_minified(self) -> None:
        malformed = """
        generated prompt:
        {
          "subject": {
            "description": "A 42-year-old adult subject in a living room"
            "mirror_rules": "medium-wide shot, both subjects visible in shot"
          },
          "photography": {
            "shot_type": "medium-wide shot",
            "composition": "full sofa visible, visible floor plane, background headroom"
          },
          "background": {
            "setting": "living room with ceiling light",
            "elements": ["full sofa visible", "visible floor plane"]
          },
          "constraints": {
            "must_keep": ["exact requested age: 42", "feet visible in shot"]
          },
          "negative_prompt": ["blurry", "cropped head"]
        }
        """

        compact = nodes._compact_krea2_json_prompt(malformed, max_chars=700)
        value = json.loads(compact)

        self.assertEqual(value["subject"]["description"], "A 42-year-old adult subject in a living room")
        self.assertIn("full sofa visible", compact)
        self.assertIn("visible floor plane", compact)
        self.assertIn('"subject"', compact)
        self.assertIn("\n", compact)

    def test_text_only_conditioning_cache_reuses_identical_prompt(self) -> None:
        class FakeClip:
            def __init__(self) -> None:
                self.tokenize_calls = 0
                self.encode_calls = 0

            def tokenize(self, text, images=None, llama_template=None):
                self.tokenize_calls += 1
                return {"text": text, "images": images, "template": llama_template}

            def encode_from_tokens_scheduled(self, tokens):
                self.encode_calls += 1
                return [["conditioning", {"call": self.encode_calls, "text": tokens["text"]}]]

        clip = FakeClip()
        encoder = nodes.TextEncodeKrea2()

        first = encoder.encode(clip, "same prompt", auto_compact_json=False)
        second = encoder.encode(clip, "same prompt", auto_compact_json=False)
        third = encoder.encode(clip, "different prompt", auto_compact_json=False)

        self.assertEqual(clip.tokenize_calls, 2)
        self.assertEqual(clip.encode_calls, 2)
        self.assertEqual(first[0][0][1]["call"], 1)
        self.assertEqual(second[0][0][1]["call"], 1)
        self.assertEqual(third[0][0][1]["call"], 2)


if __name__ == "__main__":
    unittest.main()
