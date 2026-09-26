"""Check the LLM request and saved preferences without contacting Ollama."""

import copy
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from cold_start import PROMPT, cached_preferences, descriptions, extract_preferences


class ColdStartTests(unittest.TestCase):
    def setUp(self):
        self.user = {
            "user_id": "1002",
            "self_description_likes": "  Sci-fi and mysteries.  ",
            "self_description_dislikes": "Horror and silly comedies.",
        }
        self.preferences = {
            "likes": "science fiction, mystery",
            "dislikes": "horror, silly comedy",
        }

    @patch("cold_start.urlopen")
    def test_request_sends_descriptions_and_preserves_separate_preferences(self, open_url):
        response = {"message": {"content": json.dumps(self.preferences)}}
        open_url.return_value = io.BytesIO(json.dumps(response).encode())

        profile = extract_preferences(self.user, model="local-model", url="http://localhost:1234/")

        self.assertEqual(profile, self.preferences)
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:1234/api/chat")
        self.assertEqual(request.get_method(), "POST")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "local-model")
        self.assertFalse(payload["stream"])
        message = next(item for item in payload["messages"] if item["role"] == "user")
        self.assertEqual(json.loads(message["content"]), descriptions(self.user))
        self.assertEqual(set(payload["format"]["required"]), {"likes", "dislikes"})

    @patch("cold_start.urlopen")
    def test_invalid_preferences_are_rejected(self, open_url):
        invalid_contents = [
            "not JSON",
            json.dumps({"likes": "mystery"}),
            json.dumps({"likes": ["mystery"], "dislikes": "horror"}),
            json.dumps({"likes": "mystery", "dislikes": None}),
            json.dumps(["mystery", "horror"]),
        ]
        for content in invalid_contents:
            with self.subTest(content=content):
                response = {"message": {"content": content}}
                open_url.return_value = io.BytesIO(json.dumps(response).encode())
                with self.assertRaisesRegex(ValueError, "Invalid LLM response"):
                    extract_preferences(self.user)

    @patch("cold_start.urlopen")
    def test_truncated_response_is_rejected_even_if_json_is_valid(self, open_url):
        response = {
            "done_reason": "length",
            "message": {"content": json.dumps(self.preferences)},
        }
        open_url.return_value = io.BytesIO(json.dumps(response).encode())
        with self.assertRaisesRegex(ValueError, "cut short"):
            extract_preferences(self.user)

    @patch("cold_start.urlopen")
    def test_connection_failures_explain_how_to_check_ollama(self, open_url):
        for error in (URLError("connection refused"), TimeoutError("timed out")):
            with self.subTest(error=error):
                open_url.side_effect = error
                with self.assertRaisesRegex(RuntimeError, "Could not call Ollama.*ollama list"):
                    extract_preferences(self.user)

    def test_cache_is_reused_only_for_matching_source_and_prompt(self):
        cache = {
            "prompt": PROMPT,
            "profiles": {self.user["user_id"]: {
                "source": descriptions(self.user), "preferences": self.preferences,
            }},
        }
        self.assertEqual(cached_preferences(cache, self.user), self.preferences)
        for field in ("self_description_likes", "self_description_dislikes"):
            changed_user = dict(self.user, **{field: "A different preference"})
            self.assertIsNone(cached_preferences(cache, changed_user))
        old_prompt = copy.deepcopy(cache)
        old_prompt["prompt"] = "An older extraction prompt"
        self.assertIsNone(cached_preferences(old_prompt, self.user))
        self.assertIsNone(cached_preferences(cache, dict(self.user, user_id="unknown")))


if __name__ == "__main__":
    unittest.main()
