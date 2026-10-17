#!/usr/bin/env python3
"""GET /api/marii/ask/<request> — run: python3 tests/test_marii_ask.py"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("PYX_API_KEY", None)
os.environ.pop("PYX_API_KEYS", None)

from app import app  # noqa: E402


class MariiAskPathTests(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()

    def test_path_echoes_request(self):
        res = self.c.get("/api/marii/ask/hello%20world")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["request"], "hello world")
        self.assertEqual(data["echo"], "hello world")
        self.assertEqual(data["reply"], "hello world")
        self.assertFalse(data["ai"])

    def test_faq_marii(self):
        res = self.c.get("/api/marii/ask/what%20is%20marii")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["request"], "what is marii")
        self.assertIn("Mainline Artificial Realtime Instant Intelligence", data["reply"])

    def test_query_string(self):
        res = self.c.get("/api/marii/ask", query_string={"request": "who are you"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("MARII ask API", data["reply"])

    def test_missing_request(self):
        res = self.c.get("/api/marii/ask")
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()["ok"])

    def test_post_stays_501(self):
        res = self.c.post("/api/marii/ask", json={"request": "hi"})
        self.assertEqual(res.status_code, 501)
        self.assertFalse(res.get_json().get("ai", True))

    def test_alias_path(self):
        res = self.c.get("/marii/ask/yo")
        self.assertEqual(res.status_code, 200)
        self.assertIn("marii ask is live", res.get_json()["reply"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
