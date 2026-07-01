# tests/unit/test_router.py
"""Tests for the router completeness backstop.

Live failure this guards against: an 8B router intermittently drops a trailing
requested action (e.g. "...and save the file"), so the agent finishes without
saving. Detection is deterministic + general; the FIX is an LLM re-prompt, so
filenames/steps stay model-chosen rather than hardcoded.
"""
import sys

sys.path.insert(0, ".")

import json
from unittest.mock import MagicMock

from agents.router import RouterAgent
from core.protocols import SubTask


def _subs(*descriptions) -> list[SubTask]:
    return [
        SubTask(id=i + 1, description=d, depends_on=([i] if i else []))
        for i, d in enumerate(descriptions)
    ]


class TestMissingActionDetection:
    """_missing_actions flags requested-but-uncovered finalizing actions."""

    def test_save_requested_but_dropped(self):
        subs = _subs("open Notepad", "type the haiku")
        assert RouterAgent._missing_actions(
            "Open Notepad and write a haiku and save the file", subs) == ["save"]

    def test_save_present_is_not_flagged(self):
        subs = _subs("open Notepad", "type the haiku", "save the document as x.txt")
        assert RouterAgent._missing_actions(
            "Open Notepad and write a haiku and save the file", subs) == []

    def test_no_finalizing_verb_means_nothing_missing(self):
        subs = _subs("open Firefox", "navigate to github.com")
        assert RouterAgent._missing_actions(
            "open firefox and go to github", subs) == []

    def test_close_requested_but_dropped(self):
        subs = _subs("open Notepad", "type text")
        assert "close" in RouterAgent._missing_actions(
            "open notepad, type text, and close it", subs)

    def test_send_covered_by_email_synonym(self):
        # The instruction says "send"; a sub-task that "emails" it counts.
        subs = _subs("open Thunderbird", "email the report to bob")
        # "send" regex also matches "email" in coverage, so not flagged.
        assert RouterAgent._missing_actions(
            "compose and send the report", subs) == []


class TestEnsureComplete:
    """_ensure_complete re-prompts once and only accepts a genuine improvement."""

    def _router(self, retry_payload):
        client = MagicMock()
        client.query_llm = MagicMock(return_value=MagicMock(content=retry_payload))
        return RouterAgent(client)

    def test_reprompt_adds_missing_save(self):
        original = _subs("open Notepad", "type the haiku")
        retry_json = json.dumps([
            {"id": 1, "description": "open Notepad", "depends_on": []},
            {"id": 2, "description": "type the haiku", "depends_on": [1]},
            {"id": 3, "description": "save the document as C:/Users/x/Desktop/haiku.txt",
             "depends_on": [2]},
        ])
        router = self._router(retry_json)
        out = router._ensure_complete(
            "Open Notepad and write a haiku and save the file",
            "Instruction: Open Notepad and write a haiku and save the file",
            original,
        )
        assert any("save" in s.description.lower() for s in out)
        assert len(out) == 3
        router.client.query_llm.assert_called_once()

    def test_reprompt_that_still_drops_save_keeps_original(self):
        original = _subs("open Notepad", "type the haiku")
        # Retry STILL omits the save → must keep the original, not make it worse.
        retry_json = json.dumps([
            {"id": 1, "description": "open Notepad", "depends_on": []},
            {"id": 2, "description": "type the haiku", "depends_on": [1]},
        ])
        router = self._router(retry_json)
        out = router._ensure_complete(
            "Open Notepad and write a haiku and save the file",
            "Instruction: ...",
            original,
        )
        assert out is original

    def test_no_missing_action_skips_reprompt(self):
        original = _subs("open Notepad", "type", "save as x.txt")
        router = self._router("[]")
        out = router._ensure_complete(
            "Open Notepad and write and save the file", "Instruction: ...", original)
        assert out is original
        router.client.query_llm.assert_not_called()
