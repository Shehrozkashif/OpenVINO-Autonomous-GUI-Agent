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
from core.types import SubTask


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
        subs = _subs("open Microsoft Teams", "open the calendar")
        assert RouterAgent._missing_actions(
            "open teams and go to the calendar", subs) == []

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


class TestParseTruncatedJSON:
    """Salvage of truncated decompose output (hit max_tokens mid-object).

    Live failure: an 8-subtask decomposition was cut mid-string, parse failed,
    and the schema-only retry produced conversational to-do items the planner
    could not act on. The complete prefix of a truncated array is real work —
    it must be recovered, not thrown away.
    """

    def _router(self):
        return RouterAgent(client=MagicMock())

    def test_truncated_mid_string_salvages_prefix(self):
        # Shape taken from the live log: cut off inside subtask 3's description.
        raw = (
            '[{"id":1,"description":"open Thunderbird","depends_on":[]},'
            '{"id":2,"description":"click the Write new message button","depends_on":[1]},'
            '{"id":3,"description":"set To to alex@example.com, and type the message'
        )
        subs = self._router()._parse_subtasks(raw)
        assert [s.id for s in subs] == [1, 2]
        assert subs[0].description == "open Thunderbird"

    def test_truncated_after_object_salvages_all_complete(self):
        raw = (
            '[{"id":1,"description":"open Notepad","depends_on":[]},'
            '{"id":2,"description":"type: hello","depends_on":[1]},'
        )
        subs = self._router()._parse_subtasks(raw)
        assert [s.id for s in subs] == [1, 2]

    def test_intact_json_still_parses(self):
        raw = '[{"id":1,"description":"open Calculator","depends_on":[]}]'
        subs = self._router()._parse_subtasks(raw)
        assert len(subs) == 1 and subs[0].description == "open Calculator"

    def test_unsalvageable_still_raises(self):
        import pytest
        with pytest.raises((ValueError, json.JSONDecodeError)):
            self._router()._parse_subtasks('[{"id":1,"description":"open')


class TestInstalledAppHint:
    """Ground-truth app hint: OS app list × the user's own words. No hardcoding."""

    def _hint(self, instruction, apps):
        from unittest.mock import patch
        with patch("desktop.system.installed_apps", return_value=apps):
            return RouterAgent._installed_app_hint(instruction)

    def test_mentioned_installed_app_is_hinted(self):
        hint = self._hint(
            "schedule an outlook meeting with shehroz at 3pm",
            ["Outlook (new)", "Microsoft Edge", "Calculator"],
        )
        assert "Outlook (new)" in hint
        assert "INSTALLED" in hint

    def test_unmentioned_apps_are_not_hinted(self):
        hint = self._hint(
            "schedule an outlook meeting",
            ["Zoom Workplace", "Spotify"],
        )
        assert hint == ""

    def test_no_apps_detected_means_no_hint(self):
        assert self._hint("open outlook", []) == ""

    def test_any_app_name_works_generally(self):
        hint = self._hint("play some music on spotify", ["Spotify", "Outlook"])
        assert "Spotify" in hint and "Outlook" not in hint


class TestAppHintWholeWordMatching:
    """Regression: loose substring matching sent a run into Task Scheduler.

    'schedule a meeting WITH shehroz' must NOT hint 'Task Scheduler'
    ("schedule" ⊂ "Scheduler") or 'Windows Defender Firewall with Advanced
    Security' ("with" is a word of it). Only actually-named apps count.
    """

    _APPS = ["Task Scheduler",
             "Windows Defender Firewall with Advanced Security",
             "Outlook (new)", "Zoom Workplace"]

    def _hint(self, instruction):
        from unittest.mock import patch
        with patch("desktop.system.installed_apps", return_value=self._APPS):
            return RouterAgent._installed_app_hint(instruction)

    def test_no_app_named_means_no_hint(self):
        assert self._hint(
            "schedule a meeting with shehroz alex@example.com 3pm 6 july"
        ) == ""

    def test_substring_of_app_word_does_not_match(self):
        # "schedule" must not summon "Scheduler"
        assert "Task Scheduler" not in self._hint("schedule something for me")

    def test_named_app_still_hinted(self):
        hint = self._hint("schedule a meeting in outlook with shehroz")
        assert "Outlook (new)" in hint
        assert "Firewall" not in hint and "Task Scheduler" not in hint

    def test_zoom_by_name_still_works(self):
        hint = self._hint("start a zoom call with my friend")
        assert "Zoom Workplace" in hint


class TestFailureSummaryCarriesBlocker:
    """Live failure 2026-07-05 17:53-18:02: Teams sat at its sign-in window
    for 9 minutes and the run ended with just 'task failed' — while the goal
    check had named the blocker every cycle. The summary must surface it.
    """

    def _router(self):
        client = MagicMock()
        resp = MagicMock()
        resp.content = "Task failed: Teams is at a sign-in screen."
        client.query_llm.return_value = resp
        return RouterAgent(client), client

    def test_blocker_included_on_failure(self):
        router, client = self._router()
        router.summarize_completion(
            "t1", [1, 2], False,
            blocker="the schedule-meeting form is not open; only "
                    "'Switch to another account' is on screen",
        )
        sent = client.query_llm.call_args[0][0][1]["content"]
        assert "Switch to another account" in sent
        assert "blocker" in sent.lower()

    def test_blocker_ignored_on_success(self):
        router, client = self._router()
        router.summarize_completion("t1", [1, 2], True, blocker="stale note")
        sent = client.query_llm.call_args[0][0][1]["content"]
        assert "stale note" not in sent

    def test_backward_compatible_without_blocker(self):
        router, client = self._router()
        out = router.summarize_completion("t1", [1], False)
        assert isinstance(out, str) and out


class TestEmptyPlanIsRetried:
    """Live failure: 'add two numbers' returned '[]' in two output tokens.

    An empty array is valid JSON, so it never raised, and every backstop below
    only inspects sub-tasks that exist. The bad generation therefore reached the
    orchestrator untouched and the run was reported complete without executing
    anything. An empty plan now retries exactly like a parse failure.
    """

    def _router(self, *replies):
        client = MagicMock()
        client.query_llm = MagicMock(
            side_effect=[MagicMock(content=r) for r in replies]
        )
        return RouterAgent(client), client

    def test_empty_array_triggers_a_retry(self):
        good = json.dumps([{"id": 1, "description": "open Calculator", "depends_on": []}])
        router, client = self._router("[]", good)

        _, subs = router.decompose("add two numbers")

        assert client.query_llm.call_count == 2
        assert [s.description for s in subs] == ["open Calculator"]

    def test_the_retry_demands_json_only(self):
        good = json.dumps([{"id": 1, "description": "open Calculator", "depends_on": []}])
        router, client = self._router("[]", good)

        router.decompose("add two numbers")

        retry_system = client.query_llm.call_args[0][0][0]["content"]
        assert "Output ONLY the JSON array" in retry_system

    def test_a_non_empty_plan_is_not_retried(self):
        good = json.dumps([{"id": 1, "description": "open Calculator", "depends_on": []}])
        router, client = self._router(good)

        _, subs = router.decompose("add two numbers")

        assert client.query_llm.call_count == 1
        assert len(subs) == 1

    def test_an_empty_retry_still_returns_empty(self):
        """The orchestrator refuses an empty plan; the router must not pretend."""
        router, client = self._router("[]", "[]")

        _, subs = router.decompose("add two numbers")

        assert subs == []
        assert client.query_llm.call_count == 2
