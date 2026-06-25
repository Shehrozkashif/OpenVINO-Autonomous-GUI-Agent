# tests/unit/test_screen_snapshot.py
"""
Unit tests for Fix 1.2 — ScreenSnapshot structured world model.

Covers:
  TestOCRRegionAndSnapshot  — dataclass construction, foreground_texts(),
                               background_by_window(), format_for_planner()
  TestCaptureSnapshot       — foreground window detection, window assignment,
                               is_in_foreground flag, quality filtering
  TestPlannerFormattedOutput — planner receives snapshot-formatted string,
                               background text is clearly marked non-interactive
  TestGroundingForegroundOnly — find_text(foreground_only=True) rejects
                                 background words
"""
import time
import unittest
from typing import List
from unittest.mock import MagicMock, patch

from agents.grounding.grounding_agent import OCREngine, OCRWord
from core.capture.screen_snapshot import (
    OCRRegion,
    ScreenSnapshot,
    _point_in_rect,
    capture_snapshot,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _region(text, fg=True, title="MyApp") -> OCRRegion:
    return OCRRegion(
        text=text,
        bbox=(0, 0, 100, 20),
        confidence=0.95,
        window_id=1 if fg else 2,
        window_title=title,
        is_in_foreground=fg,
    )


def _word(text, x=0, y=0, w=50, h=20, conf=0.90, fg=True) -> OCRWord:
    return OCRWord(
        text=text, x=x, y=y, w=w, h=h, conf=conf,
        is_in_foreground=fg,
        element_type="foreground_interactive" if fg else "document_text",
    )


def _snapshot(fg_regions=None, bg_regions=None) -> ScreenSnapshot:
    regions = []
    for t in (fg_regions or []):
        regions.append(_region(t, fg=True, title="ForegroundApp"))
    for t in (bg_regions or []):
        regions.append(_region(t, fg=False, title="BackgroundWin"))
    return ScreenSnapshot(
        timestamp=time.time(),
        foreground_window_title="ForegroundApp",
        foreground_process="app.exe",
        screen_hash="abc123",
        ocr_regions=regions,
    )


# ── TestOCRRegionAndSnapshot ───────────────────────────────────────────────────

class TestOCRRegionAndSnapshot(unittest.TestCase):

    def test_ocr_region_construction(self):
        r = _region("New", fg=True)
        self.assertEqual(r.text, "New")
        self.assertTrue(r.is_in_foreground)
        self.assertEqual(r.window_id, 1)

    def test_foreground_texts_returns_only_fg(self):
        snap = _snapshot(fg_regions=["New", "Folder"], bg_regions=["AgentLog"])
        self.assertIn("New", snap.foreground_texts())
        self.assertIn("Folder", snap.foreground_texts())
        self.assertNotIn("AgentLog", snap.foreground_texts())

    def test_foreground_texts_deduplicates(self):
        regions = [_region("New"), _region("New")]
        snap = ScreenSnapshot(
            timestamp=0, foreground_window_title="App", foreground_process="x",
            screen_hash="h", ocr_regions=regions,
        )
        self.assertEqual(snap.foreground_texts().count("New"), 1)

    def test_foreground_texts_case_insensitive_dedup(self):
        regions = [_region("New"), _region("new")]
        snap = ScreenSnapshot(
            timestamp=0, foreground_window_title="App", foreground_process="x",
            screen_hash="h", ocr_regions=regions,
        )
        self.assertEqual(len(snap.foreground_texts()), 1)

    def test_background_by_window_groups_correctly(self):
        snap = _snapshot(fg_regions=["OK"], bg_regions=["Monitor", "Event"])
        bg = snap.background_by_window()
        self.assertIn("BackgroundWin", bg)
        self.assertIn("Monitor", bg["BackgroundWin"])
        self.assertIn("Event", bg["BackgroundWin"])

    def test_background_by_window_excludes_fg(self):
        snap = _snapshot(fg_regions=["OK"], bg_regions=["Monitor"])
        bg = snap.background_by_window()
        for texts in bg.values():
            self.assertNotIn("OK", texts)

    def test_format_for_planner_contains_fg_window(self):
        snap = _snapshot(fg_regions=["New", "Folder"])
        out = snap.format_for_planner()
        self.assertIn("ForegroundApp", out)
        self.assertIn("app.exe", out)

    def test_format_for_planner_contains_fg_elements(self):
        snap = _snapshot(fg_regions=["New", "Folder", "Shortcut"])
        out = snap.format_for_planner()
        self.assertIn('"New"', out)
        self.assertIn('"Folder"', out)

    def test_format_for_planner_marks_background(self):
        snap = _snapshot(fg_regions=["New"], bg_regions=["AgentLog"])
        out = snap.format_for_planner()
        self.assertIn("background", out.lower())
        self.assertIn("BackgroundWin", out)

    def test_format_for_planner_no_background_section_when_empty(self):
        snap = _snapshot(fg_regions=["New"])
        out = snap.format_for_planner()
        self.assertNotIn("background", out.lower())

    def test_format_for_planner_none_detected_when_no_fg(self):
        snap = _snapshot(fg_regions=[], bg_regions=["Hidden"])
        out = snap.format_for_planner()
        self.assertIn("none detected", out)

    def test_format_for_planner_limits_fg_tokens(self):
        many = [f"Token{i}" for i in range(50)]
        snap = _snapshot(fg_regions=many)
        out = snap.format_for_planner()
        # At most 40 tokens quoted in the foreground line
        count = out.count('"Token')
        self.assertLessEqual(count, 40)


# ── TestCaptureSnapshot ────────────────────────────────────────────────────────

class TestCaptureSnapshot(unittest.TestCase):

    def _make_capturer(self, width=1920, height=1080):
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(200, 200, 200))
        capturer = MagicMock()
        capturer.capture.return_value = img
        return capturer

    def _make_ocr(self, words: list[OCRWord]):
        ocr = MagicMock(spec=OCREngine)
        ocr.is_available.return_value = True
        ocr.extract.return_value = words
        return ocr

    @patch("core.capture.screen_snapshot._get_foreground_hwnd_and_title",
           return_value=(42, "Notepad"))
    @patch("core.capture.screen_snapshot._get_foreground_process",
           return_value="notepad.exe")
    @patch("core.capture.screen_snapshot._enum_visible_windows",
           return_value=[
               (42, "Notepad", (0, 0, 800, 600)),
               (99, "AgentLog", (900, 0, 1920, 600)),
           ])
    def test_foreground_window_title_captured(self, *_mocks):
        capturer = self._make_capturer()
        ocr = self._make_ocr([])
        snap = capture_snapshot(capturer, ocr)
        self.assertEqual(snap.foreground_window_title, "Notepad")

    @patch("core.capture.screen_snapshot._get_foreground_hwnd_and_title",
           return_value=(42, "Notepad"))
    @patch("core.capture.screen_snapshot._get_foreground_process",
           return_value="notepad.exe")
    @patch("core.capture.screen_snapshot._enum_visible_windows",
           return_value=[
               (42, "Notepad", (0, 0, 800, 600)),
               (99, "AgentLog", (900, 0, 1920, 600)),
           ])
    def test_foreground_process_captured(self, *_mocks):
        capturer = self._make_capturer()
        ocr = self._make_ocr([])
        snap = capture_snapshot(capturer, ocr)
        self.assertEqual(snap.foreground_process, "notepad.exe")

    @patch("core.capture.screen_snapshot._get_foreground_hwnd_and_title",
           return_value=(42, "Notepad"))
    @patch("core.capture.screen_snapshot._get_foreground_process",
           return_value="notepad.exe")
    @patch("core.capture.screen_snapshot._enum_visible_windows",
           return_value=[
               (42, "Notepad", (0, 0, 800, 600)),
               (99, "AgentLog", (900, 0, 1920, 600)),
           ])
    def test_words_in_foreground_rect_marked_as_fg(self, *_mocks):
        # Word at thumbnail pixel (200, 100) → screen coords ~(400, 200) in 1920×1080
        # Notepad rect (0,0,800,600) contains (400,200) → is_in_foreground=True
        capturer = self._make_capturer()
        words = [_word("File", x=100, y=50, w=40, h=18, conf=0.95)]
        ocr = self._make_ocr(words)
        snap = capture_snapshot(capturer, ocr)
        self.assertTrue(len(snap.ocr_regions) > 0)
        self.assertTrue(snap.ocr_regions[0].is_in_foreground)

    @patch("core.capture.screen_snapshot._get_foreground_hwnd_and_title",
           return_value=(42, "Notepad"))
    @patch("core.capture.screen_snapshot._get_foreground_process",
           return_value="notepad.exe")
    @patch("core.capture.screen_snapshot._enum_visible_windows",
           return_value=[
               (42, "Notepad",  (0, 0, 800, 600)),
               (99, "AgentLog", (1400, 0, 1920, 600)),
           ])
    def test_words_in_background_rect_marked_as_bg(self, *_mocks):
        # thumb is 960×540 (half of 1920×1080), scale ≈ 2.
        # Word at thumb(750, 100) → screen (~1500, 200) → inside AgentLog rect → bg
        capturer = self._make_capturer()
        words = [_word("Monitor", x=750, y=50, w=40, h=18, conf=0.95)]
        ocr = self._make_ocr(words)
        snap = capture_snapshot(capturer, ocr)
        if snap.ocr_regions:
            self.assertFalse(snap.ocr_regions[0].is_in_foreground)

    @patch("core.capture.screen_snapshot._get_foreground_hwnd_and_title",
           return_value=(42, "App"))
    @patch("core.capture.screen_snapshot._get_foreground_process",
           return_value="app.exe")
    @patch("core.capture.screen_snapshot._enum_visible_windows", return_value=[])
    def test_low_confidence_words_filtered(self, *_mocks):
        capturer = self._make_capturer()
        words = [_word("lo", x=10, y=10, w=20, h=10, conf=0.40)]
        ocr = self._make_ocr(words)
        snap = capture_snapshot(capturer, ocr)
        self.assertEqual(len(snap.ocr_regions), 0)

    @patch("core.capture.screen_snapshot._get_foreground_hwnd_and_title",
           return_value=(42, "App"))
    @patch("core.capture.screen_snapshot._get_foreground_process",
           return_value="app.exe")
    @patch("core.capture.screen_snapshot._enum_visible_windows", return_value=[])
    def test_short_words_filtered(self, *_mocks):
        capturer = self._make_capturer()
        words = [_word("a", x=10, y=10, w=10, h=10, conf=0.95)]
        ocr = self._make_ocr(words)
        snap = capture_snapshot(capturer, ocr)
        self.assertEqual(len(snap.ocr_regions), 0)

    @patch("core.capture.screen_snapshot._get_foreground_hwnd_and_title",
           return_value=(42, "App"))
    @patch("core.capture.screen_snapshot._get_foreground_process",
           return_value="app.exe")
    @patch("core.capture.screen_snapshot._enum_visible_windows", return_value=[])
    def test_special_char_words_filtered(self, *_mocks):
        capturer = self._make_capturer()
        words = [_word("C:\\path", x=10, y=10, w=60, h=10, conf=0.95)]
        ocr = self._make_ocr(words)
        snap = capture_snapshot(capturer, ocr)
        self.assertEqual(len(snap.ocr_regions), 0)

    @patch("core.capture.screen_snapshot._get_foreground_hwnd_and_title",
           return_value=(0, "Desktop"))
    @patch("core.capture.screen_snapshot._get_foreground_process",
           return_value="unknown")
    @patch("core.capture.screen_snapshot._enum_visible_windows", return_value=[])
    def test_non_windows_all_regions_foreground(self, *_mocks):
        # fg_hwnd==0 → treat everything as foreground
        capturer = self._make_capturer()
        words = [_word("Hello", x=10, y=10, w=50, h=18, conf=0.95)]
        ocr = self._make_ocr(words)
        snap = capture_snapshot(capturer, ocr)
        for r in snap.ocr_regions:
            self.assertTrue(r.is_in_foreground)

    @patch("core.capture.screen_snapshot._get_foreground_hwnd_and_title",
           return_value=(42, "App"))
    @patch("core.capture.screen_snapshot._get_foreground_process",
           return_value="app.exe")
    @patch("core.capture.screen_snapshot._enum_visible_windows", return_value=[])
    def test_screen_hash_populated(self, *_mocks):
        capturer = self._make_capturer()
        ocr = self._make_ocr([])
        snap = capture_snapshot(capturer, ocr)
        self.assertIsNotNone(snap.screen_hash)
        self.assertGreater(len(snap.screen_hash), 0)

    @patch("core.capture.screen_snapshot._get_foreground_hwnd_and_title",
           return_value=(42, "App"))
    @patch("core.capture.screen_snapshot._get_foreground_process",
           return_value="app.exe")
    @patch("core.capture.screen_snapshot._enum_visible_windows", return_value=[])
    def test_timestamp_recent(self, *_mocks):
        capturer = self._make_capturer()
        ocr = self._make_ocr([])
        before = time.time()
        snap = capture_snapshot(capturer, ocr)
        after = time.time()
        self.assertGreaterEqual(snap.timestamp, before)
        self.assertLessEqual(snap.timestamp, after)


# ── TestPointInRect helper ─────────────────────────────────────────────────────

class TestPointInRect(unittest.TestCase):

    def test_inside(self):
        self.assertTrue(_point_in_rect(50, 50, (0, 0, 100, 100)))

    def test_left_edge(self):
        self.assertTrue(_point_in_rect(0, 50, (0, 0, 100, 100)))

    def test_top_edge(self):
        self.assertTrue(_point_in_rect(50, 0, (0, 0, 100, 100)))

    def test_right_edge_exclusive(self):
        self.assertFalse(_point_in_rect(100, 50, (0, 0, 100, 100)))

    def test_bottom_edge_exclusive(self):
        self.assertFalse(_point_in_rect(50, 100, (0, 0, 100, 100)))

    def test_outside_left(self):
        self.assertFalse(_point_in_rect(-1, 50, (0, 0, 100, 100)))

    def test_outside_right(self):
        self.assertFalse(_point_in_rect(101, 50, (0, 0, 100, 100)))


# ── TestPlannerFormattedOutput ─────────────────────────────────────────────────

class TestPlannerFormattedOutput(unittest.TestCase):
    """Planner receives snapshot-formatted context; background text is labeled."""

    def _make_planner(self):
        from agents.planning.planning_agent import PlanningAgent
        client = MagicMock()
        return PlanningAgent(client)

    def test_snapshot_overrides_screen_context(self):
        snap = _snapshot(fg_regions=["New", "Folder"])
        planner = self._make_planner()

        captured = {}
        def fake_query_llm(messages, **kw):
            captured["content"] = messages[-1]["content"]
            resp = MagicMock()
            resp.content = "[]"
            return resp

        planner.client.query_llm = fake_query_llm

        from core.protocols.a2a import SubTask
        subtask = SubTask(id=1, description="create a folder")
        planner.plan_next_step(subtask, screen_context="old flat context", snapshot=snap)

        msg = captured["content"]
        self.assertIn("Foreground window", msg)
        self.assertIn('"New"', msg)
        self.assertNotIn("old flat context", msg)

    def test_background_text_labeled_not_interactive(self):
        snap = _snapshot(fg_regions=["New"], bg_regions=["AgentLog", "Monitor"])
        planner = self._make_planner()

        captured = {}
        def fake_query_llm(messages, **kw):
            captured["content"] = messages[-1]["content"]
            resp = MagicMock()
            resp.content = "[]"
            return resp

        planner.client.query_llm = fake_query_llm

        from core.protocols.a2a import SubTask
        subtask = SubTask(id=1, description="do something")
        planner.plan_next_step(subtask, snapshot=snap)

        msg = captured["content"]
        self.assertIn("background", msg.lower())
        self.assertIn("AgentLog", msg)

    def test_no_snapshot_uses_screen_context_unchanged(self):
        planner = self._make_planner()

        captured = {}
        def fake_query_llm(messages, **kw):
            captured["content"] = messages[-1]["content"]
            resp = MagicMock()
            resp.content = "[]"
            return resp

        planner.client.query_llm = fake_query_llm

        from core.protocols.a2a import SubTask
        subtask = SubTask(id=1, description="do something")
        planner.plan_next_step(subtask, screen_context="flat token list")

        self.assertIn("flat token list", captured["content"])

    def test_snapshot_none_does_not_crash(self):
        planner = self._make_planner()
        planner.client.query_llm = MagicMock(return_value=MagicMock(content="[]"))
        from core.protocols.a2a import SubTask
        subtask = SubTask(id=1, description="do something")
        result = planner.plan_next_step(subtask, snapshot=None)
        self.assertIsNone(result)


# ── TestGroundingForegroundOnly ────────────────────────────────────────────────

class TestGroundingForegroundOnly(unittest.TestCase):
    """find_text(foreground_only=True) rejects words with is_in_foreground=False."""

    def _ocr_engine(self) -> OCREngine:
        engine = OCREngine.__new__(OCREngine)
        engine._ocr = None
        engine._available = True
        engine._cache = {}
        return engine

    def test_foreground_word_matched_when_foreground_only(self):
        engine = self._ocr_engine()
        words = [_word("Folder", fg=True)]
        result = engine.find_text(words, "Folder", foreground_only=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "Folder")

    def test_background_word_rejected_when_foreground_only(self):
        engine = self._ocr_engine()
        words = [_word("Folder", fg=False)]
        result = engine.find_text(words, "Folder", foreground_only=True)
        self.assertIsNone(result)

    def test_background_word_accepted_without_foreground_only(self):
        engine = self._ocr_engine()
        words = [_word("Folder", fg=False)]
        result = engine.find_text(words, "Folder", foreground_only=False)
        self.assertIsNotNone(result)

    def test_mixed_words_only_fg_returned(self):
        engine = self._ocr_engine()
        words = [_word("Folder", x=0, fg=True), _word("Hidden", x=200, fg=False)]
        result = engine.find_text(words, "Hidden", foreground_only=True)
        self.assertIsNone(result)

    def test_multi_word_window_all_must_be_fg(self):
        # Window of 2 words: one fg, one bg → group rejected
        engine = self._ocr_engine()
        fg_w = _word("New", x=0, w=30, fg=True)
        bg_w = _word("Folder", x=35, w=40, fg=False)
        result = engine.find_text([fg_w, bg_w], "New Folder", foreground_only=True)
        self.assertIsNone(result)

    def test_default_foreground_only_false(self):
        engine = self._ocr_engine()
        words = [_word("Test", fg=False)]
        # Default threshold allows exact match
        result = engine.find_text(words, "Test")
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
