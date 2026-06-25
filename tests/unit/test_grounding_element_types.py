# tests/unit/test_grounding_element_types.py
"""Unit tests for OCR element_type semantics in the grounding pipeline.

Test 1: OCRWord with element_type="document_text" is rejected by
        find_text(foreground_only=True).

Test 2: _execute_step() accepts document_text grounding (Start menu / search
        overlays) — these are valid targets despite not appearing as foreground windows.

Additional coverage:
  - element_type propagates through the merged-word path in find_text()
  - GroundingResult.element_type defaults to "foreground_interactive"
  - UIA / VLM grounding results keep element_type="foreground_interactive"
  - OCR grounding result carries the matched word's element_type
"""
import unittest
from unittest.mock import MagicMock

from agents.grounding.grounding_agent import (
    ElementCache,
    GroundingResult,
    OCREngine,
    OCRWord,
)
from core.protocols.a2a import ActionStep

# ── helpers ────────────────────────────────────────────────────────────────────

def _word(text, fg=True, etype="foreground_interactive", x=0, y=0, w=60, h=20, conf=0.95):
    return OCRWord(text=text, x=x, y=y, w=w, h=h, conf=conf,
                   is_in_foreground=fg, element_type=etype)


def _ocr_engine():
    engine = OCREngine.__new__(OCREngine)
    engine._ocr = None
    engine._available = True
    engine._cache = {}
    return engine


def _grounding_result(found=True, conf=0.90, etype="foreground_interactive", x=100, y=200):
    return GroundingResult(
        x=x, y=y, confidence=conf, found=found,
        latency_ms=10.0, target="SomeButton", method="ocr_direct",
        element_type=etype,
    )


# ── Test 1: find_text rejects document_text words when foreground_only=True ───

class TestFindTextRejectsDocumentText(unittest.TestCase):

    def test_document_text_word_rejected_foreground_only(self):
        """Core spec requirement: element_type='document_text' → None when foreground_only."""
        engine = _ocr_engine()
        words = [_word("Folder", fg=True, etype="document_text")]
        result = engine.find_text(words, "Folder", foreground_only=True)
        self.assertIsNone(result)

    def test_foreground_interactive_word_accepted(self):
        engine = _ocr_engine()
        words = [_word("Folder", fg=True, etype="foreground_interactive")]
        result = engine.find_text(words, "Folder", foreground_only=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "Folder")

    def test_background_fg_false_rejected(self):
        """is_in_foreground=False guard still fires independently of element_type."""
        engine = _ocr_engine()
        words = [_word("Folder", fg=False, etype="foreground_interactive")]
        result = engine.find_text(words, "Folder", foreground_only=True)
        self.assertIsNone(result)

    def test_both_guards_needed_fg_false_and_doc_text(self):
        """Both flags False → rejected (same result, dual guard)."""
        engine = _ocr_engine()
        words = [_word("Folder", fg=False, etype="document_text")]
        result = engine.find_text(words, "Folder", foreground_only=True)
        self.assertIsNone(result)

    def test_foreground_only_false_accepts_document_text(self):
        """Default foreground_only=False: element_type doesn't matter."""
        engine = _ocr_engine()
        words = [_word("Folder", fg=False, etype="document_text")]
        result = engine.find_text(words, "Folder")
        self.assertIsNotNone(result)

    def test_multi_word_group_one_doc_text_rejected(self):
        """In a 2-word group, one document_text word rejects the whole group."""
        engine = _ocr_engine()
        w1 = _word("New",    x=0,  w=30, fg=True, etype="foreground_interactive")
        w2 = _word("Folder", x=35, w=40, fg=True, etype="document_text")
        result = engine.find_text([w1, w2], "New Folder", foreground_only=True)
        self.assertIsNone(result)

    def test_multi_word_group_all_interactive_accepted(self):
        """All foreground_interactive in group → accepted."""
        engine = _ocr_engine()
        w1 = _word("New",    x=0,  w=30, fg=True, etype="foreground_interactive")
        w2 = _word("Folder", x=35, w=40, fg=True, etype="foreground_interactive")
        result = engine.find_text([w1, w2], "New Folder", foreground_only=True)
        self.assertIsNotNone(result)


# ── Test 2: _execute_step allows document_text grounding (Start menu / search) ──

class TestExecuteStepRejectsNonInteractive(unittest.TestCase):
    """_execute_step() now allows OCR hits with element_type="document_text" so
    that Start menu search results (which Windows 11 does not expose as a
    foreground window) can be clicked correctly.

    The old hard-rejection was removed because:
    - GUI window masking (exclude_regions) prevents log-text false positives
    - Windows 11 Start menu does not appear in GetForegroundWindow()
    - High-confidence OCR exact matches (score=1.0) in search panels ARE interactive
    """

    def _make_orchestrator(self, ground_result: GroundingResult):
        from core.orchestrator import OrchestratorConfig, TaskOrchestrator

        grounder = MagicMock()
        grounder.min_confidence = 0.5
        grounder.ground.return_value = ground_result

        actor = MagicMock()
        actor.execute.return_value = True

        capturer = MagicMock()
        reflector = MagicMock()
        task_memory = MagicMock()
        task_memory.find_similar.return_value = None

        ocr = MagicMock()
        ocr.is_available.return_value = False
        ocr.extract.return_value = []


        orch = TaskOrchestrator.__new__(TaskOrchestrator)
        orch.grounder = grounder
        orch.actor = actor
        orch.capturer = capturer
        orch.reflector = reflector
        orch.memory = task_memory
        orch.config = OrchestratorConfig()
        orch.log = lambda msg: None
        orch._stop_event = MagicMock()
        orch._stop_event.is_set.return_value = False
        orch._ocr = ocr
        orch._extracted_data = {}
        orch._screen_w = 1920
        orch._screen_h = 1080
        # Burst executor (needed by _execute_subtask but not _execute_step)
        from core.executor.burst_executor import BurstExecutor
        orch.burst_executor = MagicMock(spec=BurstExecutor)

        return orch, grounder, actor

    def _click_step(self):
        return ActionStep(
            id=1, subtask_id=1,
            action_type="click",
            target="SomeButton",
            value=None, key=None,
            description="Click SomeButton",
            verification="Button clicked",
        )

    def test_document_text_grounding_now_allowed(self):
        """_execute_step now accepts document_text hits (Start menu / search results)."""
        result = _grounding_result(found=True, conf=0.95, etype="document_text")
        orch, _, actor = self._make_orchestrator(result)
        step = self._click_step()

        ok = orch._execute_step(step)

        self.assertTrue(ok)

    def test_actor_called_for_document_text(self):
        """actor.execute IS called for document_text hits (no longer blocked)."""
        result = _grounding_result(found=True, conf=0.95, etype="document_text")
        orch, _, actor = self._make_orchestrator(result)
        step = self._click_step()

        orch._execute_step(step)

        actor.execute.assert_called_once()

    def test_foreground_interactive_grounding_calls_actor(self):
        """Control: interactive grounding → actor.execute IS called."""
        result = _grounding_result(found=True, conf=0.95, etype="foreground_interactive")
        orch, _, actor = self._make_orchestrator(result)
        step = self._click_step()

        orch._execute_step(step)

        actor.execute.assert_called_once()

    def test_right_click_document_text_allowed(self):
        """right_click with document_text grounding is now allowed through."""
        result = _grounding_result(found=True, conf=0.95, etype="document_text")
        orch, _, actor = self._make_orchestrator(result)
        step = ActionStep(
            id=1, subtask_id=1, action_type="right_click",
            target="Desktop", value=None, key=None,
            description="Right-click desktop", verification="",
        )

        ok = orch._execute_step(step)

        self.assertTrue(ok)
        actor.execute.assert_called_once()

    def test_double_click_document_text_allowed(self):
        """double_click with document_text grounding is now allowed through."""
        result = _grounding_result(found=True, conf=0.95, etype="document_text")
        orch, _, actor = self._make_orchestrator(result)
        step = ActionStep(
            id=1, subtask_id=1, action_type="double_click",
            target="Icon", value=None, key=None,
            description="Double-click icon", verification="",
        )

        ok = orch._execute_step(step)

        self.assertTrue(ok)
        actor.execute.assert_called_once()

    def test_type_step_not_checked(self):
        """Non-click actions bypass the element_type check entirely."""
        orch, grounder, actor = self._make_orchestrator(
            _grounding_result(etype="document_text")
        )
        grounder.ground.return_value = _grounding_result(etype="document_text")
        actor.execute.return_value = True

        step = ActionStep(
            id=1, subtask_id=1, action_type="type",
            target=None, value="hello", key=None,
            description="Type hello", verification="",
        )

        ok = orch._execute_step(step)

        # actor.execute should be called for type (grounder not even used)
        self.assertTrue(ok)


# ── Additional structural tests ───────────────────────────────────────────────

class TestGroundingResultDefaults(unittest.TestCase):

    def test_default_element_type_is_foreground_interactive(self):
        r = GroundingResult(x=0, y=0, confidence=1.0, found=True,
                            latency_ms=5.0, target="x")
        self.assertEqual(r.element_type, "foreground_interactive")

    def test_failed_result_default_element_type(self):
        r = GroundingResult(x=0, y=0, confidence=0.0, found=False,
                            latency_ms=5.0, target="x", method="failed")
        self.assertEqual(r.element_type, "foreground_interactive")


class TestOCRWordDefaults(unittest.TestCase):

    def test_element_type_default_is_document_text(self):
        w = OCRWord(text="Hello", x=0, y=0, w=50, h=20, conf=0.9)
        self.assertEqual(w.element_type, "document_text")

    def test_is_in_foreground_default_true(self):
        w = OCRWord(text="Hello", x=0, y=0, w=50, h=20, conf=0.9)
        self.assertTrue(w.is_in_foreground)

    def test_element_type_settable(self):
        w = OCRWord(text="Hello", x=0, y=0, w=50, h=20, conf=0.9)
        w.element_type = "foreground_interactive"
        self.assertEqual(w.element_type, "foreground_interactive")


class TestMergedWordElementType(unittest.TestCase):
    """element_type of merged multi-word groups follows the weakest word."""

    def _engine(self):
        return _ocr_engine()

    def test_all_interactive_merged_interactive(self):
        engine = self._engine()
        w1 = _word("New",    x=0,  w=30, fg=True, etype="foreground_interactive")
        w2 = _word("Folder", x=35, w=40, fg=True, etype="foreground_interactive")
        result = engine.find_text([w1, w2], "New Folder")
        self.assertIsNotNone(result)
        self.assertEqual(result.element_type, "foreground_interactive")

    def test_one_doc_text_merged_doc_text(self):
        engine = self._engine()
        w1 = _word("New",    x=0,  w=30, fg=True, etype="foreground_interactive")
        w2 = _word("Folder", x=35, w=40, fg=True, etype="document_text")
        result = engine.find_text([w1, w2], "New Folder")
        self.assertIsNotNone(result)
        self.assertEqual(result.element_type, "document_text")

    def test_single_word_retains_element_type(self):
        engine = self._engine()
        w = _word("Save", etype="foreground_interactive")
        result = engine.find_text([w], "Save")
        self.assertIsNotNone(result)
        self.assertEqual(result.element_type, "foreground_interactive")

    def test_single_doc_text_word_retains_element_type(self):
        engine = self._engine()
        w = _word("readme", etype="document_text")
        result = engine.find_text([w], "readme")
        self.assertIsNotNone(result)
        self.assertEqual(result.element_type, "document_text")


class TestElementCachePreservesElementType(unittest.TestCase):

    def test_cache_round_trip(self):
        cache = ElementCache()
        cache.put("target", 100, 200, 0.95, "ocr_direct", "hash123", "document_text")
        result = cache.get("target", "hash123")
        self.assertIsNotNone(result)
        x, y, conf, method, element_type = result
        self.assertEqual(element_type, "document_text")

    def test_cache_default_element_type(self):
        cache = ElementCache()
        cache.put("target", 100, 200, 0.95, "uia", "hash123")
        result = cache.get("target", "hash123")
        x, y, conf, method, element_type = result
        self.assertEqual(element_type, "foreground_interactive")

    def test_cache_miss_returns_none(self):
        cache = ElementCache()
        self.assertIsNone(cache.get("missing", "hash"))


if __name__ == "__main__":
    unittest.main()
