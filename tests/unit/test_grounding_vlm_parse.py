# tests/unit/test_grounding_vlm_parse.py
"""Regression tests for UIGroundingAgent._parse_coords — formats UI-TARS emits
in practice, including malformed bracket counts seen in live runs.
"""
import sys

sys.path.insert(0, ".")

from agents.grounding.grounding_agent import UIGroundingAgent


def _agent():
    """Bare agent instance — _parse_coords needs no collaborators."""
    return UIGroundingAgent.__new__(UIGroundingAgent)


_W, _H = 1000, 1000   # identity scaling for 0-1000 coords


class TestParseCoordsBracketTolerance:

    def test_standard_four_value_bbox(self):
        r = _agent()._parse_coords("click(start_box='[[100, 200, 300, 400]]')", _W, _H)
        assert r is not None
        x, y, _ = r
        assert (x, y) == (200, 300)

    def test_triple_bracket_two_value_form(self):
        """Regression: live UI-TARS output 'click(start_box='[[[287, 569]')'."""
        r = _agent()._parse_coords("click(start_box='[[[287, 569]')", _W, _H)
        assert r is not None
        x, y, _ = r
        assert (x, y) == (287, 569)

    def test_triple_bracket_four_value_form(self):
        r = _agent()._parse_coords("click(start_box='[[[10, 20, 30, 40]]]')", _W, _H)
        assert r is not None
        x, y, _ = r
        assert (x, y) == (20, 30)

    def test_not_found_returns_none(self):
        assert _agent()._parse_coords("not_found()", _W, _H) is None

    def test_garbage_returns_none(self):
        assert _agent()._parse_coords("the element is near the top", _W, _H) is None
