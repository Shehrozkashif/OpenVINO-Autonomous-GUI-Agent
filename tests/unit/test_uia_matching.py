# tests/unit/test_uia_matching.py
"""Two-tier UIA matching (core/windows_uia._walk_and_match).

Regression for a live failure: the planner targeted "NEW" (meaning Zoom's
"New meeting" button) and the walk returned a decorative "NEW" feature badge
(TextControl) at conf=1.00 — exact name match pruned the tree before the
real button was reached. Clicking decoration changes zero pixels, which fed
the delta=0 dead-click loop.

Rules under test:
  1. An INTERACTIVE substring match beats a DECORATIVE exact match.
  2. Decorative matches still work as a fallback when nothing interactive
     matches (some apps only label a clickable region via a text child),
     at a discounted confidence.
  3. An exact interactive match wins with conf=1.0.
  4. GetClickablePoint() is preferred over the bounding-rect center.
"""
from core.windows_uia import _walk_and_match


class _Rect:
    def __init__(self, left, top, right, bottom):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom


class _FakeCtrl:
    """Minimal stand-in for a uiautomation control."""

    def __init__(self, name, ctype, rect, children=(), clickable=None):
        self.Name = name
        self.ControlTypeName = ctype
        self.BoundingRectangle = rect
        self._children = list(children)
        self._clickable = clickable   # (x, y) or None → pattern unsupported

    def GetChildren(self):
        return self._children

    def GetValuePattern(self):
        raise RuntimeError("no ValuePattern")

    def GetClickablePoint(self):
        if self._clickable is None:
            raise RuntimeError("no clickable point")
        return (*self._clickable, True)


def _root(*children):
    return _FakeCtrl("", "WindowControl", _Rect(0, 0, 1852, 963), children)


BADGE = _FakeCtrl("NEW", "TextControl", _Rect(1100, 20, 1168, 32))          # center (1134, 26)
BUTTON = _FakeCtrl("New meeting", "ButtonControl", _Rect(600, 400, 760, 460))  # center (680, 430)


class TestInteractivePreference:

    def test_button_substring_beats_badge_exact(self):
        """'NEW' badge (TextControl, exact) must lose to the 'New meeting'
        ButtonControl (substring) — decoration is unclickable."""
        result = _walk_and_match(_root(BADGE, BUTTON), "new", 0.65, max_depth=5)
        assert result is not None
        x, y, conf = result
        assert (x, y) == (680, 430)

    def test_order_does_not_matter(self):
        """Same outcome when the interactive control comes first in the tree."""
        result = _walk_and_match(_root(BUTTON, BADGE), "new", 0.65, max_depth=5)
        assert (result[0], result[1]) == (680, 430)

    def test_decorative_fallback_when_nothing_interactive(self):
        """A text-only match is still returned (discounted) when no
        interactive control matches at all."""
        result = _walk_and_match(_root(BADGE), "new", 0.65, max_depth=5)
        assert result is not None
        x, y, conf = result
        assert (x, y) == (1134, 26)
        assert conf < 0.9   # visibly weaker than an interactive hit

    def test_exact_interactive_match_is_conf_1(self):
        exact = _FakeCtrl("save", "ButtonControl", _Rect(10, 10, 110, 40))
        result = _walk_and_match(_root(exact), "save", 0.65, max_depth=5)
        assert result == (60, 25, 1.0)


class TestClickablePoint:

    def test_clickable_point_preferred_over_rect_center(self):
        ctrl = _FakeCtrl("save", "ButtonControl", _Rect(0, 0, 1000, 40),
                         clickable=(55, 20))
        result = _walk_and_match(_root(ctrl), "save", 0.65, max_depth=5)
        assert (result[0], result[1]) == (55, 20)

    def test_rect_center_when_pattern_unsupported(self):
        ctrl = _FakeCtrl("save", "ButtonControl", _Rect(0, 0, 100, 40))
        result = _walk_and_match(_root(ctrl), "save", 0.65, max_depth=5)
        assert (result[0], result[1]) == (50, 20)
