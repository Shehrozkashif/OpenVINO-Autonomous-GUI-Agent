"""
Live grounding test — tests Stage 0 (UIA), Stage 1 (OCR), Stage 2 (VLM),
coordinate parsing, and mouse click accuracy.

Requires: OpenVINO Model Server running (python start.py), real display, Windows.
Run: python tests/test_grounding_live.py
"""
import sys
import time
sys.path.insert(0, ".")

from core.capture.screenshot import ScreenCapture
from core.pipeline.ovms_client import OVMSClient
from agents.grounding.grounding_agent import UIGroundingAgent, OCREngine
from tools.desktop_control.controller import DesktopController


def sep(label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print('='*55)


def ground_and_report(grounder, target):
    print(f"\n  Target: '{target}'")
    r = grounder.ground(target)
    status = "FOUND" if r.found else "NOT FOUND"
    print(f"  Result: {status}")
    print(f"  Coords: ({r.x}, {r.y})")
    print(f"  Conf:   {r.confidence:.2f}")
    print(f"  Method: {r.method}")
    print(f"  Time:   {r.latency_ms:.0f}ms")
    return r


def test_parse_coords():
    sep("1. VLM coordinate parsing (_parse_coords)")

    from agents.grounding.grounding_agent import UIGroundingAgent
    capturer = ScreenCapture()
    client = OVMSClient()
    ocr = OCREngine()
    grounder = UIGroundingAgent(client, capturer, ocr=ocr)

    cases = [
        # (input_text, description, expect_found)
        ('{"x": 0.5, "y": 0.3, "confidence": 0.9, "found": true}',
         "fractional JSON 0-1 normalised", True),
        ('{"x": 378, "y": 654, "confidence": 0.92, "found": true}',
         "JSON 0-1000 integers (UI-TARS native scale) — must NOT be treated as display pixels", True),
        ('{"x": 0.0, "y": 0.0, "confidence": 0.7, "found": false}',
         "found:false", False),
        ("click(start_box='[[312, 972, 360, 1000]]')",
         "native action format (0-1000 scale) — primary UI-TARS output", True),
        ('<point>500 300</point>',
         "<point> 0-1000 scale", True),
        ('<point>0.5 0.3</point>',
         "<point> fractional", True),
        ('(0.2,0.3),(0.8,0.7)',
         "bbox fractional", True),
        ('(192,108),(576,324)',
         "bbox pixel", True),
        ('No element found here',
         "unrecognised — should return None", False),
        ('not_found()',
         "explicit not_found() — should return None", False),
    ]

    all_pass = True
    for text, desc, expect_found in cases:
        result = grounder._parse_coords(text, 1920, 1080)
        found = result is not None
        ok = (found == expect_found)
        mark = "PASS" if ok else "FAIL"
        print(f"\n  [{mark}] {desc}")
        print(f"       Input:  {text[:60]}")
        print(f"       Output: {result}")
        if not ok:
            all_pass = False

    print(f"\n  {'All parsing tests PASSED.' if all_pass else 'SOME TESTS FAILED!'}")
    return all_pass


def test_uia_grounding():
    sep("2. Stage 0 — Windows UIA grounding")
    capturer = ScreenCapture()
    client = OVMSClient()
    ocr = OCREngine()
    grounder = UIGroundingAgent(client, capturer, ocr=ocr)

    targets = [
        "Start",               # Windows Start button (always in taskbar)
        "Taskbar",             # Taskbar itself
        "Show desktop",        # Show desktop button (bottom-right)
        "Task View",           # Task View button in taskbar
    ]
    results = []
    for t in targets:
        r = ground_and_report(grounder, t)
        results.append((t, r))
    return results


def test_ocr_grounding():
    sep("3. Stage 1 — OCR text grounding")
    capturer = ScreenCapture()
    client = OVMSClient()
    ocr = OCREngine()
    grounder = UIGroundingAgent(client, capturer, ocr=ocr)

    print("  (Reading current screen — open any window with visible text for best results)")
    img = capturer.capture()
    img.thumbnail((960, 540))
    words = ocr.extract(img)
    visible_words = [w.text for w in words if w.conf >= 0.75 and len(w.text) > 2][:5]
    print(f"  Visible OCR words: {visible_words}")

    results = []
    for word in visible_words[:3]:
        r = ground_and_report(grounder, word)
        results.append((word, r))
    return results


def test_click_accuracy():
    sep("4. Click accuracy test")
    capturer = ScreenCapture()
    client = OVMSClient()
    ocr = OCREngine()
    grounder = UIGroundingAgent(client, capturer, ocr=ocr)
    controller = DesktopController()

    print("\n  Test: Win key → open Start menu → Escape to close")
    before_hash = _screen_hash(capturer)
    controller.press_key("winleft")
    time.sleep(1.0)
    after_hash = _screen_hash(capturer)
    changed = before_hash != after_hash
    print(f"  Win key: screen changed = {changed} ({'PASS' if changed else 'FAIL - Start menu did not open'})")
    time.sleep(0.3)
    controller.press_key("escape")
    time.sleep(0.5)

    print("\n  Test: Click at center of screen → record if click registers")
    cx, cy = 960, 540
    result = controller.click(cx, cy)
    print(f"  Click at ({cx}, {cy}): returned {result} ({'PASS' if result else 'FAIL'})")

    print("\n  Test: Ground and click 'Start' via UIA, verify screen changes")
    r = grounder.ground("Start")
    if r.found:
        print(f"  'Start' found at ({r.x}, {r.y}) via {r.method}")
        before = _screen_hash(capturer)
        controller.click(r.x, r.y)
        time.sleep(1.0)
        after = _screen_hash(capturer)
        changed = before != after
        print(f"  Screen changed after clicking Start: {changed} ({'PASS' if changed else 'FAIL'})")
        controller.press_key("escape")
        time.sleep(0.3)
    else:
        print("  'Start' not found — skipping click test")


def _screen_hash(capturer):
    import imagehash
    img = capturer.capture()
    img.thumbnail((320, 180))
    return str(imagehash.phash(img))


def test_vlm_direct():
    sep("5. Stage 2 — VLM direct coordinate test")
    capturer = ScreenCapture()
    client = OVMSClient()
    ocr = OCREngine()
    grounder = UIGroundingAgent(client, capturer, ocr=ocr)

    print("\n  Forcing VLM call for 'taskbar clock' (UIA+OCR are very reliable here,")
    print("  so we call _vlm_coords directly to test coordinate quality):")

    import base64
    import io
    img = capturer.capture()
    img.thumbnail((960, 540))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    elements_to_find = ["taskbar clock", "Start button", "desktop background"]
    for elem in elements_to_find:
        print(f"\n  VLM query: '{elem}'")
        result = grounder._vlm_coords(elem, img_b64)
        if result:
            x, y, conf, method, element_type = result
            print(f"  → screen({x}, {y}), conf={conf:.2f}, method={method}, type={element_type}")
        else:
            print("  → not found (VLM returned null)")


if __name__ == "__main__":
    print("\nDesktop Grounding Live Test Suite")
    print("==================================")
    print("This test sends real mouse/keyboard events to your display.")
    print("Keep the desktop visible and don't interact during the test.\n")

    test_parse_coords()
    test_uia_grounding()
    test_ocr_grounding()
    test_click_accuracy()
    test_vlm_direct()

    print("\n\nAll tests complete.")
