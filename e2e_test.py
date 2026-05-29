"""
e2e_test.py — Full pipeline end-to-end test.
Tests every stage with clear pass/fail reporting.
Run: python e2e_test.py
"""
import sys
import time
import json

from loguru import logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="DEBUG", colorize=True)

# ── imports ──────────────────────────────────────────────────────────────────
from core.pipeline.ollama_client import OllamaClient
from agents.router.router_agent import RouterAgent
from agents.planning.planning_agent import PlanningAgent
from agents.grounding.grounding_agent import UIGroundingAgent
from agents.action.action_agent import ActionExecutionAgent
from agents.reflection.reflection_agent import ReflectionAgent
from core.capture.screenshot import ScreenCapture
from tools.desktop_control.controller import DesktopController
from core.protocols.a2a import SubTask

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"
results = {}

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════
# 1. Backend health
# ═══════════════════════════════════════════════════════════════
section("1. Backend Health")
client = OllamaClient()
health = client.check_health()
all_ok = all(v == "OK" for v in health.values())
for k, v in health.items():
    print(f"  {'✅' if v=='OK' else '❌'} {k}: {v}")
results["backend"] = PASS if all_ok else FAIL
print(f"\n  → {results['backend']}")
if not all_ok:
    print("  Pull model first: ollama pull qwen2.5vl:3b")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# 2. Tool server
# ═══════════════════════════════════════════════════════════════
section("2. Tool Server")
controller = DesktopController()
if controller.is_server_running():
    print("  ✅ Tool server on port 8015: OK")
    results["tool_server"] = PASS
else:
    print("  ❌ Tool server not reachable — run: python -m tools.desktop_control.server")
    results["tool_server"] = FAIL
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# 3. Screen capture
# ═══════════════════════════════════════════════════════════════
section("3. Screen Capture")
cap = ScreenCapture()
img = cap.capture()
print(f"  Resolution: {img.width}×{img.height}")
b64 = cap.capture_as_base64()
print(f"  Base64 size: {len(b64)//1024} KB")
results["screen_capture"] = PASS if img.width > 0 and len(b64) > 0 else FAIL
print(f"\n  → {results['screen_capture']}")

# ═══════════════════════════════════════════════════════════════
# 4. OCR grounding (fast — no VLM)
# ═══════════════════════════════════════════════════════════════
section("4. OCR Grounding (no VLM, should be <5s)")
from core.grounding.ocr_engine import OCREngine
ocr = OCREngine()
img_small = img.copy()
img_small.thumbnail((960, 540))
t = time.time()
words = ocr.extract(img_small)
elapsed = time.time() - t
print(f"  Words found: {len(words)}  ({elapsed:.1f}s)")
for w in words[:8]:
    print(f"    '{w.text}' at ({w.x},{w.y})")
results["ocr"] = PASS if len(words) > 0 else FAIL
print(f"\n  → {results['ocr']}")

# ═══════════════════════════════════════════════════════════════
# 5. Router — task decomposition
# ═══════════════════════════════════════════════════════════════
section("5. Router — Task Decomposition")
router = RouterAgent(client)
t = time.time()
task_id, subtasks = router.decompose("open the terminal")
elapsed = time.time() - t
print(f"  Subtasks ({elapsed:.1f}s):")
for st in subtasks:
    print(f"    [{st.id}] {st.description}  depends_on={st.depends_on}")
ok = len(subtasks) == 1 and "terminal" in subtasks[0].description.lower()
results["router"] = PASS if ok else FAIL
print(f"\n  → {results['router']}  (expected: 1 subtask about terminal)")

# ═══════════════════════════════════════════════════════════════
# 6. Planner — step generation
# ═══════════════════════════════════════════════════════════════
section("6. Planner — Step Generation")
planner = PlanningAgent(client)
t = time.time()
steps = planner.plan(subtasks[0])
elapsed = time.time() - t
print(f"  Steps ({elapsed:.1f}s):")
for s in steps:
    print(f"    [{s.id}] {s.action_type:12s} key={s.key!r:12} value={s.value!r}")
# Accept any valid sequence that would open a terminal:
#   Option A: super → type "gnome-terminal" → enter  (search launcher)
#   Option B: hotkey ctrl+alt+t                       (direct shortcut)
#   Option C: any sequence containing at least 1 step
uses_launcher = any(s.action_type == "key_press" and s.key == "super" for s in steps)
uses_direct   = any(s.action_type == "hotkey" and "t" in (s.key or "") for s in steps)
has_steps     = len(steps) >= 1
ok = (uses_launcher or uses_direct) and has_steps
results["planner"] = PASS if ok else FAIL
print(f"\n  → {results['planner']}  (accepted: launcher or direct hotkey, got {len(steps)} step(s))")

# ═══════════════════════════════════════════════════════════════
# 7. Action executor — keyboard steps (no grounding)
# ═══════════════════════════════════════════════════════════════
section("7. Action Execution — Keyboard Steps")
actor = ActionExecutionAgent(controller)
from core.protocols.a2a import ActionStep

print("  Executing: key_press 'super'  (opens GNOME Activities)")
step_super = ActionStep(id=1, subtask_id=1, action_type="key_press", key="super",
                        target=None, value=None, description="open launcher", verification="")
ok1 = actor.execute(step_super)
time.sleep(1.2)

print("  Executing: type 'gnome-terminal'")
step_type = ActionStep(id=2, subtask_id=1, action_type="type", value="gnome-terminal",
                       key=None, target=None, description="type app name", verification="")
ok2 = actor.execute(step_type)
time.sleep(0.5)

print("  Executing: key_press 'enter'")
step_enter = ActionStep(id=3, subtask_id=1, action_type="key_press", key="enter",
                        target=None, value=None, description="launch", verification="")
ok3 = actor.execute(step_enter)
time.sleep(3.0)

results["action_keyboard"] = PASS if (ok1 and ok2 and ok3) else FAIL
print(f"\n  → {results['action_keyboard']}  (check: did terminal open?)")

# ═══════════════════════════════════════════════════════════════
# 8. OCR-based click grounding
# ═══════════════════════════════════════════════════════════════
section("8. OCR Grounding + Click — Text Element")
grounder = UIGroundingAgent(client, cap, min_confidence=0.5)

# Find a word we know is on screen from step 4
if words:
    target_text = words[0].text.strip()[:20]
    print(f"  Grounding target (from OCR step): '{target_text}'")
    t = time.time()
    result = grounder.ground(target_text)
    elapsed = time.time() - t
    print(f"  Result ({elapsed:.1f}s): found={result.found} x={result.x} y={result.y} conf={result.confidence:.2f} method={result.method}")
    results["ocr_grounding"] = PASS if result.found and result.confidence >= 0.5 else FAIL
else:
    print("  Skipping — no OCR words available")
    results["ocr_grounding"] = SKIP
print(f"\n  → {results['ocr_grounding']}")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
section("RESULTS SUMMARY")
for stage, result in results.items():
    print(f"  {result}  {stage}")

passed = sum(1 for r in results.values() if r == PASS)
total  = sum(1 for r in results.values() if r != SKIP)
print(f"\n  Score: {passed}/{total} stages passed")

if passed == total:
    print("\n  🎉 Full pipeline verified — all stages working!")
else:
    failed = [k for k, v in results.items() if v == FAIL]
    print(f"\n  Failing stages: {failed}")
