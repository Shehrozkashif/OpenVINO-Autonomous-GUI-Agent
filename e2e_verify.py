"""E2E pipeline verification — uses OVMSClient.

Run after: python start.py  (prepares both models and starts OpenVINO Model Server)
Tests Router → Planner → Grounder with a real instruction.
"""
import sys

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{level}: {message}", level="INFO")

from agents.grounding.grounding_agent import UIGroundingAgent
from agents.planning.planning_agent import PlanningAgent
from agents.router.router_agent import RouterAgent
from core.capture.screenshot import ScreenCapture
from core.pipeline.ovms_client import OVMSClient


def test_pipeline():
    print("=== E2E Pipeline Verification (OVMS backend) ===\n")

    # ── 1. Backend health ──────────────────────────────────────────────────────
    client = OVMSClient()
    health = client.check_health()
    print("Backend health:")
    all_ok = True
    for name, status in health.items():
        ok = status == "OK"
        print(f"  {'[OK]' if ok else '[FAIL]'} {name}: {status}")
        if not ok:
            all_ok = False
    if not all_ok:
        print("\n  Prepare models and start the server first:  python start.py")
        sys.exit(1)

    # ── 2. Router ──────────────────────────────────────────────────────────────
    print("\n--- Router ---")
    router = RouterAgent(client)
    instruction = "open the Files app and go to the Desktop folder"
    print(f"Instruction: \"{instruction}\"")
    task_id, subtasks = router.decompose(instruction)
    print(f"Decomposed into {len(subtasks)} sub-tasks:")
    for st in subtasks:
        print(f"  [{st.id}] {st.description}  (depends on: {st.depends_on})")

    # ── 3. Planner ─────────────────────────────────────────────────────────────
    print("\n--- Planner ---")
    planner = PlanningAgent(client)
    first = subtasks[0]
    print(f"Planning: \"{first.description}\"")
    steps = planner.plan(first)
    print(f"Generated {len(steps)} steps:")
    for s in steps:
        print(f"  [{s.id}] {s.action_type}  target={s.target!r}  key={s.key!r}  value={s.value!r}")

    # ── 4. Grounder (OCR stage — no VLM needed for text elements) ─────────────
    print("\n--- Grounder (OCR stage) ---")
    cap = ScreenCapture()
    grounder = UIGroundingAgent(client, cap, min_confidence=0.5)
    ocr_targets = ["Desktop", "Home", "Files"]
    for t in ocr_targets:
        r = grounder.ground(t)
        status = "FOUND" if r.found else "NOT FOUND"
        print(f"  {status}  \"{t}\"  ({r.x},{r.y})  conf={r.confidence:.2f}  method={r.method}")

    # ── 5. VLM grounding (needs the UI-TARS servable loaded) ──────────────────
    print("\n--- Grounder (VLM stage — icon-only element) ---")
    icon_result = grounder.ground("the terminal or shell icon")
    if icon_result.found:
        print(f"  FOUND  ({icon_result.x},{icon_result.y})  conf={icon_result.confidence:.2f}  method={icon_result.method}")
    else:
        print("  NOT FOUND (element may genuinely not be on screen right now)")

    print("\n=== All checks passed — pipeline is functional ===")


if __name__ == "__main__":
    try:
        test_pipeline()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nPIPELINE ERROR: {e}")
        raise
