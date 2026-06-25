# main.py
"""Desktop GUI Agent — application entry point."""
import sys
from typing import Optional

from loguru import logger
from PyQt6.QtWidgets import QApplication, QMessageBox

from agents.action.action_agent import ActionExecutionAgent
from agents.grounding.grounding_agent import OCREngine, UIGroundingAgent
from agents.planning.planning_agent import PlanningAgent
from agents.reflection.reflection_agent import ReflectionAgent
from agents.router.router_agent import RouterAgent
from core.capture.screenshot import ScreenCapture
from core.orchestrator import OrchestratorConfig, TaskOrchestrator
from core.pipeline.ovms_client import OVMSClient
from memory.task.task_memory import TaskMemory
from tools.desktop_control.controller import DesktopController
from ui.main_window import DesktopGUIAgent


def _warmup_models(client: OVMSClient, task_memory: TaskMemory | None = None) -> None:
    """
    Fire cheap dummy requests to the LLM and VLM, and pre-load the
    sentence-transformer embedder, in a background thread. The first real user
    request would otherwise pay a cold-start penalty of several seconds (model
    loading into device memory) or, for the embedder, a one-time ~80s
    download/load mid-task. Failures are silently ignored — warmup is best-effort
    and must not block or crash the UI.
    """
    import threading

    def _do_warmup():
        try:
            client.query_llm(
                [{"role": "user", "content": "ping"}],
                max_tokens=1, temperature=0.0,
            )
            logger.info("[STARTUP] LLM warmup done")
        except Exception as e:
            logger.debug(f"[STARTUP] LLM warmup skipped: {e}")
        # OVMS keeps both models resident, so warming the VLM does not evict the
        # LLM — warm it too so the first grounding call is fast.
        try:
            import base64
            import io

            from PIL import Image
            tiny = Image.new("RGB", (64, 64), color=(128, 128, 128))
            buf = io.BytesIO()
            tiny.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            client.query_vlm(
                prompt="What is this?", image_base64=b64,
                max_tokens=1, temperature=0.0,
            )
            logger.info("[STARTUP] VLM warmup done")
        except Exception as e:
            logger.debug(f"[STARTUP] VLM warmup skipped: {e}")
        if task_memory is not None:
            try:
                _ = task_memory.embedder  # triggers SentenceTransformer download/load
                logger.info("[STARTUP] Embedder warmup done")
            except Exception as e:
                logger.debug(f"[STARTUP] Embedder warmup skipped: {e}")

    threading.Thread(target=_do_warmup, daemon=True).start()


def build_orchestrator() -> TaskOrchestrator:
    client = OVMSClient()

    health = client.check_health()
    for name, status in health.items():
        if status != "OK":
            logger.warning(f"[STARTUP] {name}: {status}")

    capturer = ScreenCapture()
    controller = DesktopController()
    ocr = OCREngine()
    task_memory = TaskMemory()

    _warmup_models(client, task_memory)

    return TaskOrchestrator(
        router=RouterAgent(client),
        planner=PlanningAgent(client),
        grounder=UIGroundingAgent(client, capturer, ocr=ocr),
        actor=ActionExecutionAgent(controller),
        reflector=ReflectionAgent(client, capturer, ocr=ocr),
        capturer=capturer,
        task_memory=task_memory,
        config=OrchestratorConfig(max_retries_per_step=3, reflection_wait_s=1.0),
        ocr=ocr,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Desktop GUI Agent")
    parser.add_argument("--prompt", type=str, default=None,
                        help="pre-fill the instruction input with this text")
    parser.add_argument("--auto-run", action="store_true",
                        help="run the pre-filled --prompt immediately on launch")
    args, unknown = parser.parse_known_args()

    app = QApplication([sys.argv[0]] + unknown)
    app.setApplicationName("Desktop GUI Agent")
    app.setOrganizationName("OpenVINO-GSoC")

    try:
        orchestrator = build_orchestrator()
    except Exception as e:
        QMessageBox.warning(None, "Startup Warning",
                            f"Could not connect to OpenVINO Model Server: {e}\n\n"
                            "Start it with:  python start.py")
        orchestrator = None

    window = DesktopGUIAgent(orchestrator=orchestrator)

    if args.prompt:
        window.instruction_input.setPlainText(args.prompt)

    window.show()

    if args.auto_run and args.prompt:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(1000, window._run_task)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
