# main.py
"""Desktop GUI Agent — application entry point."""
import sys

from desktop.system import enable_dpi_awareness

# Must run before Qt and before the first screen capture so screenshots, UIA
# rectangles, and mouse input all share one physical-pixel coordinate space.
enable_dpi_awareness()

from loguru import logger
from PyQt6.QtWidgets import QApplication, QMessageBox

from agents.action import ActionExecutionAgent
from agents.grounding import UIGroundingAgent
from agents.planning import PlanningAgent
from agents.reflection import ReflectionAgent
from agents.router import RouterAgent
from core.history import TaskHistory
from core.inference import OVMSClient
from core.orchestrator import OrchestratorConfig, TaskOrchestrator
from desktop.capture import ScreenCapture
from desktop.input import DesktopController
from desktop.ocr import OCREngine
from ui.main_window import DesktopGUIAgent


def _warmup_models(client: OVMSClient, ocr: OCREngine) -> None:
    """Fire cheap dummy requests to the LLM and VLM in a background thread.
    The first real user request would otherwise pay a cold-start penalty of
    several seconds (model loading into device memory). Failures are silently
    ignored — warmup is best-effort and must not block or crash the UI.
    """
    import threading

    def _do_warmup():
        # Building the RapidOCR ONNX session takes ~2.5 s. It used to happen on
        # the main thread inside UIGroundingAgent.__init__, before the window
        # was even constructed, so the app took several seconds to appear.
        # Loading it here keeps the window instant AND keeps the first screen
        # read fast — the work happens either way, just not in front of the user.
        try:
            ocr.is_available()
        except Exception as e:
            logger.debug(f"[STARTUP] OCR warmup skipped: {e}")
        # Prewarm the installed-app catalogue (Get-StartApps, ~1-3 s but can
        # be slow on locked-down machines) so the router's app hint is ready
        # before the first decompose instead of timing out inside it.
        try:
            from desktop.system import installed_apps
            apps = installed_apps()
            logger.info(f"[STARTUP] Installed-app catalogue: {len(apps)} apps")
        except Exception as e:
            logger.debug(f"[STARTUP] App catalogue skipped: {e}")
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
    history = TaskHistory()

    _warmup_models(client, ocr)

    return TaskOrchestrator(
        router=RouterAgent(client),
        planner=PlanningAgent(client),
        grounder=UIGroundingAgent(client, capturer, ocr=ocr),
        actor=ActionExecutionAgent(controller),
        reflector=ReflectionAgent(client, capturer, ocr=ocr),
        capturer=capturer,
        history=history,
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
