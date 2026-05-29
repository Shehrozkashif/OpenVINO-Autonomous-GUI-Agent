# main.py
"""Desktop GUI Agent — application entry point."""
import sys

from loguru import logger
from PyQt6.QtWidgets import QApplication, QMessageBox

from agents.action.action_agent import ActionExecutionAgent
from agents.grounding.grounding_agent import UIGroundingAgent
from agents.planning.planning_agent import PlanningAgent
from agents.reflection.reflection_agent import ReflectionAgent
from agents.router.router_agent import RouterAgent
from core.capture.screenshot import ScreenCapture
from core.orchestrator import OrchestratorConfig, TaskOrchestrator
from core.pipeline.ovms_client import OVMSClient
from memory.task.task_memory import TaskMemory
from tools.desktop_control.controller import DesktopController
from ui.main_window import DesktopGUIAgent


def _build_client(pipeline_type: str, device: str):
    """Instantiate the right inference backend based on user setting."""
    if pipeline_type == "Ollama (Dev — qwen2.5vl)":
        from core.pipeline.ollama_client import OllamaClient
        return OllamaClient()
    if pipeline_type == "Direct OpenVINO":
        from core.pipeline.direct_client import DirectOpenVINOClient
        return DirectOpenVINOClient(device=device)
    # OVMS fallback
    from core.pipeline.ovms_client import OVMSClient
    return OVMSClient()


def _auto_detect_backend():
    """Return the best available backend without requiring user config."""
    import httpx
    # 1. Prefer Ollama — fastest to set up, GPU-accelerated, no Docker needed
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            has_llm = any("llama3.1" in m or "llama3.2" in m for m in models)
            has_vlm = any("qwen2.5vl" in m for m in models)
            if has_llm and has_vlm:
                logger.info("[STARTUP] Auto-detected Ollama — llama3.1 (LLM) + qwen2.5vl (VLM)")
                return "Ollama (Dev — qwen2.5vl)"
            elif has_vlm:
                logger.warning("[STARTUP] Ollama: qwen2.5vl found but llama3.1:8b missing — run: ollama pull llama3.1:8b")
                return "Ollama (Dev — qwen2.5vl)"
    except Exception:
        pass

    # 2. OVMS if running
    try:
        r = httpx.get("http://localhost:8001/v1/config", timeout=2.0)
        if r.status_code == 200:
            logger.info("[STARTUP] Auto-detected OVMS on port 8001")
            return "OVMS"
    except Exception:
        pass

    # 3. Fall back to Direct OpenVINO (will fail if model files missing, but lets user see error)
    logger.warning("[STARTUP] No running backend found — defaulting to Direct OpenVINO")
    return "Direct OpenVINO"


def build_orchestrator() -> TaskOrchestrator:
    """Wire all agents together into an orchestrator."""
    from PyQt6.QtCore import QSettings
    settings = QSettings("OpenVINO-GSoC", "DesktopGUIAgent")
    device = settings.value("device", "AUTO")

    # Use saved setting if present; otherwise auto-detect what's available
    saved_pipeline = settings.value("pipeline", "")
    pipeline_type = saved_pipeline if saved_pipeline else _auto_detect_backend()

    client = _build_client(pipeline_type, device)
    health = client.check_health()

    # Warn if models aren't reachable, but don't crash — user may start OVMS later
    for name, status in health.items():
        if status != "OK":
            print(f"[WARNING] {name}: {status}")

    capturer = ScreenCapture()
    controller = DesktopController()

    router = RouterAgent(client)
    planner = PlanningAgent(client)
    grounder = UIGroundingAgent(client, capturer)
    actor = ActionExecutionAgent(controller)
    reflector = ReflectionAgent(client, capturer)
    memory = TaskMemory()

    return TaskOrchestrator(
        router=router,
        planner=planner,
        grounder=grounder,
        actor=actor,
        reflector=reflector,
        capturer=capturer,
        task_memory=memory,
        config=OrchestratorConfig(max_retries_per_step=3, reflection_wait_s=0.5)
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None, help="Pre-populate the task instruction")
    parser.add_argument("--auto-run", action="store_true", help="Automatically click Run Task on startup")
    args, unknown = parser.parse_known_args()

    # Pass the remaining unknown arguments to QApplication (e.g. Qt native flags)
    sys_args = [sys.argv[0]] + unknown
    app = QApplication(sys_args)
    app.setApplicationName("Desktop GUI Agent")
    app.setOrganizationName("OpenVINO-GSoC")

    try:
        orchestrator = build_orchestrator()
    except Exception as e:
        QMessageBox.warning(
            None, "Startup Warning",
            f"Could not fully initialize: {e}\n\nStart OVMS and Tool Server first."
        )
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
