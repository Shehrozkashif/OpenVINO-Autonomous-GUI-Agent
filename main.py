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
from core.pipeline.ollama_client import OllamaClient
from memory.task.task_memory import TaskMemory
from tools.desktop_control.controller import DesktopController
from ui.main_window import DesktopGUIAgent


def build_orchestrator() -> TaskOrchestrator:
    client = OllamaClient()

    health = client.check_health()
    for name, status in health.items():
        if status != "OK":
            logger.warning(f"[STARTUP] {name}: {status}")

    capturer = ScreenCapture()
    controller = DesktopController()

    return TaskOrchestrator(
        router=RouterAgent(client),
        planner=PlanningAgent(client),
        grounder=UIGroundingAgent(client, capturer),
        actor=ActionExecutionAgent(controller),
        reflector=ReflectionAgent(client, capturer),
        capturer=capturer,
        task_memory=TaskMemory(),
        config=OrchestratorConfig(max_retries_per_step=3, reflection_wait_s=1.0),
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--auto-run", action="store_true")
    args, unknown = parser.parse_known_args()

    app = QApplication([sys.argv[0]] + unknown)
    app.setApplicationName("Desktop GUI Agent")
    app.setOrganizationName("OpenVINO-GSoC")

    try:
        orchestrator = build_orchestrator()
    except Exception as e:
        QMessageBox.warning(None, "Startup Warning",
                            f"Could not connect to Ollama: {e}\n\nRun: ollama serve")
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
