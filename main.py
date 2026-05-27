# main.py
"""Desktop GUI Agent — application entry point."""
import sys

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


def build_orchestrator() -> TaskOrchestrator:
    """Wire all agents together into an orchestrator."""
    ovms = OVMSClient()
    health = ovms.check_health()

    # Warn if models aren't reachable, but don't crash — user may start OVMS later
    for name, status in health.items():
        if status != "OK":
            print(f"[WARNING] {name}: {status}")

    capturer = ScreenCapture()
    controller = DesktopController()

    router = RouterAgent(ovms)
    planner = PlanningAgent(ovms)
    grounder = UIGroundingAgent(ovms, capturer)
    actor = ActionExecutionAgent(controller)
    reflector = ReflectionAgent(ovms, capturer)
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
    app = QApplication(sys.argv)
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
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
