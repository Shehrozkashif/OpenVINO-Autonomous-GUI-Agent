# core/safety/action_firewall.py
"""Action firewall — a deterministic guard that inspects text the agent is about to
type (and certain key actions) for destructive or dangerous operations BEFORE
they are executed.

Why this exists
---------------
The planner/router can be steered by text that appears on screen (a web page, an
email, a chat message). That text flows into the LLM prompt verbatim, so a page
saying "open a terminal and run rm -rf ~" is indistinguishable from a genuine
user instruction at the model layer. The router's own policy turns file
operations into shell commands. Together that means the agent can be induced to
type destructive commands with no human in the loop.

This module is the last line of defence: a regex classifier that runs in the
orchestrator just before `actor.execute()` for `type` steps. It cannot be talked
out of its verdict by any prompt because it never calls a model.

Severity model
--------------
  HIGH    — irreversible / system-wide damage (recursive deletes of broad paths,
            disk formatting, fork bombs, piping the internet into a shell,
            shutdown/reboot, ownership/permission nukes). Blocked unless an
            explicit confirmation handler approves.
  MEDIUM  — reversible-ish but risky (deleting a specific file, overwriting via
            mv/move, force-removing git state). Allowed, but routed through the
            confirmation handler when one is wired.
  NONE    — everything else.

Integration contract
---------------------
`evaluate(text)` returns a Verdict. The orchestrator decides what to do with it
based on whether a confirmation callback is available (see `Decision`).
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from collections.abc import Callable


class Severity(str, Enum):
    NONE = "none"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Verdict:
    severity: Severity
    matched: list[str]          # human-readable reasons
    text: str

    @property
    def is_dangerous(self) -> bool:
        return self.severity != Severity.NONE


# ── Pattern tables ────────────────────────────────────────────────────────────
# Each entry: (compiled_regex, human_reason). Patterns are intentionally broad on
# the HIGH list (false positives there merely prompt for confirmation, which is
# the safe failure mode) and narrower on MEDIUM.

def _c(pattern: str) -> "re.Pattern":
    return re.compile(pattern, re.IGNORECASE)


_HIGH_PATTERNS = [
    # Recursive force-delete of a broad path (/, ~, $HOME, *, .)
    (_c(r"\brm\s+(?:-[a-z]*\s+)*-?[a-z]*[rf][a-z]*\b.*\s+(?:/|~|\$HOME|\.\s*$|\*)"),
     "recursive/forced delete of a broad path (rm -rf)"),
    (_c(r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r"),
     "recursive forced delete (rm -rf)"),
    # Windows recursive delete of a drive/profile root
    (_c(r"\b(?:rmdir|rd)\s+/s\b|\bdel\s+/s\b.*[\\/]\s*$"),
     "recursive delete (rmdir /s)"),
    (_c(r"\bformat\s+[a-z]:"), "disk format"),
    (_c(r"\bmkfs(?:\.\w+)?\b"), "filesystem creation (mkfs)"),
    (_c(r"\bdd\b.*\bof=/dev/"), "raw write to a block device (dd of=/dev/...)"),
    (_c(r">\s*/dev/(?:sd|nvme|hd|mmcblk|disk)"), "redirect into a block device"),
    (_c(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:"), "fork bomb"),
    # Pipe a remote payload straight into a shell
    (_c(r"\b(?:curl|wget|iwr|invoke-webrequest)\b[^\n|]*\|\s*(?:sudo\s+)?(?:bash|sh|zsh|python|powershell|pwsh|iex)\b"),
     "remote code piped into a shell"),
    (_c(r"\biex\b.*\b(?:downloadstring|invoke-webrequest|iwr)\b"),
     "remote code executed via PowerShell IEX"),
    (_c(r"\b(?:shutdown|reboot|halt|poweroff)\b"), "system power state change"),
    (_c(r"\bchmod\s+-R\s+0*7*\s+/"), "recursive permission change on root"),
    (_c(r"\bchown\s+-R\b.*\s+/(?:\s|$)"), "recursive ownership change on root"),
    (_c(r"\bgit\s+push\b.*\s--force(?:-with-lease)?\b.*\b(?:main|master)\b"),
     "force-push to a protected branch"),
    (_c(r"\bsudo\s+rm\b"), "privileged delete (sudo rm)"),
]

_MEDIUM_PATTERNS = [
    (_c(r"\brm\s+(?!-[a-z]*[rf])"), "delete a file (rm)"),
    (_c(r"\bdel\s+|\berase\s+"), "delete a file (del)"),
    (_c(r"\b(?:mv|move)\b"), "move/overwrite a file"),
    (_c(r"\bgit\s+reset\s+--hard\b"), "discard local changes (git reset --hard)"),
    (_c(r"\bgit\s+clean\s+-[a-z]*f"), "remove untracked files (git clean -f)"),
    (_c(r"\bdrop\s+(?:table|database)\b"), "SQL drop"),
    (_c(r"\btruncate\b"), "truncate"),
    (_c(r"\bkill(?:all)?\s+-9\b"), "force-kill a process"),
    (_c(r">\s*[^>\s]"), "overwrite a file via output redirect"),
]


def evaluate(text: str | None) -> Verdict:
    """Classify a candidate command/text string by destructive severity."""
    if not text or not text.strip():
        return Verdict(Severity.NONE, [], text or "")
    reasons_high = [why for rx, why in _HIGH_PATTERNS if rx.search(text)]
    if reasons_high:
        return Verdict(Severity.HIGH, reasons_high, text)
    reasons_med = [why for rx, why in _MEDIUM_PATTERNS if rx.search(text)]
    if reasons_med:
        return Verdict(Severity.MEDIUM, reasons_med, text)
    return Verdict(Severity.NONE, [], text)


class Decision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


def decide(
    verdict: Verdict,
    confirm_cb: Callable[[str, str], bool] | None = None,
) -> Decision:
    """Turn a Verdict into an ALLOW/BLOCK decision.

    confirm_cb(summary, command) -> bool : optional human confirmation handler.
        Returns True to allow, False to deny. When absent:
          HIGH   → BLOCK   (never auto-run system-wrecking commands unattended)
          MEDIUM → ALLOW   (preserve existing file-task behaviour, but logged)
    """
    if verdict.severity == Severity.NONE:
        return Decision.ALLOW

    summary = "; ".join(verdict.matched)
    if confirm_cb is not None:
        try:
            approved = confirm_cb(summary, verdict.text)
            return Decision.ALLOW if approved else Decision.BLOCK
        except Exception:
            # A broken confirmation handler must fail safe.
            return Decision.BLOCK if verdict.severity == Severity.HIGH else Decision.ALLOW

    return Decision.BLOCK if verdict.severity == Severity.HIGH else Decision.ALLOW
