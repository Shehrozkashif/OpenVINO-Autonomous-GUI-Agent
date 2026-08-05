"""Decision layer — the loop, and every rule it follows.

Everything here answers one of two questions: *what do we do next?* and
*did it actually work?* It is the only layer allowed to make a judgment call.

    orchestrator.py  the See -> Plan -> Act -> Verify loop
    runstate.py      every budget and limit, plus per-subtask state
    groundtruth.py   checks the OS can prove (file on disk, app launched, ...)
    subtasks.py      what a subtask's own words ask for
    apps.py          app name -> executable name and on-screen signals
    anchor.py        which window this task owns
    firewall.py      destructive typed text, blocked by regex (never a model)
    inference.py     the InferenceClient protocol + the OVMS client
    history.py       a record of finished tasks, for the UI to display
    types.py         SubTask and ActionStep

Rule: core/ decides; it never talks to Windows directly. It asks desktop/ for
facts and agents/ for model opinions, then makes the call itself. That is what
lets the whole unit suite run on Linux with no GPU and no model server.
"""
