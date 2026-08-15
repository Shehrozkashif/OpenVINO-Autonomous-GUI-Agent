"""World layer — what is really on the machine, and how to change it.

    uia.py          the Windows accessibility tree: find, invoke, set_value
    input.py        raw SendInput mouse/keyboard, and the kill switch
    capture.py      screenshots, frame hashing, masking the agent's own window
    ocr.py          RapidOCR, plus fuzzy label matching
    snapshot.py     an OCR pass that knows which window owns each word
    system.py       DPI, foreground window, processes, installed apps, GPUs
    clipboard.py    paste-typing (long or sensitive text)
    credentials.py  secrets from the OS keyring

Rule: this layer reports facts and performs effects. It never decides what a
fact means. system.count_process_windows("ms-teams.exe") returns 3; whether 3
counts as a successful launch is core/groundtruth.py's call. Keep it that way —
the policy above it is only testable because none of it lives here.
"""
