#!/usr/bin/env python
"""Standalone probe: can this AI PC serve a qwen3_vl VLM, and how well does it ground?

Answers three questions, in order, and stops at the first hard "no":

  1. Does this machine's OVMS understand the `qwen3_vl` architecture at all?
     (MAI-UI-8B is a Qwen3-VL fine-tune. UI-TARS-1.5 is qwen2.5-VL — a different
     shape. OVMS 2026.0 rejected qwen3_vl with "Unsupported VLM model type";
     later builds accept it. This is the blocker worth 10 minutes before a
     17.6 GB download.)

  2. Which coordinate convention does it answer in? The project's coords.py
     assumes qwen2.5-VL's smart-resize factor of 28 (patch 14 x merge 2).
     Qwen3-VL uses patch_size 16 x merge_size 2 = 32. Rather than trust either
     number, this script tries EVERY plausible reading of the model's numbers
     and scores each one against Windows UIA ground truth. The winner is a
     measurement, not an assumption.

  3. Is it actually better at pointing than the UI-TARS you already run?
     Same screenshot, same targets, head to head.

Nothing here touches your working setup: separate model directory, separate
port (8001), separate OVMS process. Your models/ and config.json are untouched.

    python tools/test_qwen3vl.py                    # step 1: prebuilt Qwen3-VL
    python tools/test_qwen3vl.py --baseline         # ...and compare vs UI-TARS
    python tools/test_qwen3vl.py --source Tongyi-MAI/MAI-UI-8B \
        --name mai-ui-8b-int4-ov --weight-format int4     # step 2: the real thing

Run it, then click the app you want to test (Teams is the interesting one) while
the countdown runs.
"""
import argparse
import ast
import base64
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ── Settings ──────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "qwen3vl_probe")
REPO = os.path.join(WORK, "models")
CONFIG_JSON = os.path.join(REPO, "config.json")

PORT = 8001                       # NOT 8000 — your live agent keeps that
BASE = f"http://localhost:{PORT}"
CHAT = "/v3/chat/completions"

# Prebuilt OpenVINO IR published by Intel. No conversion, no optimum-cli, and
# it is the SAME architecture as MAI-UI-8B — so if this serves, MAI-UI will too.
DEFAULT_SOURCE = "OpenVINO/Qwen3-VL-8B-Instruct-int4-ov"
DEFAULT_NAME = "qwen3-vl-8b-int4-ov"

# What your live agent runs, for the head-to-head.
BASELINE_URL = "http://localhost:8000"
BASELINE_MODEL = "ui-tars-1.5-7b-int8-ov"

# Size the screenshot is downscaled to before sending. Matches OCR_THUMB in
# desktop/capture.py, so the numbers here transfer to the real pipeline.
SEND_W, SEND_H = 1280, 720

EXPORT_TOOL_URL = (
    "https://raw.githubusercontent.com/openvinotoolkit/model_server/"
    "main/demos/common/export_models/export_model.py"
)


# ── Console ───────────────────────────────────────────────────────────────────

def _c(code: str, s: str) -> str:
    return s if os.environ.get("NO_COLOR") else f"\033[{code}m{s}\033[0m"


def green(s):  return _c("92", s)
def yellow(s): return _c("93", s)
def red(s):    return _c("91", s)
def bold(s):   return _c("1", s)


def step(n: int, title: str) -> None:
    print()
    print(bold(f"── {n}. {title} " + "─" * max(0, 62 - len(title))))


def die(msg: str, hint: str = "") -> None:
    print(red(f"  [FAIL] {msg}"))
    if hint:
        for line in hint.strip().splitlines():
            print(yellow(f"         {line}"))
    sys.exit(1)


# ── 1. Preflight ──────────────────────────────────────────────────────────────

def find_ovms() -> str:
    """Locate ovms.exe the same way start.py does."""
    exe = "ovms.exe"
    for env in ("OVMS_PATH", "OVMS_DIR"):
        base = os.environ.get(env)
        if not base:
            continue
        for cand in (base, os.path.join(base, exe), os.path.join(base, "ovms", exe)):
            if os.path.isfile(cand):
                return cand
    return shutil.which("ovms") or shutil.which(exe) or ""


def ovms_version(binary: str) -> str:
    """Best-effort version string.

    ovms.exe is not consistent here: some builds print to stdout, some to
    stderr, some answer `--version` with nothing at all and only talk under
    `--help`. Never let this stop the probe — the version is a hint printed for
    the human, and step 3 is the real test of whether qwen3_vl loads.
    """
    for flag in ("--version", "--help"):
        try:
            out = subprocess.run([binary, flag], capture_output=True,
                                 text=True, timeout=30)
        except Exception as e:
            return f"(could not read version: {e})"
        blob = f"{out.stdout or ''}\n{out.stderr or ''}"
        for line in blob.splitlines():
            line = line.strip()
            if line and re.search(r"\d{4}\.\d", line):
                return line
        if flag == "--version":
            for line in blob.splitlines():
                if line.strip():
                    return line.strip()
    return "(version not reported by this build)"


def preflight() -> str:
    step(1, "Preflight")
    if platform.system() != "Windows":
        die(f"Windows only (detected {platform.system()})")
    print(green("  [OK] Windows"))

    missing = []
    for mod, pkg in (("PIL", "pillow"), ("uiautomation", "uiautomation")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        die(f"Missing packages: {', '.join(missing)}",
            f"pip install {' '.join(missing)}")
    print(green("  [OK] pillow + uiautomation"))

    binary = find_ovms()
    if not binary:
        die("ovms.exe not found",
            "Set OVMS_DIR to the folder holding ovms.exe, or add it to PATH.\n"
            "https://docs.openvino.ai/latest/model-server/ovms_docs_deploying_server.html")
    print(green(f"  [OK] {binary}"))
    print(f"       {ovms_version(binary)}")
    print(yellow("       Version matters: qwen3_vl was rejected by OVMS 2026.0."))
    return binary


# ── 2. Prepare the model ──────────────────────────────────────────────────────

def already_exported(name: str) -> bool:
    if not os.path.isfile(CONFIG_JSON):
        return False
    try:
        with open(CONFIG_JSON) as f:
            cfg = json.load(f)
    except Exception:
        return False
    names = [e.get("name") for e in cfg.get("mediapipe_config_list", [])]
    names += [e.get("config", {}).get("name") for e in cfg.get("model_config_list", [])]
    return name in names


def export_tool() -> str:
    """Reuse the project's copy of export_model.py, else fetch one."""
    local = os.path.join(HERE, "ovms", "export_model.py")
    if os.path.isfile(local):
        print(green(f"  [OK] Using {local}"))
        return local
    os.makedirs(WORK, exist_ok=True)
    dest = os.path.join(WORK, "export_model.py")
    if os.path.isfile(dest):
        return dest
    print(yellow("  [..] Fetching export_model.py (one-time)..."))
    try:
        urllib.request.urlretrieve(EXPORT_TOOL_URL, dest)
    except Exception as e:
        die(f"Could not download export_model.py: {e}",
            f"Download it manually to {dest} from {EXPORT_TOOL_URL}")
    print(green("  [OK] export_model.py"))
    return dest


# Module name -> pip name, where the two differ.
_PIP_NAME = {
    "optimum": "optimum-intel[openvino]",
    "openvino_tokenizers": "openvino-tokenizers",
    "openvino_genai": "openvino-genai",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "cv2": "opencv-python",
}

# Modules that requirements-export.txt already pulls in, directly or as
# dependencies of optimum-intel[openvino]. If everything missing is on this
# list, one documented command fixes it — no need to name packages one at a
# time and have the user rediscover them on the next run.
_EXPORT_TOOLCHAIN = {
    "huggingface_hub", "jinja2", "nncf", "numpy", "openvino",
    "openvino_genai", "openvino_tokenizers", "optimum", "torch",
    "transformers",
}


def check_export_deps(tool: str) -> None:
    """Fail before the download, not after.

    export_model.py is Intel's script, not ours, and it imports things this
    venv may not have (jinja2 is the usual one). Left alone it raises
    ModuleNotFoundError partway through, which reads like the model is
    unsupported when the real problem is one missing pip package. So read its
    top-level imports and check them up front.
    """
    try:
        with open(tool, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:
        return  # Not worth failing the run over; the real import will speak.

    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])

    missing = []
    for mod in sorted(mods):
        if mod in sys.builtin_module_names:
            continue
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except Exception:
            missing.append(mod)

    if not missing:
        print(green("  [OK] export_model.py dependencies present"))
        return

    names = ", ".join(missing)
    reqs = os.path.join(os.path.dirname(HERE), "requirements-export.txt")
    if " " in reqs:
        reqs = f'"{reqs}"'
    if os.path.isfile(reqs.strip('"')) and set(missing) <= _EXPORT_TOOLCHAIN:
        # requirements.txt deliberately omits the ML stack — the agent talks to
        # OVMS over HTTP and imports no framework. Conversion is the one job
        # that needs it, and the repo already pins that set.
        die(f"The model-conversion toolchain is not installed: {names}",
            "Install CPU-only torch first — otherwise pip pulls ~3.4 GB of\n"
            "CUDA wheels this Intel GPU will never use:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            f"  pip install -r {reqs}\n"
            "Then run this script again — nothing was downloaded yet.")

    # Quote extras like optimum-intel[openvino] — bare brackets are glob syntax
    # in some shells and get eaten before pip sees them.
    pkgs = [_PIP_NAME.get(m, m) for m in missing]
    args = " ".join(f'"{p}"' if "[" in p else p for p in pkgs)
    die(f"export_model.py needs packages this venv lacks: {names}",
        f"pip install {args}\n"
        "Then run this script again — nothing was downloaded yet.")


def prepare(source: str, name: str, weight_format: str, device: str, cache_gb: int) -> None:
    step(2, f"Prepare {name}")
    os.makedirs(REPO, exist_ok=True)
    if already_exported(name):
        print(green(f"  [OK] {name} already in {REPO} — skipping download"))
        return

    tool = export_tool()
    check_export_deps(tool)
    print(yellow(f"  [..] Pulling {source} ({weight_format}, {device})"))
    print(yellow("       A prebuilt IR is a plain download; a raw checkpoint"))
    print(yellow("       (e.g. MAI-UI-8B, 17.6 GB) converts on CPU and is slow."))
    cmd = [
        sys.executable, tool, "text_generation",
        "--source_model", source,
        "--model_name", name,
        "--weight-format", weight_format,
        "--config_file_path", CONFIG_JSON,
        "--model_repository_path", REPO,
        "--target_device", device,
        "--cache_size", str(cache_gb),
    ]
    ret = subprocess.run(cmd, cwd=WORK).returncode
    if ret != 0 or not already_exported(name):
        die(f"Could not prepare {name}",
            "If the error mentions 'qwen3_vl' or an unknown model type, the\n"
            "conversion toolchain is too old:\n"
            "  pip install -U 'optimum-intel[openvino]' nncf openvino\n"
            "If it ran out of memory, a raw checkpoint conversion needs ~16 GB RAM.")
    print(green(f"  [OK] {name} ready"))


# ── 3. Serve ──────────────────────────────────────────────────────────────────

def http_get(url: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return 0, ""


def model_ready(name: str, base: str = BASE) -> bool:
    status, body = http_get(f"{base}/v1/config", timeout=3.0)
    if status != 200:
        return False
    try:
        cfg = json.loads(body)
    except Exception:
        return False
    entry = cfg.get(name)
    if not entry:
        return False
    states = [v.get("state") for v in entry.get("model_version_status", [])]
    return "AVAILABLE" in states


def serve(binary: str, name: str) -> subprocess.Popen | None:
    step(3, f"Serve on port {PORT}")

    if model_ready(name):
        print(green(f"  [OK] Already serving {name} on {PORT}"))
        return None

    if http_get(f"{BASELINE_URL}/v1/config", timeout=2.0)[0] == 200:
        print(yellow("  [WARN] Your live agent is running on 8000."))
        print(yellow("         Two VLMs on one GPU may not fit in 27 GB."))
        print(yellow("         If the load below fails or hangs, stop it first"))
        print(yellow("         (taskkill /F /IM ovms.exe) and re-run."))

    log_path = os.path.join(WORK, "ovms_probe.log")
    launcher = os.path.join(WORK, "_run_probe_ovms.bat")
    setupvars = os.path.join(os.path.dirname(binary), "setupvars.bat")
    inner = subprocess.list2cmdline([
        binary, "--config_path", CONFIG_JSON, "--rest_port", str(PORT),
    ])
    lines = ["@echo off"]
    if os.path.isfile(setupvars):
        lines.append(f'call "{setupvars}"')
    else:
        print(yellow(f"  [WARN] No setupvars.bat next to {binary} — DLLs may not resolve"))
    lines.append(inner)
    with open(launcher, "w") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(yellow(f"  [..] Starting OVMS   log: {log_path}"))
    log_file = open(log_path, "w")
    proc = subprocess.Popen(["cmd", "/c", launcher], cwd=WORK,
                            stdout=log_file, stderr=log_file)

    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(5)
        if model_ready(name):
            print(green(f"\n  [OK] {name} is AVAILABLE"))
            return proc
        if proc.poll() is not None:
            print()
            _report_load_failure(log_path)
        left = int(deadline - time.time())
        sys.stdout.write(f"\r  Loading model into the GPU... {left}s left ")
        sys.stdout.flush()

    print()
    _report_load_failure(log_path)
    return proc


def _report_load_failure(log_path: str) -> None:
    """Read the OVMS log and say plainly whether the architecture was rejected."""
    text = ""
    try:
        with open(log_path, errors="replace") as f:
            text = f.read()
    except Exception:
        pass
    tail = "\n".join(text.strip().splitlines()[-25:])

    if re.search(r"[Uu]nsupported.*(qwen3_vl|VLM model type)", text):
        print(red("  [FAIL] This OVMS build does not support qwen3_vl."))
        print()
        print(bold("  This is the answer to the main question: NO, not on this version."))
        print(yellow("  MAI-UI-8B cannot run here until OVMS is upgraded."))
        print(yellow("  Upgrade OVMS and re-run this script. Do NOT download the"))
        print(yellow("  17.6 GB MAI-UI checkpoint until this step passes."))
    elif re.search(r"out of|OUT_OF|memory|allocat", text, re.I):
        print(red("  [FAIL] Model did not load — looks like GPU memory."))
        print(yellow("  Stop your live agent (taskkill /F /IM ovms.exe) and re-run,"))
        print(yellow("  or pass --device CPU to check correctness without the GPU."))
    else:
        print(red("  [FAIL] Model did not become available in time."))
    print()
    print(bold("  Last lines of the OVMS log:"))
    for line in tail.splitlines():
        print(f"    {line}")
    sys.exit(1)


# ── 4. Ground truth from Windows ──────────────────────────────────────────────

def pick_targets(count: int, countdown: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Ask Windows where real controls are. This is the yardstick.

    Uses the accessibility tree — the same Stage 0 the agent trusts at
    confidence 1.0 — so "correct" here means what the agent already means by it.
    """
    step(4, "Ground truth from the accessibility tree")
    import uiautomation as auto

    if countdown > 0:
        print(bold("  Click the window you want to test — Teams is the interesting one."))
        for i in range(countdown, 0, -1):
            sys.stdout.write(f"\r  Capturing the foreground window in {i}s... ")
            sys.stdout.flush()
            time.sleep(1)
        print()

    wanted = {
        "ButtonControl", "MenuItemControl", "TabItemControl", "ListItemControl",
        "CheckBoxControl", "RadioButtonControl", "HyperlinkControl",
        "ComboBoxControl", "EditControl", "TreeItemControl",
    }
    found: list[tuple[str, tuple[int, int, int, int]]] = []
    seen: set[str] = set()

    try:
        root = auto.GetForegroundControl()
        print(f"  Window: {bold(root.Name or '(no title)')}")
    except Exception as e:
        die(f"Could not read the foreground window: {e}")

    def walk(ctrl, depth: int) -> None:
        if depth > 14 or len(found) >= count * 4:
            return
        try:
            children = ctrl.GetChildren()
        except Exception:
            return
        for child in children:
            try:
                name = (child.Name or "").strip()
                ctype = child.ControlTypeName
                r = child.BoundingRectangle
                rect = (r.left, r.top, r.right, r.bottom)
                w, h = rect[2] - rect[0], rect[3] - rect[1]
                key = name.lower()
                if (
                    name and 2 <= len(name) <= 40
                    and ctype in wanted
                    and any(ch.isalnum() for ch in name)
                    and key not in seen
                    and 14 <= w <= 600 and 8 <= h <= 200
                    and rect[0] >= 0 and rect[1] >= 0
                    and not child.IsOffscreen
                ):
                    seen.add(key)
                    found.append((name, rect))
            except Exception:
                pass
            walk(child, depth + 1)

    walk(root, 0)
    if not found:
        die("No usable controls found in that window.",
            "Pick a window with visible buttons and run again.\n"
            "If the window is a web view with no accessibility tree, this probe\n"
            "cannot score it automatically — there is no ground truth to score against.")

    # Spread the picks across the list so they are not all in one toolbar.
    stride = max(1, len(found) // count)
    targets = found[::stride][:count]
    print(green(f"  [OK] {len(targets)} targets with known positions:"))
    for name, rect in targets:
        print(f"       {name!r:<32} {rect}")
    return targets


def grab() -> tuple["object", int, int, str]:
    """Full screenshot + the downscaled copy actually sent to the model."""
    from PIL import ImageGrab
    full = ImageGrab.grab().convert("RGB")
    screen_w, screen_h = full.width, full.height
    sent = full.copy()
    sent.thumbnail((SEND_W, SEND_H))
    buf = io.BytesIO()
    sent.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    print(f"  Screen {screen_w}x{screen_h}  →  sent to model at {sent.width}x{sent.height}")
    return full, screen_w, screen_h, b64, sent.width, sent.height


# ── 5. Ask the model ──────────────────────────────────────────────────────────

QWEN_PROMPT = (
    'Locate the UI element described as: "{target}".\n'
    "Reply with ONLY its bounding box as [x1, y1, x2, y2]. No other text."
)

# What the project sends UI-TARS today (agents/grounding.py).
UITARS_SYSTEM = (
    "You are a GUI grounding agent. Locate the requested UI element in the "
    "screenshot and output a click action with its bounding box on a 0-1000 scale."
)
UITARS_PROMPT = (
    'In this screenshot, locate the UI element described as: "{target}"\n\n'
    "Respond with the bounding box in this EXACT format:\n"
    "click(start_box='[[x1, y1, x2, y2]]')\n\n"
    "Rules:\n"
    "  - x1,y1 = top-left corner of the element\n"
    "  - x2,y2 = bottom-right corner of the element\n"
    "  - All values are on a 0-1000 scale where (0,0)=top-left and (1000,1000)=bottom-right\n"
    "  - Output the action line only, nothing else\n"
    "If the element is not visible in the screenshot: not_found()"
)


def ask(base_url: str, model: str, prompt: str, b64: str,
        system: str = "", timeout: float = 180.0) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ],
    })
    payload = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": 128, "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}{CHAT}", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        return f"<<error: {e}>>"


_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")
_NUM = r"-?\d+(?:\.\d+)?"
_GROUP = re.compile(r"[\[\(]\s*(" + _NUM + r"(?:\s*,\s*" + _NUM + r")+)\s*[\]\)]")
# Pure geometry: brackets, digits, commas, dots, signs, spaces — nothing else.
_GEOMETRY_ONLY = re.compile(r"^['\"][\s\[\(\d.,+\-\]\)]+['\"]$")


def _drop_text(m: "re.Match") -> str:
    """Delete a quoted run unless the quotes are wrapping the coordinates.

    UI-TARS writes click(start_box='[[100,200,140,230]]') — the box lives
    INSIDE the quotes there, so a blanket strip would throw the answer away and
    score the baseline as a total failure.
    """
    return m.group(0) if _GEOMETRY_ONLY.match(m.group(0)) else " "


def _centre(vals: list[float]) -> tuple[float, float] | None:
    """A bbox becomes its centre; a bare pair is already the point."""
    if len(vals) >= 4:
        x1, y1, x2, y2 = vals[:4]
        return (x1 + x2) / 2, (y1 + y2) / 2
    if len(vals) >= 2:
        return vals[0], vals[1]
    return None


def extract_xy(text: str) -> tuple[float, float] | None:
    """Pull one point out of whatever the model said.

    Quoted strings are deleted BEFORE any digit is read. Qwen3-VL's usual reply
    is {"bbox_2d": [x1,y1,x2,y2], "label": "Send"} — scanning that for digits
    naively picks up the 2 in "bbox_2d" as a coordinate and silently ruins the
    answer. Keys and labels are text, never geometry.

    Then a bracketed group of numbers wins over loose digits, because
    "[100, 200, 140, 230]" is unambiguous and a stray token count is not.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "not_found" in text.lower():
        return None

    stripped = _QUOTED.sub(_drop_text, text)
    groups = [
        [float(v) for v in re.split(r"\s*,\s*", g)]
        for g in _GROUP.findall(stripped)
    ]
    for want in (4, 2):
        for g in groups:
            if len(g) == want:
                # (x1,y1),(x2,y2) — two pairs are one box, not two points.
                if want == 2 and len(groups) >= 2 and len(groups[1]) == 2:
                    return _centre(groups[0] + groups[1])
                return _centre(g)

    nums = [float(m) for m in re.findall(_NUM, stripped)]
    return _centre(nums)


# ── 6. Every way those numbers could mean a screen pixel ──────────────────────

def smart_resize(dim: int, factor: int) -> int:
    """Qwen rounds each side to a multiple of patch_size x merge_size."""
    return max(factor, int(round(dim / factor)) * factor)


def interpretations(sent_w: int, sent_h: int, screen_w: int, screen_h: int) -> dict:
    """Name -> f(vx, vy) -> (screen_x, screen_y).

    The project pins "pixels" with factor 28 (qwen2.5-VL). Qwen3-VL is
    patch 16 x merge 2 = 32. On a 1280x720 send that is 728 vs 704 vertically
    — about 3%, which is ~30 px of drift on a 1080p screen. Small, but the same
    order as the misses recorded in config.py. So: measure, don't assume.
    """
    r28w, r28h = smart_resize(sent_w, 28), smart_resize(sent_h, 28)
    r32w, r32h = smart_resize(sent_w, 32), smart_resize(sent_h, 32)

    def mk(dw, dh):
        return lambda vx, vy: (vx / dw * screen_w, vy / dh * screen_h)

    return {
        "norm_0_1":      lambda vx, vy: (vx * screen_w, vy * screen_h),
        "norm_0_1000":   mk(1000, 1000),
        "sent_pixels":   mk(sent_w, sent_h),
        f"resized_28 ({r28w}x{r28h})": mk(r28w, r28h),
        f"resized_32 ({r32w}x{r32h})": mk(r32w, r32h),
        "screen_pixels": lambda vx, vy: (vx, vy),
    }


def inside(pt: tuple[float, float], rect: tuple[int, int, int, int]) -> bool:
    x, y = pt
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def distance(pt: tuple[float, float], rect: tuple[int, int, int, int]) -> float:
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    return ((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2) ** 0.5


# ── 7. Run the grounding test ─────────────────────────────────────────────────

def run_grounding(base_url: str, model: str, label: str, targets: list,
                  b64: str, sent_w: int, sent_h: int,
                  screen_w: int, screen_h: int,
                  prompt_tpl: str, system: str) -> dict:
    print()
    print(bold(f"  {label}  ({model})"))
    readings = interpretations(sent_w, sent_h, screen_w, screen_h)
    scores = {k: {"hits": 0, "dists": []} for k in readings}
    replies = []

    for name, rect in targets:
        t0 = time.time()
        raw = ask(base_url, model, prompt_tpl.format(target=name), b64, system)
        ms = (time.time() - t0) * 1000
        xy = extract_xy(raw)
        replies.append((name, rect, raw, xy))

        short = raw.replace("\n", " ")[:56]
        if xy is None:
            print(f"    {name!r:<28} {ms:6.0f}ms  {red('no numbers')}  {short}")
            continue
        print(f"    {name!r:<28} {ms:6.0f}ms  raw={xy[0]:.0f},{xy[1]:.0f}")
        for key, fn in readings.items():
            pt = fn(*xy)
            scores[key]["dists"].append(distance(pt, rect))
            if inside(pt, rect):
                scores[key]["hits"] += 1

    n = len(targets)
    print()
    print(f"    {'reading':<26} {'hits':>8}   {'avg miss (px)':>14}")
    print(f"    {'-' * 26} {'-' * 8}   {'-' * 14}")
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1]["hits"],
                        sum(kv[1]["dists"]) / len(kv[1]["dists"]) if kv[1]["dists"] else 9e9),
    )
    for key, s in ranked:
        avg = sum(s["dists"]) / len(s["dists"]) if s["dists"] else float("nan")
        line = f"    {key:<26} {s['hits']:>4}/{n:<3}   {avg:>14.0f}"
        print(green(line) if s is ranked[0][1] else line)

    best_key, best = ranked[0]
    return {
        "label": label, "model": model, "best_reading": best_key,
        "hits": best["hits"], "total": n,
        "avg_miss": (sum(best["dists"]) / len(best["dists"])) if best["dists"] else None,
        "replies": replies, "readings": readings,
    }


def annotate(full, result: dict, path: str) -> None:
    """Save a picture: green boxes are truth, red dots are the model's answer."""
    from PIL import ImageDraw
    img = full.copy()
    draw = ImageDraw.Draw(img)
    fn = result["readings"][result["best_reading"]]
    for name, rect, _raw, xy in result["replies"]:
        draw.rectangle(rect, outline=(0, 220, 0), width=3)
        if xy is None:
            continue
        x, y = fn(*xy)
        draw.line([(x - 14, y), (x + 14, y)], fill=(255, 40, 40), width=3)
        draw.line([(x, y - 14), (x, y + 14)], fill=(255, 40, 40), width=3)
        draw.text((x + 16, y + 4), name[:24], fill=(255, 40, 40))
    img.save(path)
    print(f"  Picture saved: {bold(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="Hugging Face model id")
    ap.add_argument("--name", default=DEFAULT_NAME, help="OVMS servable name")
    ap.add_argument("--weight-format", default="int4", choices=["int4", "int8", "fp16"])
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU", "NPU", "AUTO"])
    ap.add_argument("--cache-gb", type=int, default=2)
    ap.add_argument("--targets", type=int, default=6, help="how many controls to test")
    ap.add_argument("--wait", type=int, default=8, help="seconds to click your app")
    ap.add_argument("--baseline", action="store_true",
                    help="also test UI-TARS on :8000 for a head-to-head")
    ap.add_argument("--keep-serving", action="store_true",
                    help="leave the probe OVMS running after the test")
    args = ap.parse_args()

    print(bold("\n  qwen3_vl probe — does MAI-UI-8B have a chance on this machine?\n"))
    os.makedirs(WORK, exist_ok=True)

    binary = preflight()
    prepare(args.source, args.name, args.weight_format, args.device, args.cache_gb)
    proc = serve(binary, args.name)

    print()
    print(green("  ══ Question 1 answered: this OVMS DOES serve qwen3_vl. ══"))

    targets = pick_targets(args.targets, args.wait)

    step(5, "Grounding test")
    full, screen_w, screen_h, b64, sent_w, sent_h = grab()

    result = run_grounding(BASE, args.name, "candidate", targets, b64,
                           sent_w, sent_h, screen_w, screen_h,
                           QWEN_PROMPT, "")

    baseline = None
    if args.baseline:
        if http_get(f"{BASELINE_URL}/v1/config", timeout=3.0)[0] == 200:
            baseline = run_grounding(BASELINE_URL, BASELINE_MODEL,
                                     "baseline (what you run today)", targets, b64,
                                     sent_w, sent_h, screen_w, screen_h,
                                     UITARS_PROMPT, UITARS_SYSTEM)
        else:
            print(yellow(f"\n  [SKIP] Nothing serving on {BASELINE_URL} — no comparison."))

    step(6, "Verdict")
    annotate(full, result, os.path.join(WORK, "candidate.png"))
    if baseline:
        annotate(full, baseline, os.path.join(WORK, "baseline.png"))

    print()
    print(f"  {'model':<34} {'hits':>8}  {'avg miss':>9}  reading")
    print(f"  {'-' * 34} {'-' * 8}  {'-' * 9}  {'-' * 20}")
    rows = [result] + ([baseline] if baseline else [])
    for r in rows:
        miss = f"{r['avg_miss']:.0f}px" if r["avg_miss"] is not None else "n/a"
        print(f"  {r['model']:<34} {r['hits']:>4}/{r['total']:<3}  {miss:>9}  {r['best_reading']}")

    print()
    print(bold("  What to do with this:"))
    print(f"  • Coordinate reading that won: {bold(result['best_reading'])}")
    if result["best_reading"].startswith("resized_32"):
        print("    → change factor 28 to 32 in agents/coords.py (_qwen_resize_dim)")
    elif result["best_reading"].startswith("resized_28"):
        print("    → agents/coords.py is already correct; leave factor 28 alone")
    elif result["best_reading"] == "sent_pixels":
        print("    → the model answers in the sent image's own pixels; the /28")
        print("      rounding in agents/coords.py is wrong for it, drop it")
    elif result["best_reading"] == "norm_0_1000":
        print("    → set VLM_COORD_SPACE = 'norm1000' in config.py")
    if baseline:
        if result["hits"] > baseline["hits"]:
            print(f"  • {green('The candidate pointed better.')} A real swap is justified.")
        elif result["hits"] == baseline["hits"]:
            print(f"  • {yellow('Tie on hits.')} Compare avg miss, and re-run on Teams —")
            print("    that is the case where UI-TARS actually fails you.")
        else:
            print(f"  • {red('UI-TARS still wins here.')} Try these same targets in Teams")
            print("    before deciding; native apps are not where the gap shows.")
    else:
        print("  • Re-run with --baseline to compare against UI-TARS on the same")
        print("    screenshot. That is the only comparison that means anything.")
    print()
    print("  Only the prebuilt base model was tested here. MAI-UI-8B itself:")
    print(f"    python {os.path.relpath(__file__)} --source Tongyi-MAI/MAI-UI-8B \\")
    print("        --name mai-ui-8b-int4-ov --weight-format int4 --baseline")
    print("  (17.6 GB download + a slow CPU conversion — worth it only now that")
    print("   the architecture is proven to load.)")

    if proc and not args.keep_serving:
        print()
        print(yellow("  Stopping the probe server..."))
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30)
        except Exception:
            pass
        print(green("  Done. Your live setup on 8000 was never touched."))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(red("\n  Interrupted."))
        sys.exit(130)
