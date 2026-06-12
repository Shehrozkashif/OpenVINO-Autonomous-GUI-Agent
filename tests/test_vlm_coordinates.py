"""
End-to-end VLM coordinate accuracy test for UI-TARS-1.5-7B.

What it does:
  1. Captures a live screenshot.
  2. Calls _vlm_coords() directly (bypasses UIA/OCR so ONLY the VLM is tested).
  3. Checks whether predicted (x, y) land in the expected screen region.
  4. Saves an annotated image  tests/vlm_coord_result.png  showing:
       - predicted point (green dot)
       - expected region bounding box (blue rect)
       - PASS/FAIL label per element.

Run:
    cd intel-openvino-desktop-agent
    python tests/test_vlm_coordinates.py
"""
import base64
import io
import sys
import time
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, ".")

from PIL import Image, ImageDraw, ImageFont

from agents.grounding.grounding_agent import OCREngine, UIGroundingAgent
from core.capture.screenshot import ScreenCapture, _screen_size
from core.pipeline.ollama_client import OllamaClient


# ── Expected regions ──────────────────────────────────────────────────────────
# Each entry: (label, expected_region_as_fraction (x0,y0,x1,y1), description)
# Windows 11 default: taskbar at bottom, icons CENTERED (not left-aligned like Win10).
# Start button sits left of the centered cluster (~35-55% x).
# Clock/tray stays in the right corner (>80% x).
TARGETS = [
    ("Start button",    (0.30, 0.88, 0.60, 1.00), "center-bottom (Win11 centered taskbar)"),
    ("taskbar clock",   (0.80, 0.88, 1.00, 1.00), "bottom-right corner"),
    ("taskbar",         (0.00, 0.88, 1.00, 1.00), "full bottom strip"),
    ("desktop background", (0.05, 0.00, 0.95, 0.88), "main screen area"),
    ("search bar",      (0.20, 0.88, 0.65, 1.00), "center-bottom search area"),
]


@dataclass
class CoordResult:
    target: str
    predicted_x: Optional[int]
    predicted_y: Optional[int]
    conf: float
    in_region: bool
    raw_response: str
    latency_ms: float


def _encode(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _in_region(x, y, region, sw, sh) -> bool:
    x0, y0, x1, y1 = region
    return (x0 * sw <= x <= x1 * sw) and (y0 * sh <= y <= y1 * sh)


def run_vlm_coord_test():
    print("\nUI-TARS-1.5-7B — End-to-End Coordinate Accuracy Test")
    print("=" * 57)

    # ── Setup ──────────────────────────────────────────────────────────────────
    capturer = ScreenCapture()
    client   = OllamaClient()
    ocr      = OCREngine()
    grounder = UIGroundingAgent(client, capturer, ocr=ocr)
    sw, sh   = _screen_size()

    print(f"\nScreen: {sw}×{sh}")
    print(f"VLM model: {client.vlm_model}")
    print(f"VLM backend: {'vLLM' if client.vlm_base_url != client.llm_base_url else 'Ollama'}")

    # ── Capture once and reuse ─────────────────────────────────────────────────
    print("\nCapturing screenshot…")
    screenshot = capturer.capture()
    display    = screenshot.copy()
    display.thumbnail((1280, 720), Image.LANCZOS)
    dw, dh     = display.width, display.height
    scale_x    = sw / dw
    scale_y    = sh / dh
    img_b64    = _encode(display)

    print(f"Display size for VLM: {dw}×{dh}  (scale {scale_x:.2f}×{scale_y:.2f})")

    # ── Run VLM for each target ────────────────────────────────────────────────
    results = []
    for (target, region, hint) in TARGETS:
        print(f"\n  [{target}]  (expected: {hint})")
        t0 = time.time()
        raw = "—"
        try:
            # Patch _vlm_coords to also return raw response for debugging
            from agents.grounding.grounding_agent import _VLM_COORD_PROMPT, _UITARS_SYSTEM_PROMPT
            resp = client.query_vlm(
                prompt=_VLM_COORD_PROMPT.format(target=target),
                image_base64=img_b64,
                max_tokens=150,
                temperature=0.0,
                system_prompt=_UITARS_SYSTEM_PROMPT,
            )
            raw = resp.content.strip()
            latency = (time.time() - t0) * 1000
            parsed = grounder._parse_coords(raw, sw, sh, dw, dh)
        except Exception as e:
            latency = (time.time() - t0) * 1000
            print(f"    ERROR: {e}")
            results.append(CoordResult(target, None, None, 0.0, False, str(e), latency))
            continue

        if parsed:
            px, py, conf = parsed
            ok = _in_region(px, py, region, sw, sh)
            status = "PASS" if ok else "FAIL"
            print(f"    Predicted : ({px}, {py})  conf={conf:.2f}")
            x0f, y0f, x1f, y1f = region
            print(f"    Expected  : x in [{int(x0f*sw)},{int(x1f*sw)}]  y in [{int(y0f*sh)},{int(y1f*sh)}]")
            print(f"    Raw VLM   : {raw[:120]}")
            print(f"    Latency   : {latency:.0f}ms")
            print(f"    Result    : [{status}]")
            results.append(CoordResult(target, px, py, conf, ok, raw, latency))
        else:
            print("    Predicted : NOT FOUND / parse failed")
            print(f"    Raw VLM   : {raw[:120]}")
            print(f"    Latency   : {latency:.0f}ms")
            print("    Result    : [FAIL — no coordinates parsed]")
            results.append(CoordResult(target, None, None, 0.0, False, raw, latency))

    # ── Annotate and save screenshot ───────────────────────────────────────────
    annotated = screenshot.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        font_sm = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    colors = {
        "pass_region": (0, 120, 255, 80),
        "fail_region": (255, 60, 60, 80),
        "pass_dot":    (0, 230, 80),
        "fail_dot":    (255, 50, 50),
        "label_bg":    (0, 0, 0, 160),
    }

    for r, (target, region, _) in zip(results, TARGETS):
        x0f, y0f, x1f, y1f = region
        rx0, ry0 = int(x0f * sw), int(y0f * sh)
        rx1, ry1 = int(x1f * sw), int(y1f * sh)

        color = colors["pass_region"] if r.in_region else colors["fail_region"]
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        ov_draw.rectangle([rx0, ry0, rx1, ry1], fill=color)
        annotated = Image.alpha_composite(annotated.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(annotated)
        draw.rectangle([rx0, ry0, rx1, ry1],
                       outline=colors["pass_dot"] if r.in_region else colors["fail_dot"],
                       width=3)

        if r.predicted_x is not None:
            dot_color = colors["pass_dot"] if r.in_region else colors["fail_dot"]
            dot_r = 10
            draw.ellipse(
                [r.predicted_x - dot_r, r.predicted_y - dot_r,
                 r.predicted_x + dot_r, r.predicted_y + dot_r],
                fill=dot_color, outline=(255, 255, 255), width=2,
            )
            status = "PASS" if r.in_region else "FAIL"
            label = f"{status}: {target}  ({r.predicted_x},{r.predicted_y})"
            draw.text((r.predicted_x + 14, r.predicted_y - 12), label, fill=dot_color, font=font_sm)

    out_path = "tests/vlm_coord_result.png"
    annotated.save(out_path)
    print(f"\nAnnotated screenshot saved -> {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r.in_region)
    total  = len(results)
    found  = sum(1 for r in results if r.predicted_x is not None)
    avg_ms = sum(r.latency_ms for r in results) / total if total else 0

    print("\n" + "=" * 57)
    print("SUMMARY")
    print("=" * 57)
    print(f"  Targets tested  : {total}")
    print(f"  Coords found    : {found}/{total}")
    print(f"  In expected zone: {passed}/{total}  ({'ALL PASS' if passed == total else 'SOME FAILED'})")
    print(f"  Avg VLM latency : {avg_ms:.0f}ms")
    print(f"  Model           : {client.vlm_model}")
    print()

    for r in results:
        mark = "PASS" if r.in_region else "FAIL"
        coord = f"({r.predicted_x},{r.predicted_y})" if r.predicted_x is not None else "not found"
        print(f"  {mark}  {r.target:<25} {coord}  conf={r.conf:.2f}  {r.latency_ms:.0f}ms")

    print()
    return passed, total


if __name__ == "__main__":
    passed, total = run_vlm_coord_test()
    sys.exit(0 if passed == total else 1)
