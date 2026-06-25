# ui/theme.py
"""
Design tokens + global stylesheet for the agent command center.

Single source of truth for color, typography, spacing and radii.
All glass surfaces are translucent fills + 1px light strokes (no real blur —
QGraphicsBlurEffect is CPU-rendered and would fight the VLM for resources).
"""
from PyQt6.QtGui import QColor

# ── Color tokens ──────────────────────────────────────────────────────────────

class C:
    # Backgrounds (deep space, blue-black)
    BG0        = "#07090E"   # window base
    BG1        = "#0B0E15"   # workspace
    PANEL      = "rgba(255, 255, 255, 8)"    # glass fill   (alpha 0-255 in qss rgba)
    PANEL_HI   = "rgba(255, 255, 255, 14)"   # hovered glass
    PANEL_SOLID = "#11151E"
    STROKE     = "rgba(255, 255, 255, 18)"   # 1px hairlines
    STROKE_HI  = "rgba(255, 255, 255, 36)"

    # Text
    TEXT       = "#E8EDF4"
    TEXT_DIM   = "#97A3B4"
    TEXT_FAINT = "#5C6877"

    # Brand / state accents
    ACCENT     = "#22D3EE"   # electric cyan — the agent's presence color
    ACCENT2    = "#7C6CF6"   # violet — intelligence / planning
    SUCCESS    = "#34D399"
    WARNING    = "#F5B544"
    DANGER     = "#F8716E"
    INFO       = "#60A5FA"

    # Action-type hues (timeline chips)
    ACTION_HUES = {
        "click":        "#22D3EE",
        "double_click": "#22D3EE",
        "right_click":  "#2DD4BF",
        "type":         "#7C6CF6",
        "key_press":    "#60A5FA",
        "hotkey":       "#60A5FA",
        "scroll":       "#2DD4BF",
        "drag":         "#F472B6",
        "extract":      "#F5B544",
        "wait":         "#97A3B4",
    }


def qcolor(hex_or_rgba: str, alpha: int = 255) -> QColor:
    c = QColor(hex_or_rgba)
    if alpha != 255:
        c.setAlpha(alpha)
    return c


# ── Typography ────────────────────────────────────────────────────────────────

class T:
    FAMILY      = '"Segoe UI Variable Display", "Segoe UI", "Inter", sans-serif'
    FAMILY_MONO = '"Cascadia Code", "Consolas", monospace'

    DISPLAY = 26   # hero headline
    H1      = 19   # page titles
    H2      = 15   # card titles
    BODY    = 13
    SMALL   = 12
    MICRO   = 11   # chips, captions  (uppercase + letter-spacing)


# ── Spacing / radii ───────────────────────────────────────────────────────────

class S:
    XS, SM, MD, LG, XL = 4, 8, 12, 16, 24
    RADIUS    = 12
    RADIUS_SM = 8
    RADIUS_LG = 16


# ── Agent state → presentation ────────────────────────────────────────────────

STATE_STYLE = {
    "IDLE":       (C.TEXT_DIM,  "Ready"),
    "LISTENING":  (C.ACCENT,    "Listening"),
    "ROUTING":    (C.ACCENT2,   "Decomposing task"),
    "PLANNING":   (C.ACCENT2,   "Planning"),
    "GROUNDING":  ("#2DD4BF",   "Locating element"),
    "ACTING":     (C.ACCENT,    "Executing"),
    "VERIFYING":  (C.INFO,      "Verifying"),
    "RECOVERING": (C.WARNING,   "Recovering"),
    "COMPLETE":   (C.SUCCESS,   "Complete"),
    "FAILED":     (C.DANGER,    "Failed"),
    "STOPPED":    (C.WARNING,   "Stopped"),
}


# ── Global stylesheet ─────────────────────────────────────────────────────────

def build_stylesheet() -> str:
    return f"""
    QWidget {{
        font-family: {T.FAMILY};
        font-size: {T.BODY}px;
        color: {C.TEXT};
        selection-background-color: {C.ACCENT2};
        selection-color: white;
    }}
    QToolTip {{
        background: {C.PANEL_SOLID};
        color: {C.TEXT};
        border: 1px solid {C.STROKE_HI};
        border-radius: 6px;
        padding: 6px 10px;
    }}
    QMessageBox {{ background: {C.PANEL_SOLID}; }}

    /* ── Glass surfaces ─────────────────────────────────────── */
    GlassCard {{
        background: {C.PANEL};
        border: 1px solid {C.STROKE};
        border-radius: {S.RADIUS}px;
    }}
    GlassCard[hoverable="true"]:hover {{
        background: {C.PANEL_HI};
        border: 1px solid {C.STROKE_HI};
    }}
    GlassCard[tone="accent"] {{
        border: 1px solid rgba(34, 211, 238, 60);
    }}

    /* ── Navigation rail ────────────────────────────────────── */
    NavRail {{
        background: rgba(255, 255, 255, 5);
        border-right: 1px solid {C.STROKE};
    }}
    NavItem {{
        background: transparent;
        border: none;
        border-radius: {S.RADIUS_SM}px;
        text-align: left;
        padding: 0px 10px;
        color: {C.TEXT_DIM};
        font-size: {T.BODY}px;
    }}
    NavItem:hover {{ background: {C.PANEL_HI}; color: {C.TEXT}; }}
    NavItem[active="true"] {{
        background: rgba(34, 211, 238, 26);
        color: {C.ACCENT};
        font-weight: 600;
    }}

    /* ── Command dock ───────────────────────────────────────── */
    CommandDock {{
        background: rgba(17, 21, 30, 235);
        border: 1px solid {C.STROKE_HI};
        border-radius: {S.RADIUS_LG}px;
    }}
    CommandInput {{
        background: transparent;
        border: none;
        font-size: 14px;
        padding: 4px;
        color: {C.TEXT};
    }}

    /* ── Buttons ────────────────────────────────────────────── */
    QPushButton[kind="primary"] {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1FB8D6, stop:1 {C.ACCENT2});
        color: #06121A;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 9px 20px;
    }}
    QPushButton[kind="primary"]:hover  {{ background: {C.ACCENT}; }}
    QPushButton[kind="primary"]:disabled {{
        background: rgba(255,255,255,16); color: {C.TEXT_FAINT};
    }}
    QPushButton[kind="danger"] {{
        background: rgba(248, 113, 110, 30);
        color: {C.DANGER};
        border: 1px solid rgba(248, 113, 110, 70);
        border-radius: 10px;
        padding: 9px 16px;
        font-weight: 600;
    }}
    QPushButton[kind="danger"]:hover {{ background: rgba(248, 113, 110, 55); }}
    QPushButton[kind="ghost"] {{
        background: transparent;
        color: {C.TEXT_DIM};
        border: 1px solid {C.STROKE};
        border-radius: 10px;
        padding: 7px 14px;
    }}
    QPushButton[kind="ghost"]:hover {{
        color: {C.TEXT}; border-color: {C.STROKE_HI}; background: {C.PANEL};
    }}
    QPushButton[kind="chip"] {{
        background: {C.PANEL};
        color: {C.TEXT_DIM};
        border: 1px solid {C.STROKE};
        border-radius: 14px;
        padding: 6px 14px;
        font-size: {T.SMALL}px;
    }}
    QPushButton[kind="chip"]:hover {{
        color: {C.ACCENT}; border-color: rgba(34, 211, 238, 90);
        background: rgba(34, 211, 238, 16);
    }}

    /* ── Inputs ─────────────────────────────────────────────── */
    QLineEdit, QPlainTextEdit, QTextEdit {{
        background: rgba(255, 255, 255, 10);
        border: 1px solid {C.STROKE};
        border-radius: {S.RADIUS_SM}px;
        padding: 7px 10px;
        color: {C.TEXT};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
        border: 1px solid rgba(34, 211, 238, 110);
        background: rgba(255, 255, 255, 14);
    }}

    /* ── Scrollbars (thin, ghost) ───────────────────────────── */
    QScrollArea {{ background: transparent; border: none; }}
    QScrollBar:vertical {{
        background: transparent; width: 8px; margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: rgba(255, 255, 255, 40); border-radius: 4px; min-height: 32px;
    }}
    QScrollBar::handle:vertical:hover {{ background: rgba(255, 255, 255, 80); }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
    QScrollBar::handle:horizontal {{
        background: rgba(255, 255, 255, 40); border-radius: 4px; min-width: 32px;
    }}

    /* ── Labels by role ─────────────────────────────────────── */
    QLabel[role="display"] {{ font-size: {T.DISPLAY}px; font-weight: 700; }}
    QLabel[role="h1"]      {{ font-size: {T.H1}px; font-weight: 700; }}
    QLabel[role="h2"]      {{ font-size: {T.H2}px; font-weight: 600; }}
    QLabel[role="dim"]     {{ color: {C.TEXT_DIM}; }}
    QLabel[role="faint"]   {{ color: {C.TEXT_FAINT}; font-size: {T.SMALL}px; }}
    QLabel[role="micro"]   {{
        color: {C.TEXT_FAINT}; font-size: {T.MICRO}px;
        font-weight: 700; letter-spacing: 1px;
    }}
    QLabel[role="mono"] {{
        font-family: {T.FAMILY_MONO}; font-size: {T.SMALL}px; color: {C.TEXT_DIM};
    }}
    QLabel[role="metric"] {{ font-size: 24px; font-weight: 700; color: {C.TEXT}; }}

    /* ── Console (raw log) ──────────────────────────────────── */
    QPlainTextEdit[role="console"] {{
        background: rgba(0, 0, 0, 90);
        border: 1px solid {C.STROKE};
        border-radius: {S.RADIUS_SM}px;
        font-family: {T.FAMILY_MONO};
        font-size: 11px;
        color: {C.TEXT_DIM};
    }}
    """
