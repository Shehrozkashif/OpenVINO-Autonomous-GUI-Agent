# Agent Command Center — UI Design System

A ground-up redesign of the desktop GUI agent's interface: from a 3-tab utility
window to an **AI operating console** — the surface through which a human
supervises an autonomous agent operating their computer.

---

## 1. Design vision

**One sentence:** *the operating system for an AI agent.*

| Principle | How the UI expresses it |
|---|---|
| Intelligence | Violet planning states, reasoning feed, memory-recall events |
| Autonomy | The agent has a body: the **PulseOrb** breathes when idle, orbits when working |
| Reliability | Per-step verification confidence shown everywhere, never hidden |
| Speed | Timing metrics on every mission; 60fps micro-animations |
| Trust | Firewall/guard events surfaced inline; Stop is always one click away |
| Sophistication | Glass surfaces, layered depth, restrained motion — no noise |

The interface never *pretends*. Every visualization is backed by a real signal
from the pipeline (no fake bounding boxes, no decorative progress bars).

---

## 2. Architecture

```
core/orchestrator.py ──log(str)──► WorkerSignals.log_update   (worker thread → UI thread)
                                        │
                                        ▼
                              ui/events.AgentEventBus          ← regex parser + state machine
                 ┌──────────────┬───────┴────────┬──────────────────┐
                 ▼              ▼                ▼                  ▼
        MissionPage timeline  IntelligencePanel  StatusChip/orbs  ScreenPreview chip
```

* `ui/theme.py` — design tokens (color/type/spacing) + the single global QSS.
* `ui/icons.py` — QPainter vector icon set. No asset files, DPI-perfect, recolorable.
* `ui/widgets.py` — component library (orb, cards, rail, timeline, dock…).
* `ui/events.py` — **AgentEventBus**: turns the orchestrator's log stream into
  typed signals + an agent state machine. Core stays 100 % UI-agnostic.
* `ui/panels.py` — right-hand Intelligence Panel.
* `ui/pages.py` — the seven workspace pages.
* `ui/main_window.py` — shell. Public contract for `main.py` unchanged
  (`DesktopGUIAgent(orchestrator=)`, `.instruction_input`, `._run_task`).

**Hard constraint:** the window title must contain `"Desktop GUI Agent"` —
the orchestrator masks its own window out of screen captures by that substring.

---

## 3. Color palette

| Token | Value | Use |
|---|---|---|
| `BG0 / BG1` | `#07090E → #0B0E15` | Window gradient (deep space) |
| `PANEL` | `rgba(255,255,255,8)` | Glass card fill |
| `STROKE` | `rgba(255,255,255,18)` | 1 px hairline borders |
| `TEXT / DIM / FAINT` | `#E8EDF4 / #97A3B4 / #5C6877` | 3-level text hierarchy |
| `ACCENT` | `#22D3EE` cyan | The agent acting / presence |
| `ACCENT2` | `#7C6CF6` violet | Thinking: routing, planning, vision |
| `SUCCESS / WARNING / DANGER / INFO` | `#34D399 / #F5B544 / #F8716E / #60A5FA` | Verification & safety semantics |

Action-type hues (timeline chips): click=cyan, type=violet, keys=blue,
scroll=teal, drag=pink, extract=amber, wait=gray.

**State → color language** (used by orb, chips, headlines):
idle gray · routing/planning violet · executing cyan · verifying blue ·
recovering amber · complete green · failed red.

## 4. Typography

Segoe UI Variable Display (native Win 11) → Segoe UI → Inter.
Mono: Cascadia Code → Consolas.

Scale: 26 display / 19 H1 / 15 H2 / 13 body / 12 small / 11 MICRO
(micro = uppercase, +1px letter-spacing, faint — used for section captions).

## 5. Spacing & depth

4 px grid (4/8/12/16/24). Radii 8/12/16. Depth is three layers:
window gradient + color glows → glass panels → floating elements
(one `QGraphicsDropShadowEffect` per floating card, never nested).
Glassmorphism is **simulated** (translucent fill + hairline) — real blur is
CPU-rendered in Qt and would steal cycles from inference.

---

## 6. Screens

### 6.1 Shell (every screen)

```
┌──┬──────────────────────────────────────────┬─────────────┐
│N │ Page title              ●state  ▭toggle  │ AGENT STATE │
│a │ ┌──────────────────────────────────────┐ │ ◉ orb+label │
│v │ │            page stack                │ │ OBJECTIVE   │
│  │ │                                      │ │ CONFIDENCE ▓│
│r │ │                                      │ │ REASONING   │
│a │ └──────────────────────────────────────┘ │  feed…      │
│il│ [◉ command input……………………  ■Stop ▶Run]    │ >raw log    │
└──┴──────────────────────────────────────────┴─────────────┘
```

* **NavRail** (64 px ↔ 196 px): icon-only; expands 160 ms after hover-enter
  (220 ms OutCubic on min+max width), collapses on leave. Active item gets a
  cyan tint block. Tooltips cover the collapsed state.
* **Command dock**: persistent operator bar — mini-orb, auto-growing input
  (1–4 lines, Enter runs, Shift+Enter newline), Stop (visible only while
  running), Run. Readonly while a mission runs.
* **Intelligence panel** (312 px, collapsible): agent state header,
  current objective, animated confidence bar, reasoning/activity feed
  (border-color-coded by kind), collapsible raw console (the old log,
  preserved verbatim for debugging).

### 6.2 Home — hero

Hierarchy: `PulseOrb(104)` + headline → **MissionComposer** (the primary
NL prompt box with `Run Task`) → suggested-mission chips (2 from memory +
canned) → metric tiles (automations learned / successful runs / avg duration /
failure patterns avoided) → recent automations (clickable cards).

UX rationale: the first five seconds must say *"an intelligent agent is ready —
tell it what to do."* The composer is the largest interactive element on
screen; suggestion chips drop text **into the composer** (visible feedback at
the point of click), never silently elsewhere. Empty state offers two runnable
example missions.

### 6.3 Mission Control — live execution

`ScreenPreview` (rounded live view, 1 fps frames, LIVE badge, current-action
chip, cyan scan sweep only while active) → stats strip (objective / steps /
recoveries / last confidence) → **execution timeline**: subtask headers with
numbered badges; steps with action-type chip, pulsing dot while running,
✓+confidence on verify, ✕+reason tooltip on failure, retry counters; inline
guard rows (FIREWALL red, VISION violet, SEARCH blue, CHECK amber).

Honesty rule: the sweep means "agent active", the chip shows the *actual*
current step. Bounding boxes are future work (requires the grounder exposing
coordinates — see §9).

### 6.4 Agent Sessions / 6.5 Workflows / 6.6 Memory

Sessions: history cards (runs badge, subtask count, avg duration, last run,
**Run again**). Workflows: the same memory reframed as a *library* — 2-column
grid, primary **Run workflow** per card; sells the agent's learning loop.
Memory: transparency view — semantic memory (learned tasks, reinforcement
counts) and episodic memory (failure patterns the planner now avoids). Seeing
"what it learned to avoid" converts failures into visible progress → trust.

### 6.7 Screen History

3-column grid of frames captured (~1 per 3 s, ring buffer of 48) while
missions run, captioned with time + the step being executed; click to inspect
full-size. An audit trail of literally *what the agent saw*.

### 6.8 Settings

Glass cards: Intelligence stack (models/endpoints from `config.py` +
threaded **Check backend health**), Safety & control (firewall, kill switch,
verification transparency, credential redaction — stated in plain language),
Credentials (keyring-backed manager).

---

## 7. Motion specification

| Animation | Mechanism | Duration / rate |
|---|---|---|
| Orb breathing | QTimer 33 ms + sin phase, radial gradients | continuous, visible-only |
| Orb busy arc | QConicalGradient rotation | ~3× breathing speed |
| Nav expand | 2× QPropertyAnimation (min/max width) | 220 ms OutCubic |
| Page change | opacity fade-in (effect dropped on finish) | 200 ms |
| Timeline insert | fade-in + auto-scroll | 260 ms |
| Status chip | QVariantAnimation color crossfade | 350 ms |
| Confidence bar | QVariantAnimation fill | 420 ms OutCubic |
| Scan sweep | QLinearGradient offset | 33 ms tick, active-only |
| Skeleton shimmer | gradient offset | 40 ms tick, visible-only |
| Console reveal | maximumHeight animation | 220 ms OutCubic |

Performance budget: **zero continuous timers when idle and hidden** — every
timer starts on `showEvent`/activity and stops on `hideEvent`/idle. Screen
preview stays at 1 fps (PNG over signal, as before). Opacity effects are
removed after each fade (stacked `QGraphicsOpacityEffect`s slow painting).

---

## 8. Trust design

* The state machine is always visible (chip in header + orb in 3 places).
* Every step shows its **verification confidence**; uncertainty is rendered
  amber, never hidden.
* Safety events (action firewall, own-console guard, loop guard) appear
  inline in the timeline at the moment they fire.
* Stop is visible whenever the agent is running; Settings explains the kill
  switch and firewall in plain language.
* Buttons fail safe: Run without the model server → "Agent offline" dialog with
  the fix (`python start.py`).

## 9. Future work

* **Grounding overlay**: draw real click points / element boxes on the
  preview once `UIGroundingAgent.ground()` results are surfaced as events.
* **Floating mission HUD** (always-on-top pill during execution): valuable,
  but its text would appear in the agent's own screen captures and could
  contaminate OCR/planning — needs the same masking treatment as the main
  window before shipping.
* Per-session step-level history persistence (currently sessions persist via
  TaskMemory; step traces live only in the timeline).

## 10. Verification

* `tests/unit/test_ui_smoke.py` — feeds verbatim orchestrator log lines
  through the runtime signal path and asserts timeline/panel/stats state;
  checks the `main.py` contract and the window-title masking contract.
* `tests/unit/test_ui_interactions.py` — click-through audit: every button is
  fired programmatically and its effect asserted (this caught a real
  `isVisible()`-while-minimized toggle bug).
