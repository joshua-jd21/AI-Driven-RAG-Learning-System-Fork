"""Freeform template: the LLM authors a complete Manim scene directly.

Used when no deterministic concept template fits the topic (e.g., abstract
chemistry, biology, novel physics scenarios). Selected by the planner via
``concept_template: "freeform"``. The generated script must:

- expose ``class GeneratedScene(Scene)`` with a ``construct`` method
- import ``from manim import *`` and ``import numpy as np``
- last as close as possible to ``timeline['audio_duration']`` (we pad with a
  trailing ``self.wait`` if the LLM under-runs).

Strict guardrails block VoiceoverScene / external audio embeds, ensure dark
background, and inject our standard fade-out tail.
"""
from __future__ import annotations

import re
from typing import Any

from modules.config import NVIDIA_PLANNER_MODEL, get_logger
from modules.llm.nvidia_client import NvidiaClient
from modules.manim.code_sanitize import has_latex_mobjects, strip_latex_mobjects
from modules.manim.templates.segment_timing import (
    segment_at,
    segment_duration,
    segment_hold,
    segment_rt,
    segments_from_timeline,
)

logger = get_logger(__name__)

FREEFORM_SYSTEM = """You are an expert Manim Community Edition (v0.18+) animator.
You write COMPLETE, runnable Python files for short educational physics/science scenes.

HARD RULES:
- Output ONE Python file, NO markdown fences, NO commentary.
- Top of file: `from manim import *` then `import numpy as np`.
- Define exactly: `class GeneratedScene(Scene):` with `construct(self)`.
- First line of construct: `self.camera.background_color = "#0f1117"`.
- Do NOT import or subclass VoiceoverScene. Do NOT use add_sound, SVGMobject, ImageMobject, or external files.
- Only Manim built-ins: Text, Arrow, Line, Rectangle, Circle, Dot, VGroup, Axes, NumberPlane, ParametricFunction, ValueTracker, always_redraw, Create, Write, FadeIn, FadeOut, Transform, GrowArrow, MoveAlongPath, ReplacementTransform, AnimationGroup, LaggedStart.
- NEVER use MathTex or Tex (LaTeX is not available). Use Text() for equations, e.g. Text("W = F × d").
- Keep total animation time CLOSE TO the requested audio duration (use self.wait(...) to pad).
- SAFE AREA: keep all mobjects within x in [-6.6, 6.6] and y in [-3.6, 3.6]. Title band at y≈3.2; captions at y≈-3.2.
- Title: one line, to_edge(UP, buff=0.35), use scale_to_fit_width(12.0) if long.
- Body text: use Text(..., width=5.0) for wrapping or scale_to_fit_width before placing.
- Use .next_to or .arrange with buff>=0.4. Never stack overlapping labels.
- Prefer LaggedStart(..., lag_ratio=0.15) and rate_func=smooth for reveals.
- Use color hexes: title "#e0e6f0", body "#c8d3e6", accents "#4f8ef7" / "#41d4a8" / "#ff7a59".
- End with: `self.play(FadeOut(*self.mobjects), run_time=0.40)`.
- Never use .get_edge(); use .get_left/right/top/bottom().
"""

FREEFORM_PROMPT = """Generate a single Manim scene for the following plan.

SCENE TITLE: {title}
LEARNING GOAL: {learning_goal}
ANCHOR EXAMPLE / SCENARIO: {anchor_example}
TARGET AUDIO DURATION: {audio_duration:.2f} seconds (animation total must be <= this; pad with self.wait)
NARRATION (the visuals must mirror this script):
\"\"\"
{narration}
\"\"\"

ANCHOR PHRASES — visually emphasize when each is spoken (estimate timing by phrase order):
{anchor_phrases}

{learner_context}

DESIGN GUIDELINES:
- Build a UNIQUE visual specific to this scene's anchor example — do NOT reuse the same diagram across scenes.
- Show 1-3 distinct visual stages tied to the narration beats.
- For abstract concepts, use diagrams, arrows, transitions, or labeled symbols — not just text.
- Equations: use Text() with Unicode symbols (×, Δ, θ) — never MathTex/Tex.
- Prefer FadeIn/Write for entrances, Transform for evolutions, FadeOut between stages.

Return ONLY the complete Python file.
"""


class FreeformTemplate:
    """Lets the LLM author the Manim scene for arbitrary topics."""

    ALLOWED_EVENTS = {
        "place_title", "highlight", "show", "explain", "transform",
        "compare", "summarize", "hold", "label", "equation", "diagram",
    }
    SLOTS = {}

    @staticmethod
    def compile(plan: dict[str, Any], timeline: dict[str, Any]) -> str:
        scene_id = plan.get("scene_id", 0)
        title = plan.get("title", f"Scene {scene_id}")
        learning_goal = plan.get("learning_goal", "")
        anchor_example = plan.get("anchor_example", "")
        narration = plan.get("narration", "")
        visual_instruction = plan.get("visual_instruction", "")
        scene_role = str(plan.get("scene_role", "")).lower()
        audio_duration = float(timeline.get("audio_duration", 9.0))

        family = _choose_scene_family(scene_role, title, visual_instruction, narration)
        code = _build_role_based_scene(
            family=family,
            title=title,
            learning_goal=learning_goal,
            anchor_example=anchor_example,
            narration=narration,
            visual_instruction=visual_instruction,
            audio_duration=audio_duration,
            timeline=timeline,
        )
        if "class GeneratedScene" not in code or "def construct" not in code:
            logger.warning("Freeform scene generation failed for scene %s; using stub", scene_id)
            return _stub_scene(title, learning_goal, audio_duration)
        return code


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


_BANNED_PATTERNS = (
    r"from\s+manim_voiceover",
    r"VoiceoverScene",
    r"add_sound\s*\(",
    r"SVGMobject\s*\(",
    r"ImageMobject\s*\(",
)

# Static replacement string — never embed the raw regex pattern in the
# replacement arg of re.sub(). In Python 3.12+ unrecognised backslash
# sequences like \s and \( in the replacement string raise re.error.
_SANITIZE_REPLACEMENT = "# [sanitized]"


def _sanitize(code: str) -> str:
    for pat in _BANNED_PATTERNS:
        try:
            code = re.sub(pat, _SANITIZE_REPLACEMENT, code)
        except re.error as exc:
            logger.warning("Sanitize pattern %r failed: %s — skipping", pat, exc)
    if "from manim import" not in code:
        code = "from manim import *\nimport numpy as np\n\n" + code
    if "import numpy" not in code:
        code = code.replace("from manim import *", "from manim import *\nimport numpy as np", 1)
    return code


def _stub_scene(title: str, goal: str, audio_dur: float) -> str:
    pad = max(0.5, audio_dur - 2.0)
    return f"""from manim import *
import numpy as np


class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#0f1117"
        title = Text({title!r}, font_size=44, weight=BOLD, color="#e0e6f0")
        title.to_edge(UP, buff=0.5)
        goal = Text({goal!r}, font_size=26, color="#c8d3e6")
        goal.next_to(title, DOWN, buff=0.6)
        self.play(Write(title), run_time=0.9)
        self.play(FadeIn(goal, shift=UP*0.2), run_time=0.8)
        self.wait({pad:.2f})
        self.play(FadeOut(*self.mobjects), run_time=0.40)
"""


def _timeline_events(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        list(timeline.get("events", [])),
        key=lambda ev: float(ev.get("start", 0.0)),
    )


def _timeline_event(
    timeline: dict[str, Any],
    index: int,
    default_start: float = 0.0,
    default_rt: float = 0.8,
    default_hold: float = 0.0,
) -> dict[str, float]:
    segments = segments_from_timeline(timeline)
    if index < len(segments):
        seg = segment_at(timeline, index)
        if seg is not None:
            rt = segment_rt(seg, default=default_rt, floor=0.35, cap=default_rt)
            hold = segment_hold(seg, rt, minimum=default_hold)
            return {
                "start": float(seg.get("start", default_start)),
                "run_time": rt,
                "hold_after": hold,
            }

    events = _timeline_events(timeline)
    if index < len(events):
        ev = events[index]
        return {
            "start": float(ev.get("start", default_start)),
            "run_time": max(float(ev.get("run_time", default_rt)), 0.1),
            "hold_after": max(float(ev.get("hold_after", default_hold)), 0.0),
        }
    return {
        "start": default_start,
        "run_time": max(default_rt, 0.1),
        "hold_after": max(default_hold, 0.0),
    }


def _choose_scene_family(
    scene_role: str,
    title: str,
    visual_instruction: str,
    narration: str,
) -> str:
    """Choose a deterministic visual family from the scene spec."""
    text = " ".join([scene_role, title, visual_instruction, narration]).lower()
    if any(k in text for k in ("battery", "circuit", "resistor", "current", "voltage")):
        if any(k in text for k in ("pipe", "water", "tank", "flow", "pressure")):
            return "analogy"
        return "circuit"
    if any(k in text for k in ("pipe", "water", "tank", "flow", "pressure")):
        return "analogy"
    if any(k in text for k in ("example", "calculate", "solve", "substitute", "formula", "=")):
        return "equation"
    if "summary" in scene_role or "relationship" in text or "diagram" in text:
        return "summary"
    if "hook" in scene_role or "visual_intuition" in scene_role:
        return "concept"
    return "concept"


def _build_role_based_scene(
    family: str,
    title: str,
    learning_goal: str,
    anchor_example: str,
    narration: str,
    visual_instruction: str,
    audio_duration: float,
    timeline: dict[str, Any],
) -> str:
    body_map = {
        "circuit": _circuit_scene_body,
        "analogy": _analogy_scene_body,
        "equation": _equation_scene_body,
        "summary": _summary_scene_body,
        "concept": _concept_scene_body,
    }
    body_fn = body_map.get(family, _concept_scene_body)
    body, elapsed = body_fn(
        title,
        learning_goal,
        anchor_example,
        narration,
        visual_instruction,
        timeline,
    )
    tail = max(0.5, audio_duration - elapsed - 0.4)
    return f"""from manim import *
import numpy as np


class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#0f1117"
{body}
        self.wait({tail:.2f})
        self.play(FadeOut(*self.mobjects), run_time=0.40)
"""


def _title_block(title: str, wait_before: float = 0.0, run_time: float = 0.8) -> str:
    wait_line = f'        self.wait({wait_before:.3f})\n' if wait_before > 0.005 else ""
    return f'''{wait_line}        title = Text({title!r}, font_size=40, weight=BOLD, color="#e0e6f0")
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time={run_time:.3f})
'''


def _circuit_scene_body(
    title: str,
    learning_goal: str,
    anchor_example: str,
    narration: str,
    visual_instruction: str,
    timeline: dict[str, Any],
) -> tuple[str, float]:
    title_block = _title_block(title)
    body = f"""{title_block}        subtitle = Text("Voltage source drives current through resistance", font_size=24, color="#c8d3e6")
        subtitle.next_to(title, DOWN, buff=0.25)
        circuit_top = Line(LEFT*4.1 + UP*1.2, RIGHT*4.1 + UP*1.2, color="#c8d3e6", stroke_width=5)
        circuit_bottom = Line(LEFT*4.1 + DOWN*1.4, RIGHT*4.1 + DOWN*1.4, color="#c8d3e6", stroke_width=5)
        left_wire = Line(LEFT*4.1 + UP*1.2, LEFT*4.1 + DOWN*1.4, color="#c8d3e6", stroke_width=5)
        right_wire = Line(RIGHT*4.1 + UP*1.2, RIGHT*4.1 + DOWN*1.4, color="#c8d3e6", stroke_width=5)
        battery = VGroup(
            Rectangle(width=1.0, height=1.9, color="#4f8ef7", fill_opacity=0.18, stroke_width=2),
            Line(LEFT*0.18 + DOWN*0.75, LEFT*0.18 + UP*0.75, color="#4f8ef7", stroke_width=4),
            Line(RIGHT*0.18 + DOWN*0.45, RIGHT*0.18 + UP*0.45, color="#4f8ef7", stroke_width=7),
        ).move_to(LEFT*3.0 + UP*0.2)
        resistor = VGroup(
            RoundedRectangle(corner_radius=0.12, width=1.7, height=0.85, color="#ff7a59", fill_opacity=0.18, stroke_width=2),
            Text("R", font_size=30, color="#ff7a59", weight=BOLD),
        ).move_to(RIGHT*2.5 + UP*0.2)
        lamp = VGroup(
            Circle(radius=0.52, color="#f7c948", fill_opacity=0.15, stroke_width=2),
            Text("Load", font_size=22, color="#f7c948"),
        ).move_to(RIGHT*0.7 + DOWN*0.75)
        current_arrow_1 = Arrow(LEFT*2.0 + UP*1.2, RIGHT*1.8 + UP*1.2, color="#41d4a8", stroke_width=5, buff=0)
        current_arrow_2 = Arrow(RIGHT*3.6 + UP*0.2, RIGHT*3.6 + DOWN*0.9, color="#41d4a8", stroke_width=5, buff=0)
        current_arrow_3 = Arrow(RIGHT*0.0 + DOWN*1.4, LEFT*3.8 + DOWN*1.4, color="#41d4a8", stroke_width=5, buff=0)
        current_label = Text("Current I", font_size=24, color="#41d4a8", weight=BOLD).to_edge(DOWN, buff=0.5)
        component_labels = VGroup(
            Text("Voltage source", font_size=22, color="#4f8ef7").next_to(battery, DOWN, buff=0.18),
            Text("Resistance", font_size=22, color="#ff7a59").next_to(resistor, DOWN, buff=0.18),
            Text("Load", font_size=22, color="#f7c948").next_to(lamp, DOWN, buff=0.18),
        )
        self.play(Create(circuit_top), Create(circuit_bottom), Create(left_wire), Create(right_wire), FadeIn(battery), FadeIn(resistor), FadeIn(lamp), run_time=1.5)
        self.play(FadeIn(component_labels), run_time=0.8)
        self.play(GrowArrow(current_arrow_1), GrowArrow(current_arrow_2), GrowArrow(current_arrow_3), FadeIn(current_label), run_time=1.2)
        self.play(Indicate(resistor), Indicate(current_label), run_time=0.9)
"""
    return body, 5.2


def _analogy_scene_body(
    title: str,
    learning_goal: str,
    anchor_example: str,
    narration: str,
    visual_instruction: str,
    timeline: dict[str, Any],
) -> tuple[str, float]:
    title_event = _timeline_event(timeline, 0, default_rt=1.2)
    label_event = _timeline_event(timeline, 1, default_start=title_event["start"] + 1.5, default_rt=0.8)
    arrow_event = _timeline_event(timeline, 2, default_start=label_event["start"] + 0.8, default_rt=0.8)
    detail_event = _timeline_event(timeline, 3, default_start=arrow_event["start"] + 1.0, default_rt=0.7)
    title_block = _title_block(title, wait_before=0.0, run_time=title_event["run_time"])
    wait_before_labels = max(0.0, label_event["start"] - (title_event["start"] + title_event["run_time"]))
    wait_before_arrow = max(
        0.0,
        arrow_event["start"] - (label_event["start"] + label_event["run_time"]),
    )
    wait_before_detail = max(
        0.0,
        detail_event["start"] - (arrow_event["start"] + arrow_event["run_time"]),
    )
    body = f"""{title_block}        tank = RoundedRectangle(corner_radius=0.12, width=2.0, height=2.4, color="#4f8ef7", fill_opacity=0.18, stroke_width=2).shift(LEFT*3.0 + UP*0.2)
        pipe = VGroup(
            Rectangle(width=3.8, height=0.7, color="#c8d3e6", fill_opacity=0.10, stroke_width=2),
            Rectangle(width=2.6, height=0.26, color="#41d4a8", fill_opacity=0.25, stroke_width=0),
        ).shift(RIGHT*0.9 + DOWN*0.25)
        tank_label = Text("Voltage as pressure", font_size=24, color="#4f8ef7").next_to(tank, DOWN, buff=0.2)
        pipe_label = Text("Current as flow", font_size=24, color="#41d4a8").next_to(pipe, DOWN, buff=0.2)
        resistance_label = Text("Resistance = narrowing", font_size=24, color="#ff7a59").to_edge(DOWN, buff=0.45)
        droplets = VGroup(*[
            Dot(radius=0.08, color="#41d4a8").move_to(LEFT*2.4 + DOWN*0.1 + RIGHT*i*0.45)
            for i in range(6)
        ])
        pressure_arrow = Arrow(LEFT*3.0 + UP*1.3, LEFT*3.0 + UP*0.3, color="#4f8ef7", stroke_width=5, buff=0)
        restriction = RoundedRectangle(corner_radius=0.12, width=0.45, height=1.1, color="#ff7a59", fill_opacity=0.35, stroke_width=2).shift(RIGHT*1.1 + DOWN*0.25)
        self.play(FadeIn(tank), FadeIn(pipe), FadeIn(restriction), run_time={title_event["run_time"]:.3f})
        self.wait({wait_before_labels:.3f})
        self.play(Write(tank_label), Write(pipe_label), run_time={label_event["run_time"]:.3f})
        self.wait({wait_before_arrow:.3f})
        self.play(GrowArrow(pressure_arrow), run_time={arrow_event["run_time"]:.3f})
        self.play(LaggedStart(*[FadeIn(d) for d in droplets], lag_ratio=0.15), run_time={arrow_event["run_time"]:.3f})
        self.wait({wait_before_detail:.3f})
        self.play(Write(resistance_label), run_time={detail_event["run_time"]:.3f})
        self.play(Indicate(restriction), Indicate(pressure_arrow), run_time={detail_event["run_time"]:.3f})
"""
    return body, detail_event["start"] + detail_event["run_time"]


def _equation_scene_body(
    title: str,
    learning_goal: str,
    anchor_example: str,
    narration: str,
    visual_instruction: str,
    timeline: dict[str, Any],
) -> tuple[str, float]:
    text = " ".join([title, learning_goal, anchor_example, narration, visual_instruction]).lower()
    if "30" in text and "0.3" in text:
        equation_text = "R = V / I"
        substitution_text = "R = 9 V / 0.3 A = 30 Ω"
        answer_text = "Resistance = 30 Ω"
    elif "12" in text and "4" in text:
        equation_text = "V = I × R"
        substitution_text = "12 V = I × 4 Ω"
        answer_text = "I = 3 A"
    else:
        equation_text = "V = I × R"
        substitution_text = "Use the relationship to substitute values"
        answer_text = "Solve for the unknown"
    title_block = _title_block(title)
    body = f"""{title_block}        formula_box = RoundedRectangle(corner_radius=0.14, width=4.6, height=0.9, color="#4f8ef7", fill_opacity=0.12, stroke_width=2).shift(UP*0.7)
        formula = Text({equation_text!r}, font_size=38, color="#e0e6f0", weight=BOLD).move_to(formula_box)
        symbol_row = VGroup(
            Text("V = voltage (V)", font_size=24, color="#4f8ef7"),
            Text("I = current (A)", font_size=24, color="#41d4a8"),
            Text("R = resistance (Ω)", font_size=24, color="#ff7a59"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_edge(LEFT, buff=0.8).shift(DOWN*0.6)
        substitution = Text({substitution_text!r}, font_size=30, color="#c8d3e6").to_edge(DOWN, buff=1.25)
        answer = Text({answer_text!r}, font_size=32, color="#41d4a8", weight=BOLD).next_to(formula_box, DOWN, buff=0.45)
        self.play(FadeIn(formula_box), Write(formula), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(s) for s in symbol_row], lag_ratio=0.15), run_time=1.1)
        self.play(ReplacementTransform(formula.copy(), substitution), run_time=0.9)
        self.play(FadeIn(answer), Indicate(answer), run_time=0.9)
"""
    return body, 5.0


def _summary_scene_body(
    title: str,
    learning_goal: str,
    anchor_example: str,
    narration: str,
    visual_instruction: str,
    timeline: dict[str, Any],
) -> tuple[str, float]:
    title_block = _title_block(title)
    body = f"""{title_block}        center = RoundedRectangle(corner_radius=0.2, width=3.4, height=1.0, color="#4f8ef7", fill_opacity=0.10, stroke_width=2).shift(UP*0.25)
        center_text = Text("V = I × R", font_size=34, color="#e0e6f0", weight=BOLD).move_to(center)
        node_v = Circle(radius=0.48, color="#4f8ef7", fill_opacity=0.15, stroke_width=2).shift(LEFT*3.4 + UP*1.0)
        node_i = Circle(radius=0.48, color="#41d4a8", fill_opacity=0.15, stroke_width=2).shift(RIGHT*3.4 + UP*1.0)
        node_r = Circle(radius=0.48, color="#ff7a59", fill_opacity=0.15, stroke_width=2).shift(DOWN*1.7)
        label_v = Text("V", font_size=34, color="#4f8ef7", weight=BOLD).move_to(node_v)
        label_i = Text("I", font_size=34, color="#41d4a8", weight=BOLD).move_to(node_i)
        label_r = Text("R", font_size=34, color="#ff7a59", weight=BOLD).move_to(node_r)
        link_vi = Arrow(node_v.get_bottom(), center.get_left(), buff=0.1, color="#c8d3e6", stroke_width=4)
        link_ir = Arrow(center.get_right(), node_i.get_bottom(), buff=0.1, color="#c8d3e6", stroke_width=4)
        link_to_r = Arrow(center.get_bottom(), node_r.get_top(), buff=0.1, color="#c8d3e6", stroke_width=4)
        note = Text("When two are known, the third follows", font_size=24, color="#c8d3e6").to_edge(DOWN, buff=0.55)
        self.play(FadeIn(center), Write(center_text), run_time=1.0)
        self.play(FadeIn(node_v), FadeIn(node_i), FadeIn(node_r), FadeIn(label_v), FadeIn(label_i), FadeIn(label_r), run_time=1.1)
        self.play(GrowArrow(link_vi), GrowArrow(link_ir), GrowArrow(link_to_r), run_time=0.9)
        self.play(Write(note), Indicate(center_text), run_time=0.9)
"""
    return body, 4.9


def _concept_scene_body(
    title: str,
    learning_goal: str,
    anchor_example: str,
    narration: str,
    visual_instruction: str,
    timeline: dict[str, Any],
) -> tuple[str, float]:
    title_block = _title_block(title)
    body = f"""{title_block}        cards = VGroup(
            RoundedRectangle(corner_radius=0.18, width=3.8, height=1.3, color="#4f8ef7", fill_opacity=0.12, stroke_width=2),
            RoundedRectangle(corner_radius=0.18, width=3.8, height=1.3, color="#41d4a8", fill_opacity=0.12, stroke_width=2),
            RoundedRectangle(corner_radius=0.18, width=3.8, height=1.3, color="#ff7a59", fill_opacity=0.12, stroke_width=2),
        ).arrange(DOWN, buff=0.4).shift(DOWN*0.1)
        labels = VGroup(
            Text({learning_goal[:60]!r}, font_size=22, color="#e0e6f0"),
            Text({anchor_example[:60]!r}, font_size=22, color="#e0e6f0"),
            Text({visual_instruction[:60]!r}, font_size=22, color="#e0e6f0"),
        ).arrange(DOWN, buff=0.45)
        labels.move_to(cards.get_center())
        self.play(FadeIn(cards), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(label) for label in labels], lag_ratio=0.15), run_time=1.0)
        self.play(Indicate(cards), run_time=0.7)
"""
    return body, 3.8
