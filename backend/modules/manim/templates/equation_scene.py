# modules/manim/templates/equation_scene.py

from manim import *
import re
from typing import Any
from modules.manim.code_sanitize import latex_to_plain
from ..style_config import *
from .segment_timing import (
    bounded_segment_budget,
    segment_actions,
    segment_at,
    segment_rt,
    segment_stage_groups,
    segments_from_timeline,
)


class EquationScene(Scene):

    def _sorted_events(self, timeline: dict[str, Any] | None) -> list[dict[str, Any]]:
        events = list((timeline or {}).get("events", []))
        return sorted(events, key=lambda ev: float(ev.get("start", 0.0)))

    def _event_start(
        self,
        events: list[dict[str, Any]],
        index: int,
        fallback: float,
    ) -> float:
        if index < len(events):
            return float(events[index].get("start", fallback))
        return fallback

    def _event_rt(
        self,
        events: list[dict[str, Any]],
        index: int,
        fallback: float,
    ) -> float:
        if index < len(events):
            rt = float(events[index].get("run_time", fallback))
            return rt if rt >= 0.1 else fallback
        return fallback

    def _make_equation_mobject(self, equation_text: str):
        """Render equations with math objects first, then a safe plain fallback."""
        eq_str = str(equation_text).strip() or r"E = mc^2"
        try:
            return MathTex(eq_str, color=CHALK_YELLOW).scale(1.3)
        except Exception:
            try:
                return Tex(eq_str, color=CHALK_YELLOW).scale(1.3)
            except Exception:
                return Text(
                    latex_to_plain(eq_str),
                    font=MONO_FONT,
                    font_size=34,
                color=CHALK_YELLOW,
            )

    def _equation_states(self, equation_text: str) -> list[str]:
        raw = str(equation_text).strip() or r"E = mc^2"
        tokens = [tok for tok in re.split(r"\s+", raw) if tok]
        if len(tokens) <= 1:
            return [raw]
        states = []
        for i in range(1, len(tokens) + 1):
            states.append(" ".join(tokens[:i]))
        return states

    def build_scene(
        self,
        title_text: str,
        equation_text: str,
        explanation: str = "",
        audio_duration: float = 0.0,
        timeline: dict[str, Any] | None = None,
    ):
        self.camera.background_color = SLATE_BG
        events = self._sorted_events(timeline)
        segments = segments_from_timeline(timeline)
        cursor = 0.0

        title = Text(
            str(title_text)[:80],
            font=TITLE_FONT,
            font_size=36,
            color=CHALK_WHITE,
            weight=BOLD,
        )
        fit_title(title, SAFE_W - 0.6)
        title.move_to(np.array([0, TITLE_BAND_Y, 0]))

        equation = self._make_equation_mobject(equation_text)
        fit_width(equation, SAFE_W - 1.5)

        equation.move_to(np.array([0, CONTENT_CENTER_Y + 0.3, 0]))

        explanation_text = None
        if explanation:
            explanation_text = wrapped_text(
                str(explanation)[:240],
                font_size=20,
                max_w=SAFE_W - 1.2,
                color=CHALK_WHITE,
                line_spacing=1.3,
            )
            explanation_text.next_to(equation, DOWN, buff=0.55)
            clamp_into_frame(explanation_text)

        title_rt = segment_rt(segment_at(timeline, 0), default=self._event_rt(events, 0, 0.8), floor=0.45, cap=0.9) if segments else self._event_rt(events, 0, 0.8)
        self.play(Write(title), run_time=title_rt, rate_func=smooth)
        cursor += title_rt

        if segments:
            states = self._equation_states(equation_text)
            current = self._make_equation_mobject(states[0])
            fit_width(current, SAFE_W - 1.5)
            current.move_to(np.array([0, CONTENT_CENTER_Y + 0.3, 0]))
            stages: list[dict[str, Any]] = [
                {
                    "kind": "reveal_equation",
                    "runtime": segment_rt(segment_at(timeline, 1), default=0.85, floor=0.35, cap=0.9),
                    "mob": current,
                }
            ]
            previous = current
            for state_index, state in enumerate(states[1:], start=1):
                next_state = self._make_equation_mobject(state)
                fit_width(next_state, SAFE_W - 1.5)
                next_state.move_to(np.array([0, CONTENT_CENTER_Y + 0.3, 0]))
                seg = segment_at(timeline, min(state_index + 1, len(segments) - 1))
                stages.append({
                    "kind": "transform",
                    "runtime": segment_rt(seg, default=0.75, floor=0.3, cap=1.0),
                    "mob": next_state,
                    "previous": previous,
                    "emphasize": bool(
                        seg and any(
                            action.lower() in {"highlight", "transform", "circumscribe"}
                            for action in segment_actions(seg)
                        )
                    ),
                })
                previous = next_state

            if explanation_text is not None:
                explanation_seg = segment_at(timeline, min(len(stages), len(segments) - 1))
                stages.append({
                    "kind": "explanation",
                    "runtime": segment_rt(explanation_seg, default=0.7, floor=0.3, cap=0.8),
                    "mob": explanation_text,
                })

            content_end = max(0.0, audio_duration - 0.40)
            for segment_index, stage_indices in enumerate(
                segment_stage_groups(len(segments), len(stages))
            ):
                if not stage_indices:
                    continue
                segment = segments[segment_index]
                window = bounded_segment_budget(segment, cursor, 0.0, scene_end=content_end)
                if window["start"] > cursor:
                    self.wait(window["start"] - cursor)
                    cursor = window["start"]

                for stage_index in stage_indices:
                    stage = stages[stage_index]
                    budget = bounded_segment_budget(
                        segment,
                        cursor,
                        float(stage["runtime"]),
                        scene_end=content_end,
                    )
                    if budget["start"] > cursor:
                        self.wait(budget["start"] - cursor)
                        cursor = budget["start"]
                    runtime = budget["runtime"]
                    if runtime <= 0.05:
                        continue
                    if stage["kind"] == "reveal_equation":
                        self.play(FadeIn(stage["mob"]), run_time=runtime, rate_func=smooth)
                    elif stage["kind"] == "transform":
                        self.play(
                            ReplacementTransform(stage["previous"], stage["mob"]),
                            run_time=runtime,
                            rate_func=smooth,
                        )
                        remaining = max(0.0, budget["end"] - budget["start"] - runtime)
                        if stage.get("emphasize") and remaining > 0.05:
                            emphasis_runtime = min(0.45, max(0.25, runtime * 0.4), remaining)
                            self.play(Indicate(stage["mob"]), run_time=emphasis_runtime)
                            runtime += emphasis_runtime
                    else:
                        self.play(FadeIn(stage["mob"]), run_time=runtime, rate_func=smooth)
                    cursor = budget["start"] + runtime

                if cursor < window["end"]:
                    self.wait(window["end"] - cursor)
                    cursor = window["end"]
        else:
            equation_start = self._event_start(events, 1, cursor)
            if equation_start > cursor:
                self.wait(equation_start - cursor)
                cursor = equation_start

            equation_rt = self._event_rt(events, 1, 1.0)
            self.play(Write(equation), run_time=equation_rt, rate_func=smooth)
            cursor += equation_rt

            if explanation_text is not None:
                explanation_start = self._event_start(events, 2, cursor)
                if explanation_start > cursor:
                    self.wait(explanation_start - cursor)
                    cursor = explanation_start
                explanation_rt = self._event_rt(events, 2, 0.7)
                self.play(FadeIn(explanation_text), run_time=explanation_rt, rate_func=smooth)
                cursor += explanation_rt

        tail = (
            max(0.0, audio_duration - cursor - 0.40)
            if segments
            else (max(0.5, audio_duration - cursor - 0.40) if audio_duration > 0 else 1.5)
        )
        if tail > 0.05:
            self.wait(tail)
