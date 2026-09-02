"""Intro template: title card with concept text and key term highlight."""
from __future__ import annotations

from typing import Any

from modules.templates.mechanics._base import (
    _HEADER,
    _FOOTER,
    TITLE_COLOR,
    TEXT_COLOR,
    ACCENT1,
    event_rt,
    event_hold,
    get_event_by_type,
)
from modules.manim.templates.segment_timing import (
    bounded_segment_budget,
    segment_stage_groups,
    segments_from_timeline,
)


class IntroTemplate:
    """Text-only intro scene: large title, subtitle, highlighted key term."""

    ALLOWED_EVENTS = {"place_title", "place_subtitle", "highlight_term", "hold"}
    SLOTS = {}  # no physics assets

    @staticmethod
    def compile(plan: dict[str, Any], timeline: dict[str, Any]) -> str:
        title_text = plan.get("title", plan.get("anchor_example", "Concept"))
        subtitle = plan.get("subtitle", "")
        key_term = plan.get("key_term", "")
        audio_dur = float(timeline.get("audio_duration", 8.0))

        plan_events = plan.get("events", [])
        title_event = get_event_by_type(timeline, plan_events, "place_title", "e0")
        subtitle_event = get_event_by_type(timeline, plan_events, "place_subtitle", "e1")
        term_event = get_event_by_type(timeline, plan_events, "highlight_term", "e2")

        lines: list[str] = [_HEADER]

        # Objects
        lines.append(f'        title = Text("{_esc(title_text)}", font_size=52, weight=BOLD, color="{TITLE_COLOR}")')
        lines.append(f'        title.to_edge(UP, buff=0.8)')

        if subtitle:
            lines.append(f'        subtitle = Text("{_esc(subtitle)}", font_size=30, color="{TEXT_COLOR}")')
            lines.append(f'        subtitle.next_to(title, DOWN, buff=0.5)')

        if key_term:
            lines.append(f'        key_term_box = RoundedRectangle(corner_radius=0.2, width=4.5, height=1.0,'
                         f' color="{ACCENT1}", fill_opacity=0.12, stroke_width=2)')
            lines.append(f'        key_term_text = Text("{_esc(key_term)}", font_size=36, color="{ACCENT1}", weight=BOLD)')
            lines.append(f'        key_term_box.move_to(ORIGIN + DOWN*0.4)')
            lines.append(f'        key_term_text.move_to(key_term_box.get_center())')

        lines.append("")

        # Each stage is assigned to an ordered segment. If stages share a
        # segment, bounded budgets group them inside that interval instead of
        # replaying the interval's full hold duration.
        title_rt = max(event_rt(timeline, title_event["id"], 0.8), 0.1) if title_event else 0.8
        subtitle_rt = max(event_rt(timeline, subtitle_event["id"], 0.7), 0.1) if subtitle_event else 0.7
        term_rt = max(event_rt(timeline, term_event["id"], 0.6), 0.1) if term_event else 0.6
        stages: list[tuple[str, float]] = [("title", title_rt)]
        if subtitle:
            stages.append(("subtitle", subtitle_rt))
        if key_term:
            stages.append(("term", term_rt))

        segments = segments_from_timeline(timeline)
        cursor = 0.0
        if segments:
            content_end = max(0.0, audio_dur - 0.40)
            groups = segment_stage_groups(len(segments), len(stages))
            for segment_index, stage_indices in enumerate(groups):
                if not stage_indices:
                    continue
                segment = segments[segment_index]
                first_budget = bounded_segment_budget(
                    segment, cursor, 0.0, scene_end=content_end
                )
                if first_budget["start"] > cursor:
                    lines.append(f'        self.wait({first_budget["start"] - cursor:.3f})')
                    cursor = first_budget["start"]

                for stage_index in stage_indices:
                    stage_name, requested = stages[stage_index]
                    budget = bounded_segment_budget(
                        segment, cursor, requested, scene_end=content_end
                    )
                    if budget["start"] > cursor:
                        lines.append(f'        self.wait({budget["start"] - cursor:.3f})')
                        cursor = budget["start"]
                    if budget["runtime"] <= 0.05:
                        continue
                    runtime = budget["runtime"]
                    if stage_name == "title":
                        code = f'        self.play(Write(title), run_time={runtime:.3f})'
                    elif stage_name == "subtitle":
                        code = f'        self.play(FadeIn(subtitle, shift=UP*0.2), run_time={runtime:.3f})'
                    else:
                        code = f'        self.play(FadeIn(key_term_box), Write(key_term_text), run_time={runtime:.3f})'
                    lines.append(code)
                    cursor = budget["start"] + runtime

                if cursor < first_budget["end"]:
                    lines.append(f'        self.wait({first_budget["end"] - cursor:.3f})')
                    cursor = first_budget["end"]
        else:
            # Preserve the legacy event timing path when no aligned segments
            # are available, while retaining the immediate title frame.
            lines.append(f'        self.play(Write(title), run_time={title_rt:.3f})')
            cursor += title_rt
            if title_event:
                title_hold = max(event_hold(timeline, title_event["id"], 0.4), 0.0)
                if title_hold > 0.05:
                    lines.append(f'        self.wait({title_hold:.3f})')
                    cursor += title_hold

            if subtitle and subtitle_event is not None:
                subtitle_start = float(subtitle_event.get("start", cursor))
                if subtitle_start > cursor:
                    lines.append(f'        self.wait({subtitle_start - cursor:.3f})')
                    cursor = subtitle_start
                lines.append(f'        self.play(FadeIn(subtitle, shift=UP*0.2), run_time={subtitle_rt:.3f})')
                cursor += subtitle_rt

            if key_term and term_event is not None:
                term_start = float(term_event.get("start", cursor))
                if term_start > cursor:
                    lines.append(f'        self.wait({term_start - cursor:.3f})')
                    cursor = term_start
                lines.append(f'        self.play(FadeIn(key_term_box), Write(key_term_text), run_time={term_rt:.3f})')
                cursor += term_rt

        tail = audio_dur - cursor - 0.40
        if tail > 0.05:
            lines.append(f'        self.wait({tail:.3f})')

        lines.append("")
        lines.append(_FOOTER)
        return "\n".join(lines)


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
