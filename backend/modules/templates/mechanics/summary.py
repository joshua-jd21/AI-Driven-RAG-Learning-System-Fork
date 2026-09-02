"""Summary template: animated bullet-point list with key takeaways."""
from __future__ import annotations

from typing import Any

from modules.templates.mechanics._base import (
    _HEADER,
    _FOOTER,
    TITLE_COLOR,
    TEXT_COLOR,
    ACCENT1,
    ACCENT2,
    event_rt,
    event_hold,
    get_event_by_type,
)
from modules.manim.templates.segment_timing import (
    bounded_segment_budget,
    segment_at,
    segment_rt,
    segment_stage_groups,
    segments_from_timeline,
)


class SummaryTemplate:
    ALLOWED_EVENTS = {"place_title", "place_point", "hold"}
    SLOTS = {}

    @staticmethod
    def compile(plan: dict[str, Any], timeline: dict[str, Any]) -> str:
        audio_dur = float(timeline.get("audio_duration", 10.0))
        title_text = plan.get("title", "Summary")
        points = plan.get("summary_points", [
            "Objects at rest remain at rest",
            "Objects in motion remain in motion",
            "An external force is required for change",
        ])
        # Cap at 4 points for layout
        points = points[:4]

        plan_events = plan.get("events", [])
        title_event = get_event_by_type(timeline, plan_events, "place_title", "e0")
        point_event = get_event_by_type(timeline, plan_events, "place_point", "e1")
        hold_event = get_event_by_type(timeline, plan_events, "hold")

        timeline_events = sorted(
            list(timeline.get("events", [])),
            key=lambda ev: float(ev.get("start", 0.0)),
        )

        lines: list[str] = [_HEADER]

        lines += [
            f'        title = Text("{_esc(title_text)}", font_size=44, weight=BOLD, color="{TITLE_COLOR}")',
            f'        title.to_edge(UP, buff=0.4)',
            "",
        ]

        # Build point mobjects
        for i, pt in enumerate(points):
            bullet = "\u2022"
            lines += [
                f'        pt_{i} = Text("{bullet} {_esc(pt)}", font_size=28, color="{TEXT_COLOR}")',
                f'        pt_{i}.set_opacity(0)',
            ]

        # Arrange them vertically
        if points:
            first = "pt_0"
            lines += [f'        all_pts = VGroup({", ".join(f"pt_{i}" for i in range(len(points)))})']
            lines += [f'        all_pts.arrange(DOWN, aligned_edge=LEFT, buff=0.35)']
            lines += [f'        all_pts.next_to(title, DOWN, buff=0.5).shift(LEFT*0.5)']

        lines += [""]

        cursor = 0.0
        segments = segments_from_timeline(timeline)
        if segments:
            # Allocate each title/point stage inside one ordered narration
            # interval. Holds only fill the unused remainder of that interval.
            title_rt = max(event_rt(timeline, title_event["id"], 0.8), 0.1) if title_event else 0.8
            stages: list[tuple[str, float]] = [("title", title_rt)]
            stages.extend(("point", 0.7) for _ in points)
            content_end = max(0.0, audio_dur - 0.40)

            for segment_index, stage_indices in enumerate(
                segment_stage_groups(len(segments), len(stages))
            ):
                if not stage_indices:
                    continue
                segment = segments[segment_index]
                window = bounded_segment_budget(segment, cursor, 0.0, scene_end=content_end)
                if window["start"] > cursor:
                    lines.append(f'        self.wait({window["start"] - cursor:.3f})')
                    cursor = window["start"]

                for stage_index in stage_indices:
                    stage_name, requested = stages[stage_index]
                    runtime_request = segment_rt(
                        segment,
                        default=requested,
                        floor=0.3,
                        cap=requested,
                    )
                    budget = bounded_segment_budget(
                        segment,
                        cursor,
                        runtime_request,
                        scene_end=content_end,
                    )
                    if budget["start"] > cursor:
                        lines.append(f'        self.wait({budget["start"] - cursor:.3f})')
                        cursor = budget["start"]
                    runtime = budget["runtime"]
                    if runtime <= 0.05:
                        continue
                    if stage_name == "title":
                        lines.append(f'        self.play(Write(title), run_time={runtime:.3f})')
                    else:
                        point_index = stage_index - 1
                        lines.extend([
                            f'        pt_{point_index}.set_opacity(1)',
                            f'        self.play(FadeIn(pt_{point_index}, shift=RIGHT*0.3), run_time={runtime:.3f})',
                        ])
                    cursor = budget["start"] + runtime

                if cursor < window["end"]:
                    lines.append(f'        self.wait({window["end"] - cursor:.3f})')
                    cursor = window["end"]
        else:
            def _emit_stage(start: float, code: list[str], rt: float, hold: float = 0.0) -> None:
                nonlocal cursor
                if start > cursor:
                    lines.append(f'        self.wait({start - cursor:.3f})')
                    cursor = start
                lines.extend(code)
                cursor += rt
                if hold > 0.05:
                    lines.append(f'        self.wait({hold:.3f})')
                    cursor += hold

            title_rt = max(event_rt(timeline, title_event["id"], 0.8), 0.1) if title_event else 0.8
            title_hold = max(event_hold(timeline, title_event["id"], 0.3), 0.0) if title_event else 0.3
            _emit_stage(
                0.0,
                [f'        self.play(Write(title), run_time={title_rt:.3f})'],
                title_rt,
                title_hold,
            )

            for i in range(len(points)):
                if i == 0 and point_event is not None:
                    start = float(point_event.get("start", cursor))
                    rt_pt = max(event_rt(timeline, point_event["id"], 0.7), 0.1)
                    hold_pt = max(event_hold(timeline, point_event["id"], 0.3), 0.0)
                else:
                    extra_idx = i + 1 if point_event is not None else i
                    if extra_idx < len(timeline_events):
                        ev = timeline_events[extra_idx]
                        start = float(ev.get("start", cursor + 0.8))
                        rt_pt = max(event_rt(timeline, ev["id"], 0.7), 0.1)
                        hold_pt = max(event_hold(timeline, ev["id"], 0.3), 0.0)
                    else:
                        start = cursor + 0.8
                        rt_pt = 0.7
                        hold_pt = 0.0
                _emit_stage(
                    start,
                    [
                        f'        pt_{i}.set_opacity(1)',
                        f'        self.play(FadeIn(pt_{i}, shift=RIGHT*0.3), run_time={rt_pt:.3f})',
                    ],
                    rt_pt,
                    hold_pt,
                )

        tail = audio_dur - cursor - 0.40
        if tail > 0.05:
            lines += [f'        self.wait({tail:.3f})']

        lines += ["", _FOOTER]
        return "\n".join(lines)


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
