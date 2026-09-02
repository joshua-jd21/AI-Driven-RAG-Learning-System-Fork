# modules/manim/templates/diagram_scene.py

from manim import *
import numpy as np
from typing import Any
from ..style_config import *
from .segment_timing import (
    bounded_segment_budget,
    segment_actions,
    segment_at,
    segment_rt,
    segment_stage_groups,
    segments_from_timeline,
)


class DiagramScene(Scene):

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

    def _normalize_nodes(self, nodes: list) -> list[str]:
        labels: list[str] = []
        for node in nodes:
            if isinstance(node, dict):
                labels.append(str(node.get("label", node.get("name", "?"))))
            else:
                labels.append(str(node))
        return labels[:8] if labels else ["A", "B", "C"]

    def build_scene(
        self,
        title_text: str,
        nodes: list,
        audio_duration: float = 0.0,
        timeline: dict[str, Any] | None = None,
        caption_text: str = "",
    ):
        self.camera.background_color = SLATE_BG
        labels = self._normalize_nodes(nodes)
        events = self._sorted_events(timeline)
        segments = segments_from_timeline(timeline)
        cursor = 0.0

        title = Text(
            str(title_text)[:80],
            font=TITLE_FONT,
            font_size=34,
            color=CHALK_WHITE,
            weight=BOLD,
        )
        fit_title(title, SAFE_W - 0.6)
        title.move_to(np.array([0, TITLE_BAND_Y, 0]))

        title_rt = segment_rt(segment_at(timeline, 0), default=self._event_rt(events, 0, 0.8), floor=0.45, cap=0.9) if segments else self._event_rt(events, 0, 0.8)
        self.play(Write(title), run_time=title_rt, rate_func=smooth)
        cursor += title_rt

        n = len(labels)
        max_span = SAFE_W - 1.5
        if n <= 4:
            positions = [
                np.array([x, CONTENT_CENTER_Y - 0.2, 0])
                for x in np.linspace(-max_span / 2, max_span / 2, n)
            ]
        else:
            cols = min(4, n)
            rows = (n + cols - 1) // cols
            positions = []
            for i in range(n):
                row, col = divmod(i, cols)
                positions.append(
                    np.array([
                        -max_span / 2 + col * (max_span / max(cols - 1, 1)),
                        0.8 - row * 1.6,
                        0,
                    ])
                )

        objects = []
        for label, pos in zip(labels, positions):
            circle = Circle(radius=0.5, color=CHALK_BLUE, stroke_width=2)
            circle.set_fill(CHALK_BLUE, opacity=0.15)
            label_mob = wrapped_text(
                str(label)[:28],
                font_size=16,
                max_w=0.9,
                color=CHALK_WHITE,
            )
            group = VGroup(circle, label_mob)
            group.move_to(pos)
            objects.append(group)

        graph = VGroup(*objects)
        fit_in_box(graph, SAFE_W - 0.5, SAFE_H - 2.0)
        graph.move_to(np.array([0, CONTENT_CENTER_Y - 0.25, 0]))

        arrows = VGroup(*[
            Arrow(
                objects[i].get_right(),
                objects[i + 1].get_left(),
                color=CHALK_YELLOW,
                stroke_width=2,
                buff=0.12,
                max_tip_length_to_length_ratio=0.2,
            )
            for i in range(len(objects) - 1)
        ])

        caption = None
        if caption_text:
            caption = wrapped_text(
                str(caption_text)[:180],
                font_size=18,
                max_w=SAFE_W - 1.2,
                color=CHALK_WHITE,
                line_spacing=1.2,
            )
            caption.to_edge(DOWN, buff=0.45)
            clamp_into_frame(caption)

        if segments:
            stages: list[dict[str, Any]] = [
                {
                    "kind": "object",
                    "object": obj,
                    "runtime_default": min(1.0, 0.45 * (i + 1) + 0.35),
                    "object_index": i,
                }
                for i, obj in enumerate(objects)
            ]
            if len(objects) >= 2:
                stages.append({"kind": "arrows", "runtime_default": 0.8})
            if caption is not None:
                stages.append({"kind": "caption", "runtime_default": 0.7})

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
                    runtime_default = float(stage["runtime_default"])
                    runtime_request = segment_rt(
                        segment,
                        default=runtime_default,
                        floor=0.3,
                        cap=runtime_default,
                    )
                    budget = bounded_segment_budget(
                        segment,
                        cursor,
                        runtime_request,
                        scene_end=content_end,
                    )
                    if budget["start"] > cursor:
                        self.wait(budget["start"] - cursor)
                        cursor = budget["start"]
                    runtime = budget["runtime"]
                    if runtime <= 0.05:
                        continue

                    if stage["kind"] == "object":
                        self.play(
                            FadeIn(stage["object"], scale=0.85),
                            run_time=runtime,
                            rate_func=smooth,
                        )
                        remaining = max(0.0, budget["end"] - budget["start"] - runtime)
                        if any(
                            action.lower() in {"highlight", "compare", "indicate"}
                            for action in segment_actions(segment)
                        ) and remaining > 0.05:
                            emphasis_runtime = min(0.45, max(0.25, runtime * 0.4), remaining)
                            self.play(Indicate(stage["object"]), run_time=emphasis_runtime)
                            runtime += emphasis_runtime
                    elif stage["kind"] == "arrows":
                        self.play(
                            LaggedStart(*[GrowArrow(arrow) for arrow in arrows], lag_ratio=0.2),
                            run_time=runtime,
                            rate_func=smooth,
                        )
                    else:
                        self.play(FadeIn(caption), run_time=runtime, rate_func=smooth)
                    cursor = budget["start"] + runtime

                if cursor < window["end"]:
                    self.wait(window["end"] - cursor)
                    cursor = window["end"]
        else:
            graph_start = self._event_start(events, 1, cursor)
            if graph_start > cursor:
                self.wait(graph_start - cursor)
                cursor = graph_start

            graph_rt = self._event_rt(events, 1, min(2.5, 0.45 * n + 0.5))
            self.play(
                LaggedStart(*[FadeIn(o, scale=0.85) for o in graph], lag_ratio=0.2),
                run_time=graph_rt,
                rate_func=smooth,
            )
            cursor += graph_rt

        if not segments and len(objects) >= 2:
            arrows_start = self._event_start(events, 2, cursor)
            if arrows_start > cursor:
                self.wait(arrows_start - cursor)
                cursor = arrows_start
            arrows_rt = self._event_rt(events, 2, 0.8)
            self.play(
                LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2),
                run_time=arrows_rt,
                rate_func=smooth,
            )
            cursor += arrows_rt

        if not segments and caption is not None:
            caption_start = self._event_start(events, 3, cursor)
            if caption_start > cursor:
                self.wait(caption_start - cursor)
                cursor = caption_start

            caption_rt = self._event_rt(events, 3, 0.7)
            self.play(FadeIn(caption), run_time=caption_rt, rate_func=smooth)
            cursor += caption_rt

        tail = (
            max(0.0, audio_duration - cursor - 0.40)
            if segments
            else (max(0.5, audio_duration - cursor - 0.40) if audio_duration > 0 else 1.5)
        )
        if tail > 0.05:
            self.wait(tail)
