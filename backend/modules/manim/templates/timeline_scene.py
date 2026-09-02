# modules/manim/templates/timeline_scene.py

from manim import *
import numpy as np
from ..style_config import *
from .segment_timing import (
    segment_at,
    segment_duration,
    segment_rt,
    segment_text,
    segments_from_timeline,
)


class TimelineScene(Scene):

    def build_scene(
        self,
        title_text: str,
        events: list,
        audio_duration: float = 0.0,
        timeline: dict | None = None,
    ):
        self.camera.background_color = SLATE_BG
        timeline_data = timeline
        segments = segments_from_timeline(timeline_data)
        event_labels = []
        if segments:
            for seg in segments[:6]:
                label = segment_text(seg) or str(seg.get("visual_goal", "")).strip()
                event_labels.append((label or "Step").split(".")[0][:40])
        else:
            event_labels = [
                str(e.get("label", e) if isinstance(e, dict) else e)[:40]
                for e in (events or [])
            ]
        if not event_labels:
            event_labels = ["Step 1", "Step 2", "Step 3"]
        event_labels = event_labels[:6]

        title = Text(
            str(title_text)[:80],
            font=TITLE_FONT,
            font_size=34,
            color=CHALK_WHITE,
            weight=BOLD,
        )
        fit_title(title, SAFE_W - 0.6)
        title.move_to(np.array([0, TITLE_BAND_Y, 0]))
        title_rt = segment_rt(segment_at(timeline_data, 0), default=0.8, floor=0.45, cap=0.9) if segments else 0.8
        self.play(Write(title), run_time=title_rt, rate_func=smooth)

        n = len(event_labels)
        span = min(SAFE_W - 1.0, max(5.5, n * 1.8))
        timeline_line = Line(LEFT * span / 2, RIGHT * span / 2, color=CHALK_WHITE, stroke_width=2)
        timeline_line.move_to(np.array([0, CONTENT_CENTER_Y - 0.4, 0]))
        line_rt = segment_rt(segment_at(timeline_data, 1), default=0.6, floor=0.35, cap=0.7) if segments else 0.6
        self.play(Create(timeline_line), run_time=line_rt, rate_func=smooth)

        xs = list(np.linspace(-span / 2 + 0.4, span / 2 - 0.4, n)) if n > 1 else [0.0]
        line_y = timeline_line.get_center()[1]
        dots_and_labels = []
        for i, (event, x) in enumerate(zip(event_labels, xs)):
            dot = Dot(point=np.array([x, line_y, 0]), color=CHALK_YELLOW, radius=0.08)
            label = wrapped_text(
                event,
                font_size=15,
                max_w=1.8,
                color=CHALK_WHITE,
            )
            direction = UP if i % 2 == 0 else DOWN
            label.next_to(dot, direction, buff=0.22)
            dots_and_labels.append(VGroup(dot, label))

        timeline_group = VGroup(timeline_line, *dots_and_labels)
        fit_in_box(timeline_group, SAFE_W, SAFE_H - 1.8)
        timeline_group.move_to(np.array([0, CONTENT_CENTER_Y - 0.35, 0]))

        cursor = title_rt + line_rt
        if segments:
            for i, group in enumerate(dots_and_labels):
                seg = segment_at(timeline_data, min(i + 1, len(segments) - 1))
                if seg is not None:
                    start = float(seg.get("start", cursor))
                    if start > cursor:
                        self.wait(start - cursor)
                        cursor = start
                rt = segment_rt(seg, default=0.55, floor=0.3, cap=0.8)
                self.play(FadeIn(group, scale=0.9), run_time=rt, rate_func=smooth)
                cursor += rt
                hold = max(0.0, segment_duration(seg) - rt)
                if hold > 0.05:
                    self.wait(hold)
                    cursor += hold
        else:
            self.play(
                LaggedStart(
                    *[FadeIn(g, scale=0.9) for g in dots_and_labels],
                    lag_ratio=0.18,
                ),
                run_time=min(2.2, 0.5 + 0.35 * n),
                rate_func=smooth,
            )
            cursor += min(2.2, 0.5 + 0.35 * n)

        tail = max(0.5, audio_duration - cursor - 0.4) if audio_duration > 0 else 1.5
        self.wait(tail)
