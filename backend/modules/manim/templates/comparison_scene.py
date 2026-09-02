# modules/manim/templates/comparison_scene.py

from manim import *
from ..style_config import *
from .segment_timing import (
    segment_actions,
    segment_at,
    segment_duration,
    segment_rt,
    segments_from_timeline,
)


class ComparisonScene(Scene):

    def build_scene(
        self,
        left_title: str,
        left_content: str,
        right_title: str,
        right_content: str,
        audio_duration: float = 0.0,
        timeline: dict | None = None,
    ):
        self.camera.background_color = SLATE_BG
        segments = segments_from_timeline(timeline)

        box_w = min(5.0, (SAFE_W - 1.2) / 2)
        box_h = min(4.0, SAFE_H - 2.0)
        inner_w = box_w - 0.6
        gap = 0.8
        shift_x = box_w / 2 + gap / 2

        left_box = RoundedRectangle(
            width=box_w,
            height=box_h,
            corner_radius=0.25,
            stroke_color=CHALK_BLUE,
            stroke_width=2,
            fill_color=CARD_BG,
            fill_opacity=0.9,
        ).shift(LEFT * shift_x)

        right_box = RoundedRectangle(
            width=box_w,
            height=box_h,
            corner_radius=0.25,
            stroke_color=CHALK_GREEN,
            stroke_width=2,
            fill_color=CARD_BG,
            fill_opacity=0.9,
        ).shift(RIGHT * shift_x)

        left_header = wrapped_text(
            str(left_title)[:50],
            font_size=24,
            max_w=inner_w,
            color=CHALK_BLUE,
            font=TITLE_FONT,
            weight=BOLD,
        ).move_to(left_box.get_top() + DOWN * 0.55)

        right_header = wrapped_text(
            str(right_title)[:50],
            font_size=24,
            max_w=inner_w,
            color=CHALK_GREEN,
            font=TITLE_FONT,
            weight=BOLD,
        ).move_to(right_box.get_top() + DOWN * 0.55)

        left_text = wrapped_text(
            str(left_content)[:280],
            font_size=17,
            max_w=inner_w,
            color=CHALK_WHITE,
            line_spacing=1.3,
        ).move_to(left_box.get_center() + DOWN * 0.15)

        right_text = wrapped_text(
            str(right_content)[:280],
            font_size=17,
            max_w=inner_w,
            color=CHALK_WHITE,
            line_spacing=1.3,
        ).move_to(right_box.get_center() + DOWN * 0.15)

        panel = VGroup(left_box, right_box, left_header, right_header, left_text, right_text)
        fit_in_box(panel, SAFE_W, SAFE_H - 0.5)
        panel.move_to(np.array([0, CONTENT_CENTER_Y - 0.1, 0]))

        cursor = 0.0
        if segments:
            stages = [
                (left_box, right_box),
                (left_header, right_header),
                (left_text, right_text),
            ]
            for i, stage in enumerate(stages):
                seg = segment_at(timeline, min(i, len(segments) - 1))
                if seg is not None:
                    start = float(seg.get("start", cursor))
                    if start > cursor:
                        self.wait(start - cursor)
                        cursor = start
                rt = segment_rt(seg, default=0.7 if i == 0 else 0.6, floor=0.35, cap=0.9)
                anims = [FadeIn(mob) for mob in stage]
                self.play(*anims, run_time=rt, rate_func=smooth)
                cursor += rt
                hold = max(0.0, segment_duration(seg) - rt)
                if hold > 0.05:
                    self.wait(hold)
                    cursor += hold
                if seg and any(a.lower() in {"highlight", "compare", "contrast"} for a in segment_actions(seg)):
                    emph_rt = min(0.45, max(0.25, rt * 0.4))
                    self.play(Indicate(left_box), Indicate(right_box), run_time=emph_rt)
                    cursor += emph_rt
            if len(segments) > 3:
                seg = segment_at(timeline, 3)
                if seg is not None:
                    rt = segment_rt(seg, default=0.45, floor=0.3, cap=0.7)
                    self.play(Indicate(left_box), Indicate(right_box), run_time=rt)
                    cursor += rt
        else:
            self.play(
                LaggedStart(Create(left_box), Create(right_box), lag_ratio=0.2),
                run_time=0.8,
                rate_func=smooth,
            )
            self.play(
                FadeIn(left_header), FadeIn(right_header),
                run_time=0.6,
                rate_func=smooth,
            )
            self.play(
                FadeIn(left_text), FadeIn(right_text),
                run_time=0.7,
                rate_func=smooth,
            )
            cursor = 2.1

        tail = max(0.5, audio_duration - cursor - 0.4) if audio_duration > 0 else 1.5
        self.wait(tail)
