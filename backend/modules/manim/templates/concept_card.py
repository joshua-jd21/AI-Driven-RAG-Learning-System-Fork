# modules/manim/templates/concept_card.py
from manim import *
import numpy as np
from ..style_config import *
from .segment_timing import (
    segment_actions,
    segment_at,
    segment_duration,
    segment_rt,
    segments_from_timeline,
)

_DEFAULT_COLORS = [CHALK_BLUE, CHALK_GREEN, CHALK_YELLOW, CHALK_PINK]


class ConceptCardScene(Scene):
    """Shows a concept broken into labeled sub-cards."""

    def setup_background(self):
        bg = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color=SLATE_BG,
            fill_opacity=1,
            stroke_width=0,
        )
        dots = VGroup(*[
            Dot(
                point=[np.random.uniform(-7, 7), np.random.uniform(-4, 4), 0],
                radius=0.02,
                color=CHALK_WHITE,
                fill_opacity=np.random.uniform(0.1, 0.3),
            )
            for _ in range(40)
        ])
        self.add(bg, dots)

    def make_card(self, title, content, accent_color, width=3.2, height=2.8):
        inner_w = width - 0.5
        card_bg = RoundedRectangle(
            corner_radius=0.3,
            width=width,
            height=height,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=accent_color,
            stroke_width=1.5,
        )
        card_title = wrapped_text(
            str(title)[:50],
            font_size=24,
            max_w=inner_w,
            color=accent_color,
            font=TITLE_FONT,
            weight=BOLD,
        ).move_to(card_bg.get_top() + DOWN * 0.45)
        card_text = wrapped_text(
            str(content)[:200],
            font_size=15,
            max_w=inner_w,
            color=CHALK_WHITE,
            line_spacing=1.25,
        ).move_to(card_bg.get_center() + DOWN * 0.2)
        fit_in_box(VGroup(card_title, card_text), inner_w, height - 0.9)
        return VGroup(card_bg, card_title, card_text)

    def build_scene(
        self,
        main_title: str,
        cards: list,
        audio_duration: float = 0.0,
        timeline: dict | None = None,
    ):
        self.setup_background()
        if not cards:
            cards = [
                {"title": "Part 1", "content": main_title, "color": CHALK_BLUE},
                {"title": "Part 2", "content": "Key idea", "color": CHALK_GREEN},
            ]

        outer_box = RoundedRectangle(
            corner_radius=0.4,
            width=SAFE_W,
            height=SAFE_H - 0.2,
            fill_opacity=0,
            stroke_color=CARD_BORDER,
            stroke_width=1,
        )
        title = chalk_title(str(main_title)[:80])
        fit_title(title, SAFE_W - 0.6)
        title.move_to(np.array([0, TITLE_BAND_Y, 0]))
        segments = segments_from_timeline(timeline)
        title_seg = segment_at(timeline, 0)
        title_rt = segment_rt(title_seg, default=1.0, floor=0.45, cap=1.0) if segments else 1.0
        self.play(DrawBorderThenFill(outer_box), Write(title), run_time=title_rt, rate_func=smooth)

        normalized = []
        for i, c in enumerate(cards[:4]):
            if isinstance(c, dict):
                normalized.append({
                    "title": c.get("title", f"Part {i + 1}"),
                    "content": c.get("content", ""),
                    "color": c.get("color", _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]),
                })
            else:
                normalized.append({
                    "title": f"Part {i + 1}",
                    "content": str(c),
                    "color": _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                })

        n = len(normalized)
        buff = 0.35
        card_w = min(3.4, (SAFE_W - buff * (n - 1)) / max(n, 1))
        card_h = min(2.8, SAFE_H - 2.2)

        card_group = VGroup(*[
            self.make_card(c["title"], c["content"], c["color"], width=card_w, height=card_h)
            for c in normalized
        ]).arrange(RIGHT, buff=buff)
        fit_in_box(card_group, SAFE_W - 0.5, SAFE_H - 2.0)
        card_group.move_to(np.array([0, CONTENT_CENTER_Y - 0.2, 0]))

        arrows = VGroup(*[
            Arrow(
                card_group[i].get_right(),
                card_group[i + 1].get_left(),
                color=CHALK_WHITE,
                stroke_width=2,
                buff=0.1,
            )
            for i in range(len(card_group) - 1)
        ])

        cursor = title_rt
        if segments:
            for i, card in enumerate(card_group):
                seg = segment_at(timeline, min(i + 1, len(segments) - 1)) or title_seg
                if seg is None:
                    seg = title_seg
                if seg is not None:
                    start = float(seg.get("start", cursor))
                    if start > cursor:
                        self.wait(start - cursor)
                        cursor = start
                rt = segment_rt(seg, default=min(1.0, 0.55 + 0.15 * i), floor=0.35, cap=1.1)
                self.play(FadeIn(card, shift=UP * 0.2), run_time=rt, rate_func=smooth)
                cursor += rt
                hold = max(0.0, segment_duration(seg) - rt)
                if hold > 0.05:
                    self.wait(hold)
                    cursor += hold
                if seg and any(a.lower() in {"highlight", "compare", "compare_and_contrast"} for a in segment_actions(seg)):
                    self.play(Indicate(card), run_time=min(0.5, max(0.3, rt * 0.5)))
                    cursor += min(0.5, max(0.3, rt * 0.5))
        else:
            self.play(
                LaggedStart(
                    *[FadeIn(card, shift=UP * 0.2) for card in card_group],
                    lag_ratio=0.15,
                ),
                run_time=min(2.0, 0.5 + 0.4 * n),
                rate_func=smooth,
            )
            cursor += min(2.0, 0.5 + 0.4 * n)

        if len(arrows) > 0:
            arrow_rt = segment_rt(segment_at(timeline, min(len(card_group), max(0, len(segments) - 1))) if segments else None, default=0.6, floor=0.35, cap=0.8)
            self.play(
                LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2),
                run_time=arrow_rt,
                rate_func=smooth,
            )
            cursor += arrow_rt

        tail = max(0.5, audio_duration - cursor - 0.4) if audio_duration > 0 else 1.5
        self.wait(tail)
