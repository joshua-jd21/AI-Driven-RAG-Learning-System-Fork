"""Shared helpers for concept template code-generators."""
from __future__ import annotations

from typing import Any

from modules.manim.templates.segment_timing import (
    segment_at,
    segment_duration,
    segment_end,
    segment_hold,
    segment_rt,
    segment_start as segment_window_start,
    segments_from_timeline,
)

_HEADER = """\
from manim import *
import numpy as np
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from modules.manim.templates.chalkboard_scene import ChalkboardScene


class GeneratedScene(ChalkboardScene):
    def construct(self):

        self.setup_chalkboard()
"""

_FOOTER = """\
        self.play(FadeOut(*self.mobjects), run_time=0.40)
"""

# ------------------------------------------------------------------
# Theme Colors
# ------------------------------------------------------------------

BG = "#0f1117"

TITLE_COLOR = "#e0e6f0"
TEXT_COLOR = "#c8d3e6"

ACCENT1 = "#4f8ef7"
ACCENT2 = "#41d4a8"

FORCE_COLOR = "#ff7a59"
GROUND_COLOR = "#909090"
ICE_COLOR = "#a8d8ea"
VEL_COLOR = "#4fc3f7"


# ------------------------------------------------------------------
# Timeline Helpers
# ------------------------------------------------------------------

def get_event(
    timeline: dict[str, Any],
    event_id: str,
) -> dict[str, Any] | None:
    """Look up event by ID."""
    for ev in timeline.get("events", []):
        if ev["id"] == event_id:
            return ev
    return None


def narration_segments(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    return segments_from_timeline(timeline)


def get_event_by_type(
    timeline: dict[str, Any],
    plan_events: list[dict[str, Any]],
    event_type: str,
    fallback_id: str | None = None,
) -> dict[str, Any] | None:
    """
    Find first timeline event corresponding
    to semantic event type.
    """

    for plan_ev in plan_events:
        if plan_ev.get("type") == event_type:
            found = get_event(timeline, plan_ev["id"])
            if found:
                return found

    if fallback_id:
        return get_event(timeline, fallback_id)

    return None


def event_rt(
    timeline: dict[str, Any],
    event_id: str,
    default: float = 0.7,
) -> float:
    """
    Get run_time for event.
    """

    ev = get_event(timeline, event_id)
    if ev is None:
        return default

    seg = _segment_for_event(timeline, event_id)
    if seg is not None:
        return segment_rt(seg, default=default, floor=0.35, cap=default)

    rt = float(ev.get("run_time", default))
    return rt if rt >= 0.1 else default


def event_rt_type(
    timeline: dict[str, Any],
    plan_events: list[dict[str, Any]],
    event_type: str,
    fallback_id: str | None = None,
    default: float = 0.7,
) -> float:
    """
    Get run_time by semantic event type.
    """

    ev = get_event_by_type(
        timeline,
        plan_events,
        event_type,
        fallback_id,
    )

    if ev is None:
        return default

    seg = _segment_for_plan_event(timeline, plan_events, event_type, fallback_id)
    if seg is not None:
        return segment_rt(seg, default=default, floor=0.35, cap=default)

    rt = float(ev.get("run_time", default))
    return rt if rt >= 0.1 else default


def event_hold(
    timeline: dict[str, Any],
    event_id: str,
    default: float = 0.0,
) -> float:
    """
    Get hold_after.
    """

    ev = get_event(timeline, event_id)
    if ev is None:
        return default

    seg = _segment_for_event(timeline, event_id)
    if seg is not None:
        return segment_hold(seg, event_rt(timeline, event_id, default), minimum=default)

    return float(ev.get("hold_after", default))


def event_hold_type(
    timeline: dict[str, Any],
    plan_events: list[dict[str, Any]],
    event_type: str = "hold",
    default: float = 1.2,
) -> float:
    """
    Get hold duration using event type.
    """

    ev = get_event_by_type(
        timeline,
        plan_events,
        event_type,
    )

    if ev is None:
        return default

    seg = _segment_for_plan_event(timeline, plan_events, event_type)
    if seg is not None:
        return segment_hold(seg, event_rt_type(timeline, plan_events, event_type, default=default), minimum=default)

    hold = float(ev.get("hold_after", 0.0))
    return hold if hold >= 0.3 else default


def event_start(
    timeline: dict[str, Any],
    event_id: str,
    default: float = 0.0,
) -> float:
    """
    Get start time.
    """

    ev = get_event(timeline, event_id)
    if ev is None:
        return default

    seg = _segment_for_event(timeline, event_id)
    if seg is not None:
        return segment_window_start(seg, default=default)

    return float(ev.get("start", default))


# ------------------------------------------------------------------
# Scene Assembly Helpers
# ------------------------------------------------------------------

def build_sequential(
    blocks: list[tuple[str, float]],
    audio_duration: float,
    outro_time: float = 0.40,
) -> str:
    """
    Concatenate animation blocks and
    automatically pad remaining audio time.
    """

    lines: list[str] = []

    elapsed = 0.0

    for code, duration in blocks:
        lines.append(code)
        elapsed += duration

    tail = audio_duration - elapsed - outro_time

    if tail > 0.05:
        lines.append(
            f"        self.wait({tail:.3f})\n"
        )

    return "".join(lines)


def _segment_for_event(
    timeline: dict[str, Any],
    event_id: str,
) -> dict[str, Any] | None:
    events = list(timeline.get("events", []))
    segments = narration_segments(timeline)
    if not events or not segments:
        return None

    for idx, ev in enumerate(events):
        if ev.get("id") == event_id:
            if idx < len(segments):
                return segments[idx]
            return segments[-1]
    return None


def _segment_for_plan_event(
    timeline: dict[str, Any],
    plan_events: list[dict[str, Any]],
    event_type: str,
    fallback_id: str | None = None,
) -> dict[str, Any] | None:
    segments = narration_segments(timeline)
    if not segments:
        return None

    if plan_events:
        for idx, plan_ev in enumerate(plan_events):
            if plan_ev.get("type") != event_type:
                continue
            if idx < len(segments):
                return segments[idx]
            return segments[-1]

    if fallback_id is not None:
        return _segment_for_event(timeline, fallback_id)
    return None


def indent(
    code: str,
    spaces: int = 8,
) -> str:
    """
    Indent generated code block.
    """

    pad = " " * spaces

    result = []

    for line in code.splitlines():
        if line.strip():
            result.append(pad + line)
        else:
            result.append("")

    return "\n".join(result)


# ------------------------------------------------------------------
# Asset Helpers
# ------------------------------------------------------------------

def asset_param(
    plan: dict[str, Any],
    role: str,
    key: str,
    default: Any = "",
) -> Any:
    """
    Get asset parameter by role.
    """

    for asset in plan.get("assets", []):

        if asset["role"] == role:

            return asset.get(
                "params",
                {},
            ).get(
                key,
                default,
            )

    return default


def narration_segments(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the higher-level narration segments when available."""
    segments = timeline.get("segments", [])
    return list(segments) if isinstance(segments, list) else []


def segment_start(
    timeline: dict[str, Any],
    index: int,
    default: float = 0.0,
) -> float:
    segments = narration_segments(timeline)
    if index < len(segments):
        return float(segments[index].get("start", default))
    return default


def segment_end(
    timeline: dict[str, Any],
    index: int,
    default: float = 0.0,
) -> float:
    segments = narration_segments(timeline)
    if index < len(segments):
        return float(segments[index].get("end", default))
    return default


def asset_instance(
    plan: dict[str, Any],
    role: str,
) -> str | None:
    """
    Get instance ID for asset role.
    """

    for asset in plan.get("assets", []):

        if asset["role"] == role:

            return asset.get("instance_id")

    return None
