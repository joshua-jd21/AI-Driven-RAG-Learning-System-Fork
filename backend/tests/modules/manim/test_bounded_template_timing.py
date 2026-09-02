from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import modules.manim.templates.diagram_scene as diagram_scene
import modules.manim.templates.equation_scene as equation_scene
from modules.templates.mechanics.intro import IntroTemplate


class _FakeMobject:
    def __init__(self, *children):
        self.children = list(children)

    def __iter__(self):
        return iter(self.children)

    def scale(self, _factor):
        return self

    def move_to(self, _point):
        return self

    def next_to(self, *_args, **_kwargs):
        return self

    def to_edge(self, *_args, **_kwargs):
        return self

    def set_fill(self, *_args, **_kwargs):
        return self

    def set_opacity(self, *_args, **_kwargs):
        return self

    def get_center(self):
        return np.zeros(3)

    def get_right(self):
        return np.array([1.0, 0.0, 0.0])

    def get_left(self):
        return np.array([-1.0, 0.0, 0.0])


def _fake_constructor(*_args, **_kwargs):
    return _FakeMobject(*_args)


class _EquationProbe(equation_scene.EquationScene):
    @property
    def camera(self):
        return self._camera


class _DiagramProbe(diagram_scene.DiagramScene):
    @property
    def camera(self):
        return self._camera


def _recording_scene(scene_cls):
    scene = object.__new__(scene_cls)
    calls: list[tuple[str, float]] = []
    scene._camera = SimpleNamespace(background_color=None)
    scene.play = lambda *_args, run_time=0.0, **_kwargs: calls.append(("play", float(run_time)))
    scene.wait = lambda duration=0.0: calls.append(("wait", float(duration)))
    return scene, calls


def _patch_common(monkeypatch, module) -> None:
    for name in (
        "Text", "MathTex", "Tex", "Circle", "Arrow", "VGroup",
        "wrapped_text", "FadeIn", "Write", "ReplacementTransform",
        "Indicate", "GrowArrow", "LaggedStart",
    ):
        monkeypatch.setattr(module, name, _fake_constructor)
    for name in ("fit_title", "fit_width", "fit_in_box", "clamp_into_frame"):
        monkeypatch.setattr(module, name, lambda *_args, **_kwargs: None)


def _total(calls: list[tuple[str, float]], outro: float = 0.4) -> float:
    return sum(duration for _kind, duration in calls) + outro


def test_intro_generated_timing_stays_inside_audio_budget() -> None:
    plan = {
        "title": "Concept",
        "subtitle": "A persistent state",
        "key_term": "State",
        "events": [
            {"id": "e0", "type": "place_title"},
            {"id": "e1", "type": "highlight_term"},
        ],
    }
    timeline = {
        "audio_duration": 10.0,
        "segments": [
            {"start": 0.0, "end": 3.0},
            {"start": 3.0, "end": 7.0},
            {"start": 7.0, "end": 10.0},
        ],
        "events": [],
    }

    code = IntroTemplate.compile(plan, timeline)
    assert code.count("self.play(") == 4
    durations = [
        float(line.split("run_time=")[1].split(")")[0])
        for line in code.splitlines()
        if "self.play(" in line
    ]
    waits = [
        float(line.split("self.wait(")[1].split(")")[0])
        for line in code.splitlines()
        if "self.wait(" in line
    ]
    assert sum(durations + waits) <= 10.0 + 1e-6


def test_equation_timing_bounds_grouped_actions(monkeypatch) -> None:
    _patch_common(monkeypatch, equation_scene)
    scene, calls = _recording_scene(_EquationProbe)
    timeline = {
        "audio_duration": 10.0,
        "segments": [
            {"start": 0.0, "end": 3.0},
            {"start": 3.0, "end": 6.0},
            {"start": 6.0, "end": 10.0},
        ],
        "events": [],
    }

    scene.build_scene("Equation", "x = y = z", "The relationship", 10.0, timeline)

    assert _total(calls) <= 10.0 + 1e-6


def test_diagram_timing_does_not_reuse_final_segment_budget(monkeypatch) -> None:
    _patch_common(monkeypatch, diagram_scene)
    scene, calls = _recording_scene(_DiagramProbe)
    timeline = {
        "audio_duration": 10.0,
        "segments": [
            {"start": 0.0, "end": 4.0, "actions": ["highlight"]},
            {"start": 4.0, "end": 10.0, "actions": ["compare"]},
        ],
        "events": [],
    }

    scene.build_scene(
        "Flow",
        ["A", "B", "C", "D", "E"],
        10.0,
        timeline,
        "A process caption",
    )

    assert _total(calls) <= 10.0 + 1e-6
    assert all(duration > 0.0 for kind, duration in calls if kind == "wait")
