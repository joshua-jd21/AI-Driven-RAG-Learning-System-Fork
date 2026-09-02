from __future__ import annotations

import modules.manim.templates.equation_scene as equation_scene


class _DummyMobject:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def scale(self, _factor: float):
        return self


def test_equation_mobject_prefers_mathtex(monkeypatch) -> None:
    calls = []

    def fake_mathtex(expr, color=None):
        calls.append(("mathtex", expr, color))
        return _DummyMobject("mathtex")

    def fake_tex(expr, color=None):
        calls.append(("tex", expr, color))
        return _DummyMobject("tex")

    monkeypatch.setattr(equation_scene, "MathTex", fake_mathtex)
    monkeypatch.setattr(equation_scene, "Tex", fake_tex)

    scene = object.__new__(equation_scene.EquationScene)
    result = equation_scene.EquationScene._make_equation_mobject(
        scene,
        r"\vec{F}_{net} = 0 \rightarrow v = constant",
    )

    assert result.kind == "mathtex"
    assert calls[0][0] == "mathtex"
    assert all(call[0] != "tex" for call in calls)


def test_equation_mobject_falls_back_to_tex_before_text(monkeypatch) -> None:
    calls = []

    def fail_mathtex(expr, color=None):
        calls.append(("mathtex", expr, color))
        raise ValueError("math failed")

    def fake_tex(expr, color=None):
        calls.append(("tex", expr, color))
        return _DummyMobject("tex")

    monkeypatch.setattr(equation_scene, "MathTex", fail_mathtex)
    monkeypatch.setattr(equation_scene, "Tex", fake_tex)

    scene = object.__new__(equation_scene.EquationScene)
    result = equation_scene.EquationScene._make_equation_mobject(
        scene,
        r"\sum \vec{F} = 0 \rightarrow \vec{a} = 0",
    )

    assert result.kind == "tex"
    assert [call[0] for call in calls] == ["mathtex", "tex"]


def test_equation_mobject_plain_text_fallback_strips_latex(monkeypatch) -> None:
    captured = {}

    def fail_mathtex(expr, color=None):
        raise ValueError("math failed")

    def fail_tex(expr, color=None):
        raise ValueError("tex failed")

    class FakeText(_DummyMobject):
        def __init__(self, text, font=None, font_size=None, color=None):
            captured["text"] = text
            super().__init__("text")

    monkeypatch.setattr(equation_scene, "MathTex", fail_mathtex)
    monkeypatch.setattr(equation_scene, "Tex", fail_tex)
    monkeypatch.setattr(equation_scene, "Text", FakeText)

    scene = object.__new__(equation_scene.EquationScene)
    result = equation_scene.EquationScene._make_equation_mobject(
        scene,
        r"\vec{F}_{net} = 0 \rightarrow v = constant",
    )

    assert result.kind == "text"
    assert "\\" not in captured["text"]
    assert "→" in captured["text"]
