from __future__ import annotations

from types import SimpleNamespace

import requests

from modules.manim import renderer
from modules.planning import semantic_plan


def test_planner_model_remains_the_planning_model() -> None:
    assert semantic_plan.NVIDIA_PLANNER_MODEL
    assert renderer.NVIDIA_PLANNER_MODEL == semantic_plan.NVIDIA_PLANNER_MODEL


def test_repair_falls_back_to_planner_model_on_404(monkeypatch) -> None:
    calls: list[str] = []

    class FakeClient:
        def chat(self, model, messages, **kwargs):
            calls.append(model)
            if len(calls) == 1:
                response = SimpleNamespace(status_code=404)
                raise requests.HTTPError("model unavailable", response=response)
            return "fixed scene"

    monkeypatch.setattr(renderer, "NVIDIA_REPAIR_MODEL", "unavailable/repair")
    monkeypatch.setattr(renderer, "NVIDIA_PLANNER_MODEL", "verified/planner")
    monkeypatch.setattr(renderer, "MANIM_REPAIR_TIMEOUT", 19)

    result = renderer._chat_with_repair_fallback(FakeClient(), [])

    assert result == "fixed scene"
    assert calls == ["unavailable/repair", "verified/planner"]


def test_repair_uses_configured_model_when_available(monkeypatch) -> None:
    calls: list[str] = []

    class FakeClient:
        def chat(self, model, messages, **kwargs):
            calls.append(model)
            return "fixed scene"

    monkeypatch.setattr(renderer, "NVIDIA_REPAIR_MODEL", "configured/repair")
    monkeypatch.setattr(renderer, "NVIDIA_PLANNER_MODEL", "verified/planner")

    assert renderer._chat_with_repair_fallback(FakeClient(), []) == "fixed scene"
    assert calls == ["configured/repair"]
