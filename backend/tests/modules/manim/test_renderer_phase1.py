from __future__ import annotations

from pathlib import Path

from modules.manim import renderer


def _source(path: Path) -> None:
    path.write_text(
        "class GeneratedScene(Scene):\n"
        "    def construct(self):\n"
        "        pass\n",
        encoding="utf-8",
    )


def test_nonzero_manim_execution_is_exposed_with_provenance(monkeypatch, tmp_path: Path) -> None:
    scene = tmp_path / "scene_3.py"
    _source(scene)
    monkeypatch.setattr(renderer, "MANIM_MAX_RETRIES", 1)

    def failed_run(_scene, _class, _media):
        renderer._last_execution = renderer._Execution(returncode=1, stderr="traceback")
        return None

    monkeypatch.setattr(renderer, "_run_manim", failed_run)
    monkeypatch.setattr(renderer, "_render_minimal_fallback", lambda *_args: (tmp_path / "missing.mp4", "minimal_safe"))

    result = renderer.render(scene, scene_class="GeneratedScene", return_report=True)

    assert result.semantic_render_succeeded is False
    assert result.fallback_used is True
    assert result.attempts[0]["returncode"] == 1
    assert result.stderr == "traceback"


def test_compiler_stub_is_exposed_before_manim_execution(monkeypatch, tmp_path: Path) -> None:
    scene = tmp_path / "scene_2.py"
    _source(scene)
    monkeypatch.setattr(renderer, "_run_manim", lambda *_args: (_ for _ in ()).throw(AssertionError("must not render")))

    result = renderer.render(
        scene,
        scene_class="GeneratedScene",
        expected_template="inertia",
        return_report=True,
    )

    assert result.render_status == "static_validation_failed"
    assert result.fallback_used is True
    assert result.fallback_type == "compile_stub"


def test_repaired_render_is_reported_only_after_actual_execution(monkeypatch, tmp_path: Path) -> None:
    scene = tmp_path / "scene_3.py"
    _source(scene)
    artifact = tmp_path / "render.mp4"
    artifact.touch()
    monkeypatch.setattr(renderer, "MANIM_MAX_RETRIES", 2)
    calls = 0

    def run(_scene, _class, _media):
        nonlocal calls
        calls += 1
        if calls == 1:
            renderer._last_execution = renderer._Execution(returncode=1, stderr="first failure")
            return None
        renderer._last_execution = renderer._Execution(returncode=0, stdout="rendered", artifact_path=artifact)
        return artifact

    monkeypatch.setattr(renderer, "_run_manim", run)
    monkeypatch.setattr(renderer, "_should_skip_llm_repair", lambda _error: False)
    monkeypatch.setattr(renderer, "_can_attempt_llm_repair", lambda: True)
    monkeypatch.setattr(renderer, "_try_repair", lambda *_args: True)

    result = renderer.render(scene, scene_class="GeneratedScene", return_report=True)

    assert result.semantic_render_succeeded is True
    assert result.primary_render_succeeded is False
    assert result.repair_attempted is True
    assert result.repair_succeeded is True
    assert result.fallback_used is False
    assert result.attempts[-1]["returncode"] == 0
