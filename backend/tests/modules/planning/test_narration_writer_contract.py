from __future__ import annotations

from pathlib import Path

from modules.llm.nvidia_client import NvidiaEmptyResponseError
from modules.planning import narration_writer


def _plan() -> dict:
    return {
        "scene_id": 1,
        "title": "A concept",
        "anchor_example": "A simple example",
        "learning_goal": "explain the idea",
        "visual_instruction": "Show the object changing state.",
        "events": [{"anchor_phrase": "the object changes"}],
    }


def test_narration_uses_dedicated_token_budget_and_retries_short_output(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    class FakeClient:
        def chat(self, model, messages, **kwargs):
            calls.append(kwargs["max_tokens"])
            return "the object changes" if len(calls) == 1 else "the object changes while the visible state explains the concept clearly"

    monkeypatch.setattr(narration_writer, "NvidiaClient", lambda: FakeClient())
    monkeypatch.setattr(narration_writer, "PATHS", {**narration_writer.PATHS, "audio": tmp_path})
    monkeypatch.setattr(narration_writer, "pace_word_budget", lambda profile: (8, 12))
    monkeypatch.setattr(narration_writer, "NVIDIA_NARRATION_MAX_TOKENS", 2048)

    result = narration_writer.write_narration(_plan(), learner_context="", subject="Physics")

    assert len(result.split()) >= 8
    assert calls == [2048, 2048]


def test_empty_length_response_is_retried_and_not_accepted(monkeypatch, tmp_path: Path) -> None:
    calls = 0

    class FakeClient:
        def chat(self, model, messages, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise NvidiaEmptyResponseError("length")
            return "the object changes while the visible state explains the concept clearly"

    monkeypatch.setattr(narration_writer, "NvidiaClient", lambda: FakeClient())
    monkeypatch.setattr(narration_writer, "PATHS", {**narration_writer.PATHS, "audio": tmp_path})
    monkeypatch.setattr(narration_writer, "pace_word_budget", lambda profile: (8, 12))

    result = narration_writer.write_narration(_plan(), learner_context="", subject="Physics")

    assert "the object changes" in result
    assert calls == 2


def test_narration_validation_rejects_short_or_out_of_order_text() -> None:
    errors = narration_writer._narration_validation_errors(
        "second phrase first phrase", ["first phrase", "second phrase"], 6, 8
    )

    assert any("word count" in error for error in errors)
    assert "required phrases are out of order" in errors


def test_narration_prompt_lists_required_phrases_in_order(monkeypatch, tmp_path: Path) -> None:
    captured = []

    class FakeClient:
        def chat(self, model, messages, **kwargs):
            captured.append(messages[1]["content"])
            return "first idea introduces the concept before second idea completes the explanation"

    plan = _plan()
    plan["events"] = [
        {"anchor_phrase": "first idea"},
        {"anchor_phrase": "second idea"},
    ]
    monkeypatch.setattr(narration_writer, "NvidiaClient", lambda: FakeClient())
    monkeypatch.setattr(narration_writer, "PATHS", {**narration_writer.PATHS, "audio": tmp_path})
    monkeypatch.setattr(narration_writer, "pace_word_budget", lambda profile: (8, 14))

    narration_writer.write_narration(plan, learner_context="", subject="Physics")

    prompt = captured[0]
    assert '1. "first idea"' in prompt
    assert '2. "second idea"' in prompt
    assert prompt.index('1. "first idea"') < prompt.index('2. "second idea"')
    assert "Output narration text only: no JSON, bullets, headings, or commentary." in prompt


def test_order_retry_repeats_ordered_phrases_and_actionable_feedback(monkeypatch, tmp_path: Path) -> None:
    prompts = []
    calls = 0

    class FakeClient:
        def chat(self, model, messages, **kwargs):
            nonlocal calls
            calls += 1
            prompts.append(messages[1]["content"])
            if calls == 1:
                return "second idea comes before first idea, which is incorrect"
            return "first idea explains the setup before second idea explains the result"

    plan = _plan()
    plan["events"] = [
        {"anchor_phrase": "first idea"},
        {"anchor_phrase": "second idea"},
    ]
    monkeypatch.setattr(narration_writer, "NvidiaClient", lambda: FakeClient())
    monkeypatch.setattr(narration_writer, "PATHS", {**narration_writer.PATHS, "audio": tmp_path})
    monkeypatch.setattr(narration_writer, "pace_word_budget", lambda profile: (8, 14))

    narration_writer.write_narration(plan, learner_context="", subject="Physics")

    retry = prompts[1]
    assert "previous attempt failed because the required phrases occurred in the wrong order" in retry
    assert '1. "first idea"' in retry
    assert '2. "second idea"' in retry
    assert retry.index('1. "first idea"') < retry.index('2. "second idea"')


def test_word_count_retry_reports_actual_count_and_preserves_phrase_contract(monkeypatch, tmp_path: Path) -> None:
    prompts = []
    calls = 0

    class FakeClient:
        def chat(self, model, messages, **kwargs):
            nonlocal calls
            calls += 1
            prompts.append(messages[1]["content"])
            if calls == 1:
                return "the object changes one two three four five six seven eight nine ten"
            return "the object changes explains concept clearly now"

    monkeypatch.setattr(narration_writer, "NvidiaClient", lambda: FakeClient())
    monkeypatch.setattr(narration_writer, "PATHS", {**narration_writer.PATHS, "audio": tmp_path})
    monkeypatch.setattr(narration_writer, "pace_word_budget", lambda profile: (4, 8))

    narration_writer.write_narration(_plan(), learner_context="", subject="Physics")

    retry = prompts[1]
    assert "word count 13 outside requested range 4-8" in retry
    assert "preserving every required phrase verbatim and in the exact required order" in retry


def test_correct_phrase_order_passes_and_wrong_order_fails() -> None:
    assert narration_writer._narration_validation_errors(
        "first idea explains the setup before second idea explains the result",
        ["first idea", "second idea"],
        8,
        14,
    ) == []
    errors = narration_writer._narration_validation_errors(
        "second idea appears before first idea",
        ["first idea", "second idea"],
        4,
        10,
    )
    assert "required phrases are out of order" in errors
