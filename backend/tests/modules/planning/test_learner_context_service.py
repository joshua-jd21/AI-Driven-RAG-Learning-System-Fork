from __future__ import annotations

import modules.planning.learner_context_service as service
import modules.planning.storyboard as storyboard


class DummyCache:
    def __init__(self):
        self.store = {}
        self.set_calls = 0
        self.invalidate_calls = 0

    def get_learner_context(self, learner_id):
        return self.store.get(learner_id)

    def set_learner_context(self, learner_id, profile_version, context_text, **kwargs):
        self.set_calls += 1
        self.store[learner_id] = {
            "profile_version": profile_version,
            "context_text": context_text,
            **kwargs,
        }

    def invalidate_learner_context(self, learner_id):
        self.invalidate_calls += 1
        self.store.pop(learner_id, None)


def test_same_learner_different_topics_and_subjects_hit_cache_without_regeneration(monkeypatch):
    cache = DummyCache()
    cache.store["learner-1"] = {
        "profile_version": 7,
        "context_text": "stable-context",
        "context_hash": "abc123",
    }

    monkeypatch.setattr(service, "get_learner_context_cache", lambda: cache)
    monkeypatch.setattr(service, "get_learner_profile", lambda learner_id: {"learner_id": learner_id, "profile_version": 7})

    calls = {"format": 0}

    def fake_format(profile, topic, subject):
        calls["format"] += 1
        return f"generated::{topic}::{subject}"

    monkeypatch.setattr(service, "format_learner_context", fake_format)

    ctx1 = service.get_learner_context("learner-1", "Momentum", "Physics")
    ctx2 = service.get_learner_context("learner-1", "Energy", "Chemistry")
    ctx3 = service.get_learner_context("learner-1", "Optics", "Mathematics")

    assert ctx1 == "stable-context"
    assert ctx2 == "stable-context"
    assert ctx3 == "stable-context"
    assert calls["format"] == 0
    assert cache.set_calls == 0
    assert set(cache.store["learner-1"].keys()) <= {"profile_version", "context_text", "context_hash"}


def test_profile_change_regenerates_and_updates_cache(monkeypatch):
    cache = DummyCache()
    cache.store["learner-1"] = {
        "profile_version": 1,
        "context_text": "old-context",
    }

    monkeypatch.setattr(service, "get_learner_context_cache", lambda: cache)
    monkeypatch.setattr(
        service,
        "get_learner_profile",
        lambda learner_id: {
            "learner_id": learner_id,
            "profile_version": 2,
            "name": "Nia",
            "academic_level": "class_11",
            "learning_style": "visual",
            "pace_preference": "slow_deep",
            "confidence_map": {"Physics": 40, "Chemistry": 50},
        },
    )

    calls = {"format": 0}

    def fake_format(profile, topic, subject):
        calls["format"] += 1
        return f"generated::{profile['pace_preference']}"

    monkeypatch.setattr(service, "format_learner_context", fake_format)

    ctx = service.get_learner_context("learner-1", "Momentum", "Physics")

    assert ctx == "generated::slow_deep"
    assert calls["format"] == 1
    assert cache.set_calls == 1
    assert cache.store["learner-1"]["profile_version"] == 2
    assert cache.store["learner-1"]["context_text"] == "generated::slow_deep"


def test_lesson_prompt_receives_topic_and_subject(monkeypatch, tmp_path):
    captured = {}

    class FakeClient:
        def chat_json(
            self,
            model,
            messages,
            temperature=0.0,
            max_tokens=0,
            extra_body=None,
        ):
            captured["messages"] = messages
            return [
                {
                    "scene_id": 1,
                    "concept_template": "intro",
                    "scene_role": "hook",
                    "title": "Momentum — Overview",
                    "anchor_example": "a moving cart",
                    "learning_goal": "introduce momentum",
                    "subtitle": "A cart in motion",
                    "key_term": "momentum",
                },
                {
                    "scene_id": 2,
                    "concept_template": "freeform",
                    "scene_role": "visual_intuition",
                    "title": "Key idea",
                    "anchor_example": "a bowling ball",
                    "learning_goal": "show intuition",
                },
                {
                    "scene_id": 3,
                    "concept_template": "freeform",
                    "scene_role": "formal_concept",
                    "title": "Formal idea",
                    "anchor_example": "a collision",
                    "learning_goal": "show the formula",
                },
                {
                    "scene_id": 4,
                    "concept_template": "freeform",
                    "scene_role": "worked_example",
                    "title": "Worked example",
                    "anchor_example": "two carts",
                    "learning_goal": "apply it",
                },
                {
                    "scene_id": 5,
                    "concept_template": "summary",
                    "scene_role": "summary",
                    "title": "Key Takeaways",
                    "anchor_example": "all scenarios",
                    "learning_goal": "consolidate learning",
                    "summary_points": ["a", "b", "c"],
                },
            ]

    monkeypatch.setattr(storyboard, "NvidiaClient", lambda: FakeClient())
    monkeypatch.setattr(storyboard, "PATHS", {**storyboard.PATHS, "json": tmp_path})

    storyboard.build_storyboard(
        topic="Momentum",
        curriculum_context="",
        curriculum_sections=[],
        learner_profile={
            "learner_id": "learner-1",
            "name": "Nia",
            "academic_level": "class_11",
            "learning_style": "visual",
            "pace_preference": "balanced",
            "confidence_map": {"Physics": 55},
        },
        subject="Physics",
        learner_context="LEARNER CONTEXT (personalize ALL output to this student):\n- Name: Nia | Level: Class 11 (Senior Secondary)\n- Learning style: visual — Lead with animated diagrams, motion, and concrete visuals BEFORE any equation. Use arrows, color, and analogies to carry the idea.\n",
    )

    user_prompt = captured["messages"][1]["content"]
    assert "LESSON SUBJECT: Physics" in user_prompt
    assert "LESSON TOPIC: Momentum" in user_prompt
