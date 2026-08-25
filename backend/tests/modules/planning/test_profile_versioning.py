from __future__ import annotations

import mongodb


def test_profile_version_increments_on_context_affecting_changes():
    mongodb._LEARNER_PROFILES = None
    mongodb._MEMORY_PROFILES.clear()

    base_profile = {
        "learner_id": "learner-1",
        "name": "Ava",
        "academic_level": "class_11",
        "grade": "11",
        "board": "CBSE",
        "language": "English",
        "exam_target": ["JEE"],
        "learning_style": "visual",
        "pace_preference": "balanced",
        "confidence_map": {"Physics": 50, "Chemistry": 50, "Mathematics": 50},
    }

    saved1 = mongodb.save_learner_profile(base_profile)
    assert saved1["profile_version"] == 1

    saved2 = mongodb.save_learner_profile({**base_profile, "name": "Ava"})
    assert saved2["profile_version"] == 1

    saved3 = mongodb.save_learner_profile({**base_profile, "name": "Ari"})
    assert saved3["profile_version"] == 2

    saved4 = mongodb.save_learner_profile(
        {
            **base_profile,
            "name": "Ari",
            "confidence_map": {"Physics": 35, "Chemistry": 50, "Mathematics": 50},
        }
    )
    assert saved4["profile_version"] == 3


def test_profile_version_does_not_reset_when_lower_version_is_sent():
    mongodb._LEARNER_PROFILES = None
    mongodb._MEMORY_PROFILES.clear()

    saved1 = mongodb.save_learner_profile(
        {
            "learner_id": "learner-2",
            "name": "Kai",
            "academic_level": "class_11",
            "exam_target": [],
            "learning_style": "visual",
            "pace_preference": "balanced",
            "confidence_map": {},
            "profile_version": 9,
        }
    )
    assert saved1["profile_version"] == 9

    saved2 = mongodb.save_learner_profile(
        {
            "learner_id": "learner-2",
            "name": "Kian",
            "academic_level": "class_11",
            "exam_target": [],
            "learning_style": "visual",
            "pace_preference": "balanced",
            "confidence_map": {},
            "profile_version": 2,
        }
    )
    assert saved2["profile_version"] == 10
