from __future__ import annotations

import logging
from hashlib import sha256
from typing import Any

from cache import get_learner_context_cache
from mongodb import get_learner_profile, save_learner_profile
from modules.planning.profile_context import format_learner_context, normalize_profile

logger = logging.getLogger(__name__)


def _context_hash(context_text: str) -> str:
    return sha256(context_text.encode("utf-8")).hexdigest()


def _profile_version(profile: dict[str, Any] | None) -> int:
    if not profile:
        return 1
    try:
        return max(1, int(profile.get("profile_version") or 1))
    except (TypeError, ValueError):
        return 1


def get_learner_context(
    learner_id: str,
    topic: str,
    subject: str = "Physics",
    fallback_profile: dict[str, Any] | None = None,
) -> str:
    """Return a cached learner context string, rebuilding it if the profile version changed."""
    cache = get_learner_context_cache()
    learner_id = str(learner_id or "").strip()
    if not learner_id and fallback_profile:
        learner_id = str(fallback_profile.get("learner_id") or "").strip()
    if not learner_id:
        learner_id = "default-learner"

    profile = get_learner_profile(learner_id)
    if profile is None and fallback_profile:
        profile = save_learner_profile({**fallback_profile, "learner_id": learner_id})

    if profile is None:
        profile = {
            "learner_id": learner_id,
            "name": "Learner",
            "academic_level": "class_11",
            "exam_target": [],
            "learning_style": "visual",
            "pace_preference": "balanced",
            "weak_subjects": [],
            "confidence_map": {},
            "subject_for_lesson": subject,
            "subject_confidence": 50,
            "profile_version": 1,
        }

    current_version = _profile_version(profile)
    cached = cache.get_learner_context(learner_id)
    if cached and int(cached.get("profile_version") or 0) == current_version:
        cached_text = cached.get("context_text")
        if isinstance(cached_text, str) and cached_text.strip():
            return cached_text

    normalized_profile = normalize_profile(profile, subject)
    stable_profile = dict(normalized_profile)
    confidence_map = stable_profile.get("confidence_map") or {}
    if confidence_map:
        try:
            stable_profile["subject_confidence"] = int(min(confidence_map.values()))
        except Exception:
            stable_profile["subject_confidence"] = 50
    context_text = format_learner_context(stable_profile, "", "")
    cache.set_learner_context(
        learner_id,
        current_version,
        context_text,
        context_hash=_context_hash(context_text),
    )
    logger.debug(
        "Rebuilt learner context for learner_id=%s profile_version=%s",
        learner_id,
        current_version,
    )
    return context_text


def invalidate_learner_context(learner_id: str) -> None:
    get_learner_context_cache().invalidate_learner_context(learner_id)
