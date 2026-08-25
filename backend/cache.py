from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from dotenv import load_dotenv

try:  # pragma: no cover - import availability depends on environment
    import redis
except Exception:  # pragma: no cover - redis may not be installed in some environments
    redis = None

load_dotenv()

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
LEARNER_CONTEXT_CACHE_TTL_SECONDS = int(os.getenv("LEARNER_CONTEXT_CACHE_TTL_SECONDS", "3600"))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _context_hash(context_text: str) -> str:
    return sha256(context_text.encode("utf-8")).hexdigest()


@dataclass
class LearnerContextCacheEntry:
    profile_version: int
    context_text: str
    context_hash: str | None = None
    generated_at: str | None = None


class LearnerContextCache:
    def __init__(self) -> None:
        self._client = None
        if redis is None:
            logger.warning("redis package is not available; learner context caching is disabled")
            return
        try:
            self._client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client.ping()
        except Exception as exc:  # pragma: no cover - only hit when Redis is unavailable
            logger.warning("Redis unavailable; learner context caching is disabled: %s", exc)
            self._client = None

    @staticmethod
    def key_for(learner_id: str) -> str:
        return f"learner_context:{learner_id}"

    def get_learner_context(self, learner_id: str) -> dict[str, Any] | None:
        if not self._client:
            return None
        raw = self._client.get(self.key_for(learner_id))
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if "profile_version" in data:
            try:
                data["profile_version"] = int(data["profile_version"])
            except (TypeError, ValueError):
                data["profile_version"] = 0
        return data

    def set_learner_context(
        self,
        learner_id: str,
        profile_version: int,
        context_text: str,
        *,
        context_hash: str | None = None,
        generated_at: str | None = None,
    ) -> None:
        if not self._client:
            return
        entry = LearnerContextCacheEntry(
            profile_version=int(profile_version),
            context_text=context_text,
            context_hash=context_hash or _context_hash(context_text),
            generated_at=generated_at or _utcnow(),
        )
        self._client.set(
            self.key_for(learner_id),
            json.dumps(entry.__dict__, ensure_ascii=False),
            ex=LEARNER_CONTEXT_CACHE_TTL_SECONDS,
        )

    def invalidate_learner_context(self, learner_id: str) -> None:
        if not self._client:
            return
        self._client.delete(self.key_for(learner_id))


_CACHE: LearnerContextCache | None = None


def get_learner_context_cache() -> LearnerContextCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = LearnerContextCache()
    return _CACHE
