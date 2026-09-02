"""Narration-segment planning from aligned words.

This module turns raw narration text plus Whisper-style word timestamps into
lightweight narration segments. The output is intentionally simple:

- contiguous timing windows that cover the narration
- a human-readable visual goal/state
- a small set of semantic animation primitives

The goal is not to replace the existing event timeline, but to add a higher
level bridge between speech and visuals that templates can consult when they
need persistent on-screen context.
"""
from __future__ import annotations

import re
from typing import Any

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def build_narration_segments(
    narration: str,
    word_timestamps: list[dict[str, Any]],
    audio_duration: float,
    events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return a narration-segment list aligned to the spoken words.

    The returned segments are sentence-ish spans. They are not intended to be
    perfect linguistic segments; they are intentionally conservative, stable,
    and good enough for deterministic animation planning.
    """
    narration = narration.strip()
    if not narration:
        return []

    words = _normalize_word_timestamps(word_timestamps)
    if not words:
        return [
            _segment_record(
                segment_id="s0",
                start=0.0,
                end=max(float(audio_duration), 0.0),
                text=narration,
                words=[],
                event=None,
                sentence_index=0,
                sentence_count=1,
            )
        ]

    sentences = _split_sentences(narration)
    if not sentences:
        sentences = [narration]

    segments: list[dict[str, Any]] = []
    cursor = 0
    for idx, sentence in enumerate(sentences):
        tokens = _tokenize(sentence)
        if not tokens:
            continue

        start_idx, end_idx = _match_sentence(words, cursor, tokens)
        if idx == len(sentences) - 1:
            end_idx = len(words)
        if end_idx <= start_idx:
            end_idx = min(len(words), max(start_idx + 1, len(words)))

        segment_words = words[start_idx:end_idx] if start_idx < len(words) else []
        if segment_words:
            start = float(segment_words[0]["start"])
            end = float(segment_words[-1]["end"])
        else:
            start = 0.0 if not segments else segments[-1]["end"]
            end = start + max(0.8, min(2.0, len(tokens) * 0.12))

        event = _pick_event(events or [], start, end, sentence)
        segments.append(
            _segment_record(
                segment_id=f"s{idx}",
                start=start,
                end=end,
                text=sentence,
                words=segment_words,
                event=event,
                sentence_index=idx,
                sentence_count=len(sentences),
            )
        )
        cursor = max(end_idx, cursor + 1)

    if segments:
        segments[0]["start"] = 0.0
        segments[-1]["end"] = max(float(audio_duration), segments[-1]["end"])

    return _merge_tiny_gaps(segments, audio_duration)


def _normalize_word_timestamps(word_timestamps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for item in word_timestamps or []:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word", "")).strip()
        if not word:
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        words.append({"word": word, "start": start, "end": max(end, start)})
    return words


def _split_sentences(narration: str) -> list[str]:
    pieces = [piece.strip() for piece in _SENTENCE_SPLIT_RE.split(narration) if piece.strip()]
    if pieces:
        return pieces
    return [narration.strip()] if narration.strip() else []


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _match_sentence(
    words: list[dict[str, Any]],
    cursor: int,
    tokens: list[str],
) -> tuple[int, int]:
    """Find the best contiguous span for tokens starting at/after cursor."""
    word_tokens = [_normalize_word(w["word"]) for w in words]
    n = len(tokens)
    if n == 0:
        return cursor, cursor

    for i in range(cursor, max(len(word_tokens) - n + 1, cursor + 1)):
        if word_tokens[i : i + n] == tokens:
            return i, i + n

    end = min(len(words), cursor + max(n, 1))
    return cursor, max(cursor + 1, end)


def _normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", word.lower())


def _pick_event(
    events: list[dict[str, Any]],
    start: float,
    end: float,
    text: str = "",
) -> dict[str, Any] | None:
    if not events:
        return None

    text_l = text.lower()
    # Anchor phrases are the stable semantic link before event times exist.
    for ev in events:
        phrase = str(ev.get("anchor_phrase", ev.get("narration_reference", ""))).strip()
        if phrase and phrase.lower() in text_l:
            return ev

    best: dict[str, Any] | None = None
    best_score = float("inf")
    for ev in events:
        ev_start = float(ev.get("start", 0.0))
        ev_end = ev_start + float(ev.get("run_time", 0.0)) + float(ev.get("hold_after", 0.0))
        if ev_start <= end and ev_end >= start:
            return ev
        score = min(abs(ev_start - start), abs(ev_start - end))
        if score < best_score:
            best = ev
            best_score = score
    return best


def _segment_record(
    *,
    segment_id: str,
    start: float,
    end: float,
    text: str,
    words: list[dict[str, Any]],
    event: dict[str, Any] | None,
    sentence_index: int,
    sentence_count: int,
) -> dict[str, Any]:
    goal = _visual_goal(text, event)
    state = _visual_state(text, goal, event, sentence_index, sentence_count)
    actions = _visual_actions(text, event)
    semantic_action = str((event or {}).get("action", "")).strip().lower()
    if semantic_action and semantic_action not in actions:
        actions.insert(0, semantic_action)
    return {
        "segment_id": segment_id,
        "start": round(max(0.0, start), 3),
        "end": round(max(start, end), 3),
        "text": text.strip(),
        "words": words,
        "visual_goal": goal,
        "visual_state": state,
        "actions": actions,
        "narration_reference": str(
            (event or {}).get("narration_reference", (event or {}).get("anchor_phrase", ""))
        ).strip(),
        "visible_objects": list((event or {}).get("visible_objects", [])),
        "action": semantic_action or (actions[0] if actions else "hold"),
        "action_reason": str((event or {}).get("action_reason", "")).strip(),
        "emphasis_targets": list((event or {}).get("emphasis_targets", [])),
        "persistence_after_action": bool((event or {}).get("persistence_after_action", True)),
    }


def _visual_goal(text: str, event: dict[str, Any] | None) -> str:
    if event and str(event.get("visual_goal", "")).strip():
        return str(event["visual_goal"]).strip()
    if event and str(event.get("anchor_phrase", "")).strip():
        return str(event["anchor_phrase"]).strip()
    tokens = _tokenize(text)
    if not tokens:
        return text.strip()[:80]
    return " ".join(tokens[:8])


def _visual_state(
    text: str,
    visual_goal: str,
    event: dict[str, Any] | None,
    sentence_index: int,
    sentence_count: int,
) -> str:
    if event and str(event.get("visual_state", "")).strip():
        return str(event["visual_state"]).strip()
    text_l = text.lower()
    event_type = str(event.get("type", "")) if event else ""

    if "equation" in text_l or "=" in text_l or "formula" in text_l:
        return f"Keep the equation visible while '{visual_goal}' is explained."
    if any(k in text_l for k in ("force", "motion", "move", "velocity", "speed", "accelerat")):
        return f"Keep the core motion visual visible while '{visual_goal}' is spoken."
    if any(k in text_l for k in ("diagram", "label", "arrow", "relationship", "cause", "effect")):
        return f"Maintain the diagram context while '{visual_goal}' is emphasized."
    if event_type == "hold":
        return f"Hold the current visual state while '{visual_goal}' continues."
    if sentence_index == 0:
        return "Introduce the main visual context immediately and keep it on screen."
    if sentence_index == sentence_count - 1:
        return "Preserve the concluding visual state until the narration ends."
    return f"Maintain the current learning visual while '{visual_goal}' is explained."


def _visual_actions(text: str, event: dict[str, Any] | None) -> list[str]:
    text_l = text.lower()
    event_type = str(event.get("type", "")) if event else ""
    actions: list[str] = []

    if event_type in {"place_title", "highlight_term", "place_point"}:
        actions.append("Write")
        actions.append("FadeIn")
    elif event_type in {"reveal", "show", "label"}:
        actions.append("FadeIn")
        actions.append("Indicate")
    elif event_type in {"highlight", "hold"}:
        actions.append("Indicate")
    else:
        actions.append("FadeIn")

    if any(k in text_l for k in ("force", "motion", "move", "velocity", "speed", "accelerat")):
        actions.append("MoveAlongPath")
    if any(k in text_l for k in ("equation", "formula", "=")):
        actions.append("Transform")
        actions.append("Circumscribe")
    if any(k in text_l for k in ("diagram", "relationship", "process", "flow")):
        actions.append("Create")
        actions.append("GrowArrow")

    # Preserve order and uniqueness without expensive machinery.
    unique: list[str] = []
    for action in actions:
        if action not in unique:
            unique.append(action)
    return unique


def _merge_tiny_gaps(
    segments: list[dict[str, Any]],
    audio_duration: float,
) -> list[dict[str, Any]]:
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["start"] < prev["end"]:
            prev["end"] = max(prev["end"], seg["end"])
            prev["text"] = f'{prev["text"]} {seg["text"]}'.strip()
            prev["words"] = list(prev.get("words", [])) + list(seg.get("words", []))
            prev["actions"] = _merge_unique(prev.get("actions", []), seg.get("actions", []))
            continue
        if 0.0 <= seg["start"] - prev["end"] <= 0.25:
            prev["end"] = seg["start"]
        merged.append(seg)

    merged[0]["start"] = 0.0
    merged[-1]["end"] = max(float(audio_duration), merged[-1]["end"])
    return merged


def _merge_unique(first: list[str], second: list[str]) -> list[str]:
    out: list[str] = []
    for item in list(first) + list(second):
        if item not in out:
            out.append(item)
    return out
