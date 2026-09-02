"""Manim renderer with Gemini-driven repair retry loop."""

from __future__ import annotations

import os

import requests

import re

import shutil

import subprocess

import time
from dataclasses import asdict, dataclass

from pathlib import Path

from modules.config import (

    MANIM_MAX_RETRIES,

    MANIM_QUALITY,

    MANIM_REPAIR_MAX_CALLS,

    MANIM_REPAIR_TIMEOUT,

    NVIDIA_REPAIR_MODEL,
    NVIDIA_PLANNER_MODEL,

    PATHS,

    RenderWorkspace,

    get_logger,

)

from modules.llm.nvidia_client import NvidiaClient

from modules.manim.code_sanitize import is_latex_render_error, strip_latex_mobjects
from modules.manim.render_validation import validate_static_source

logger = get_logger(__name__)


@dataclass
class RenderResult:
    """Renderer provenance; a path alone cannot distinguish semantic output from fallback."""

    artifact_path: Path
    scene_path: Path
    scene_class: str
    render_status: str
    semantic_render_succeeded: bool
    primary_render_succeeded: bool
    repair_attempted: bool
    repair_succeeded: bool
    fallback_used: bool
    fallback_type: str | None
    returncode: int | None
    stdout: str
    stderr: str
    attempts: list[dict[str, object]]
    static_validation: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["artifact_path"] = str(self.artifact_path)
        data["scene_path"] = str(self.scene_path)
        return data


@dataclass
class _Execution:
    artifact_path: Path | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""

REPAIR_SYSTEM = """You are an expert Manim Community Edition debugger.
Return ONLY the corrected Python file. No markdown fences, no commentary.
Keep class name GeneratedScene and keep all run_time values exactly as-is.
Never use .get_edge() (use .get_left/right/top/bottom). Never use ApplyMethod.
SAFE AREA: all mobjects must stay within x in [-6.6, 6.6] and y in [-3.6, 3.6].
Use scale_to_fit_width(12.0) on titles; Text(..., width=5.0) for body wrapping.
Ensure no mobjects overlap (use .arrange or .next_to with buff>=0.4).
Prefer LaggedStart with rate_func=smooth for multi-object reveals."""

REPAIR_PROMPT = """The following Manim script failed to render.

ERROR:
{error}

CURRENT CODE:
{code}

Return the COMPLETE fixed Python file only."""

_MINIMAL_SAFE_SCENE = """from manim import *


class GeneratedScene(Scene):
    def construct(self):
        title = Text("Lesson Scene", font_size=36)
        self.play(FadeIn(title), run_time=1.0)
        self.wait(1.0)
"""

_ULTRA_MINIMAL_SCENE = """from manim import *


class GeneratedScene(Scene):
    def construct(self):
        self.wait(2.0)
"""

_last_error: str = ""
_last_execution = _Execution()
_llm_repair_disabled = False
_llm_repair_attempts = 0


def reset_llm_repair_state() -> None:
    """Reset LLM repair circuit breaker (call once at pipeline start)."""
    global _llm_repair_disabled, _llm_repair_attempts
    _llm_repair_disabled = False
    _llm_repair_attempts = 0


def _scene_dest(scene_py: Path, workspace: RenderWorkspace | None) -> Path:
    base = workspace.scenes_dir if workspace is not None else PATHS["renders"]
    return base / f"{scene_py.stem}.mp4"


def _media_dir(workspace: RenderWorkspace | None) -> Path:
    if workspace is not None:
        return workspace.media_dir
    media = PATHS["manim"] / "media"
    media.mkdir(parents=True, exist_ok=True)
    return media


def render(
    scene_py: Path,
    scene_class: str = "GeneratedScene",
    fallback_code: str | None = None,
    workspace: RenderWorkspace | None = None,
    return_report: bool = False,
    expected_template: str | None = None,
) -> Path | RenderResult:
    """Render a Manim scene file to MP4 with retry on failure."""
    global _last_execution

    # Keep each scene's process provenance isolated from the prior scene.
    _last_execution = _Execution()
    media_dir = _media_dir(workspace)
    dest = _scene_dest(scene_py, workspace)
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[RENDER] scene=%s workspace=%s dest=%s",
        scene_py.stem,
        media_dir,
        dest,
    )

    try:
        source = scene_py.read_text(encoding="utf-8")
        static_validation = validate_static_source(
            source,
            scene_class,
            expected_template=expected_template,
        )
    except OSError as exc:
        static_validation = {
            "passed": False,
            "checks": [{"check": "source_read", "passed": False, "details": str(exc)}],
            "failure_classifications": ["STATIC_VALIDATION_FAILED"],
        }

    attempts: list[dict[str, object]] = []
    repair_attempted = False
    repair_succeeded = False
    fallback_used = "COMPILE_FALLBACK" in static_validation.get("failure_classifications", [])
    fallback_type: str | None = "compile_stub" if fallback_used else None
    primary_render_succeeded = False

    def finish(status: str, semantic_success: bool) -> Path | RenderResult:
        result = RenderResult(
            artifact_path=dest,
            scene_path=scene_py,
            scene_class=scene_class,
            render_status=status,
            semantic_render_succeeded=semantic_success,
            primary_render_succeeded=primary_render_succeeded,
            repair_attempted=repair_attempted,
            repair_succeeded=repair_succeeded,
            fallback_used=fallback_used,
            fallback_type=fallback_type,
            returncode=_last_execution.returncode,
            stdout=_last_execution.stdout,
            stderr=_last_execution.stderr,
            attempts=attempts,
            static_validation=static_validation,
        )
        return result if return_report else result.artifact_path

    if not static_validation.get("passed", False):
        logger.error("Static validation failed for %s", scene_py)
        return finish("static_validation_failed", False)

    for attempt in range(MANIM_MAX_RETRIES):
        mp4 = _run_manim(scene_py, scene_class, media_dir)
        execution = _last_execution
        attempts.append({
            "attempt": attempt + 1,
            "phase": "fallback" if fallback_used else ("repair" if repair_succeeded else "primary"),
            "returncode": execution.returncode,
            "artifact_path": str(mp4) if mp4 else None,
            "stdout": execution.stdout,
            "stderr": execution.stderr,
        })
        if mp4 and mp4.exists():
            if mp4 != dest:
                shutil.copy2(mp4, dest)
            if not fallback_used:
                primary_render_succeeded = not repair_succeeded
                logger.info("Render success: %s (attempt %d)", dest, attempt + 1)
                return finish("success", True)
            logger.info("Fallback render produced artifact: %s", dest)
            return finish("fallback", False)

        error = _last_error or "Unknown render error"
        logger.warning("Render attempt %d failed: %s", attempt + 1, error[:200])
        if repair_succeeded:
            # A repair is successful only when the repaired source also renders.
            repair_succeeded = False

        if attempt < MANIM_MAX_RETRIES - 1:
            skip_llm = _should_skip_llm_repair(error)
            repaired = False
            if is_latex_render_error(error):
                repaired = _try_strip_latex(scene_py)
            if not repaired and not skip_llm and _can_attempt_llm_repair():
                repair_attempted = True
                repaired = _try_repair(scene_py, error)
                repair_succeeded = repaired
            elif skip_llm and not repaired:
                logger.warning(
                    "Skipping LLM repair for known error pattern; using template fallback for %s",
                    scene_py.name,
                )
            if not repaired and fallback_code:
                fallback_used = True
                fallback_type = "template"
                logger.warning(
                    "Repair unavailable; writing template fallback to %s",
                    scene_py,
                )
                scene_py.write_text(strip_latex_mobjects(fallback_code), encoding="utf-8")

    if fallback_code:
        fallback_used = True
        fallback_type = "template"
        logger.warning(
            "All LLM attempts failed; falling back to deterministic template for %s",
            scene_py.name,
        )
        scene_py.write_text(strip_latex_mobjects(fallback_code), encoding="utf-8")
        mp4 = _run_manim(scene_py, scene_class, media_dir)
        execution = _last_execution
        attempts.append({
            "attempt": len(attempts) + 1,
            "phase": "fallback",
            "returncode": execution.returncode,
            "artifact_path": str(mp4) if mp4 else None,
            "stdout": execution.stdout,
            "stderr": execution.stderr,
        })
        if mp4 and mp4.exists():
            shutil.copy2(mp4, dest)
            logger.info("Template fallback render success: %s", dest)
            return finish("fallback", False)

    fallback_used = True
    fallback_type = "minimal_safe"
    mp4, fallback_type = _render_minimal_fallback(scene_py, scene_class, media_dir, dest)
    execution = _last_execution
    attempts.append({
        "attempt": len(attempts) + 1,
        "phase": fallback_type,
        "returncode": execution.returncode,
        "artifact_path": str(mp4) if mp4 else None,
        "stdout": execution.stdout,
        "stderr": execution.stderr,
    })
    return finish("fallback", False)


def _render_minimal_fallback(
    scene_py: Path,
    scene_class: str,
    media_dir: Path,
    dest: Path,
) -> tuple[Path, str]:
    """Guaranteed last-resort render so the pipeline never crashes on Manim failure."""
    for label, code in (
        ("minimal_safe", _MINIMAL_SAFE_SCENE),
        ("ultra_minimal", _ULTRA_MINIMAL_SCENE),
    ):
        logger.warning(
            "Using %s fallback scene for %s after all repair attempts failed",
            label,
            scene_py.name,
        )
        scene_py.write_text(code, encoding="utf-8")
        mp4 = _run_manim(scene_py, scene_class, media_dir)
        if mp4 and mp4.exists():
            shutil.copy2(mp4, dest)
            logger.info("%s fallback render success: %s", label, dest)
            return dest, label

    logger.error(
        "Minimal fallback renders failed for %s; writing placeholder mp4 path %s",
        scene_py.name,
        dest,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.touch()
    return dest, "minimal_safe"


def _find_scene_mp4(
    media_dir: Path,
    scene_py: Path,
    scene_class: str,
    render_start: float,
) -> Path | None:
    """Locate the MP4 produced by the current render (never a stale file)."""
    videos_root = media_dir / "videos" / scene_py.stem
    if videos_root.is_dir():
        for quality_dir in sorted(videos_root.iterdir(), reverse=True):
            candidate = quality_dir / f"{scene_class}.mp4"
            if candidate.is_file() and candidate.stat().st_mtime >= render_start:
                return candidate

    candidates = [
        p for p in media_dir.rglob("*.mp4")
        if scene_py.stem in p.as_posix() and p.stat().st_mtime >= render_start
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _run_manim(scene_py: Path, scene_class: str, media_dir: Path) -> Path | None:
    """Execute manim CLI and locate output MP4."""
    global _last_error, _last_execution
    manim_bin = shutil.which("manim") or "manim"
    cmd = [
        manim_bin,
        "render",
        MANIM_QUALITY,
        str(scene_py),
        scene_class,
        "--media_dir",
        str(media_dir),
        "--disable_caching",
    ]
    logger.info("Running: %s", " ".join(cmd))
    env = os.environ.copy()
    backend_root = str(PATHS["root"])
    env["PYTHONPATH"] = backend_root + os.pathsep + env.get("PYTHONPATH", "")
    render_start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=backend_root,
            env=env,
        )
        _last_execution = _Execution(
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )
        if result.returncode != 0:
            _last_error = result.stderr or result.stdout
            return None

        mp4 = _find_scene_mp4(media_dir, scene_py, scene_class, render_start)
        if mp4:
            _last_execution.artifact_path = mp4
            return mp4
        _last_error = "No MP4 output found after render"
        return None
    except subprocess.TimeoutExpired:
        _last_execution = _Execution(stderr="Manim render timed out after 300s")
        _last_error = "Manim render timed out after 300s"
        return None
    except Exception as exc:
        _last_execution = _Execution(stderr=str(exc))
        _last_error = str(exc)
        return None


def _has_repair_api_key() -> bool:
    return bool(os.getenv("NVIDIA_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _can_attempt_llm_repair() -> bool:
    if _llm_repair_disabled:
        logger.info("LLM repair circuit breaker open; skipping repair")
        return False
    if _llm_repair_attempts >= MANIM_REPAIR_MAX_CALLS:
        logger.info(
            "LLM repair budget exhausted (%d/%d); skipping repair",
            _llm_repair_attempts,
            MANIM_REPAIR_MAX_CALLS,
        )
        return False
    return _has_repair_api_key()


def _disable_llm_repair(reason: str) -> None:
    global _llm_repair_disabled
    if not _llm_repair_disabled:
        logger.warning("Disabling LLM repair for remainder of run: %s", reason)
    _llm_repair_disabled = True


def _should_skip_llm_repair(error: str) -> bool:
    """Skip LLM repair for errors we can fix deterministically via template recompile."""
    markers = ("ArrowTip", "NotImplementedError", "get_edge", "ApplyMethod")
    return is_latex_render_error(error) or any(m in error for m in markers)


def _try_strip_latex(scene_py: Path) -> bool:
    """Replace MathTex/Tex with Text when LaTeX is missing or misconfigured."""
    code = scene_py.read_text(encoding="utf-8")
    fixed = strip_latex_mobjects(code)
    if fixed == code:
        return False
    scene_py.write_text(fixed, encoding="utf-8")
    logger.info("Stripped LaTeX mobjects in %s", scene_py.name)
    return True


def _try_repair(scene_py: Path, error: str) -> bool:
    """Send failed code to NVIDIA NIM for repair. Returns True on success."""
    global _llm_repair_attempts

    if not _can_attempt_llm_repair():
        return False

    _llm_repair_attempts += 1
    logger.info(
        "Requesting LLM repair for %s (timeout=%ds, attempt %d/%d)",
        scene_py.name,
        MANIM_REPAIR_TIMEOUT,
        _llm_repair_attempts,
        MANIM_REPAIR_MAX_CALLS,
    )
    code = scene_py.read_text(encoding="utf-8")
    try:
        client = NvidiaClient(max_retries=1)
        messages = [
            {"role": "system", "content": REPAIR_SYSTEM},
            {
                "role": "user",
                "content": REPAIR_PROMPT.format(error=error[:3000], code=code[:8000]),
            },
        ]
        fixed = _chat_with_repair_fallback(client, messages)
    except Exception as exc:
        logger.warning("Repair LLM call failed: %s", exc)
        _disable_llm_repair(str(exc))
        return False

    fixed = fixed.strip()
    if "```python" in fixed:
        match = re.search(r"```python\s*(.*?)\s*```", fixed, re.DOTALL)
        if match:
            fixed = match.group(1)
    elif "```" in fixed:
        match = re.search(r"```\s*(.*?)\s*```", fixed, re.DOTALL)
        if match:
            fixed = match.group(1)
    fixed = fixed.strip()
    if "from manim import" in fixed and "GeneratedScene" in fixed:
        scene_py.write_text(fixed, encoding="utf-8")
        logger.info("Repaired code written to %s", scene_py.name)
        return True
    logger.warning("Repair output invalid; keeping original")
    _disable_llm_repair("invalid repair output")
    return False


def _chat_with_repair_fallback(client: NvidiaClient, messages: list[dict[str, str]]) -> str:
    """Retry the same repair request with the verified planner model on a 404."""
    models = [NVIDIA_REPAIR_MODEL]
    if NVIDIA_REPAIR_MODEL != NVIDIA_PLANNER_MODEL:
        models.append(NVIDIA_PLANNER_MODEL)

    for index, model in enumerate(models):
        try:
            return client.chat(
                model,
                messages,
                temperature=0.1,
                max_tokens=8192,
                timeout=MANIM_REPAIR_TIMEOUT,
            )
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404 and index == 0 and len(models) > 1:
                logger.warning(
                    "Repair model %s returned HTTP 404; retrying this repair with planner model %s",
                    model,
                    NVIDIA_PLANNER_MODEL,
                )
                continue
            raise
    raise RuntimeError("No configured repair model available")
