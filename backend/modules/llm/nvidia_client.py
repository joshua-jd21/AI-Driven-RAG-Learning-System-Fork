"""NVIDIA NIM API client with transparent Google Gemini API fallback."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests
import google.generativeai as genai

from modules.config import GEMINI_API_KEY, NVIDIA_API_KEY, get_logger

logger = get_logger(__name__)

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


class NvidiaEmptyResponseError(ValueError):
    """Raised when NIM returns no visible content, preserving its stop reason."""

    def __init__(self, finish_reason: str | None = None) -> None:
        self.finish_reason = finish_reason
        detail = ""
        if finish_reason:
            detail = f" (finish_reason={finish_reason})"
        super().__init__(f"NVIDIA NIM returned empty message content{detail}.")


class NvidiaClient:
    """REST client for NVIDIA NIM chat completions with Gemini fallback."""

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

        # Dynamically fetch keys so runtime/container environment changes
        # are picked up without requiring module reloads.
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY") or NVIDIA_API_KEY

        self.gemini_api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or GEMINI_API_KEY
        )

        if not self.nvidia_api_key and not self.gemini_api_key:
            logger.warning(
                "No LLM API keys found in active environmental context!"
            )
        else:
            logger.info(
                "NVIDIA NIM / Gemini client initialized successfully"
            )

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 300,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """
        Send a chat completion request to NVIDIA NIM.

        If NVIDIA is unavailable and Gemini credentials exist,
        transparently fall back to Gemini.
        """

        # Refresh credentials from environment.
        self.nvidia_api_key = (
            os.getenv("NVIDIA_API_KEY") or self.nvidia_api_key
        )

        self.gemini_api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or self.gemini_api_key
        )

        # If NVIDIA is not configured, use Gemini directly.
        if not self.nvidia_api_key:
            if self.gemini_api_key:
                logger.info(
                    "NVIDIA_API_KEY not configured; using Gemini fallback."
                )
                return self._chat_gemini(messages, temperature)

            raise RuntimeError(
                "No LLM API key configured. "
                "Set NVIDIA_API_KEY or GEMINI_API_KEY."
            )

        headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,

            # Nemotron supports thinking controls through this field.
            # Disable thinking because our pipeline expects clean,
            # machine-readable responses from planning calls.
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }
        if extra_body:
            payload.update(extra_body)

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    "NVIDIA NIM request: model=%s attempt=%d/%d",
                    model,
                    attempt + 1,
                    self.max_retries,
                )

                response = requests.post(
                    NIM_BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )

                response.raise_for_status()

                data = response.json()
                logger.info(
                    "NVIDIA NIM response metadata: %s",
                    self._summarize_nim_response(data),
                )

                choices = data.get("choices") or []
                if not choices:
                    raise ValueError(
                        "NVIDIA NIM returned no choices."
                    )

                choice = choices[0]
                message = choice.get("message") or {}

                # Normally this is the generated answer.
                content = message.get("content")

                # Some NVIDIA/Nemotron responses may expose reasoning
                # separately. We deliberately do not use reasoning_content
                # as the main answer.
                if content is None:
                    content = ""

                content = str(content).strip()

                if not content:
                    finish_reason = (
                        choice.get("finish_reason")
                        or choice.get("stop_reason")
                        or choice.get("matched_stop")
                    )
                    raise NvidiaEmptyResponseError(finish_reason)

                logger.info(
                    "NVIDIA NIM response received successfully "
                    "(%d characters)",
                    len(content),
                )

                return content

            except requests.HTTPError as exc:
                status = (
                    exc.response.status_code
                    if exc.response is not None
                    else 0
                )

                logger.warning(
                    "NVIDIA NIM attempt %d failed (%s): %s",
                    attempt + 1,
                    status,
                    exc,
                )

                # Authorization failure -> Gemini fallback if available.
                if status in (401, 403, 404) and self.gemini_api_key:
                    logger.info(
                        "NVIDIA NIM authorization failed. "
                        "Using Gemini fallback."
                    )
                    return self._chat_gemini(
                        messages,
                        temperature,
                    )

                if attempt < self.max_retries - 1:
                    backoff = (
                        15
                        if status in (403, 429, 502, 503)
                        else 2**attempt
                    )
                    time.sleep(backoff)
                else:
                    raise

            except Exception as exc:
                logger.warning(
                    "NVIDIA NIM attempt %d failed: %s",
                    attempt + 1,
                    exc,
                )

                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

        return ""

    def _chat_gemini(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> str:
        """Route requests to Google Gemini as a fallback."""

        logger.info(
            "Routing request to Google Gemini "
            "(gemini-2.5-flash)..."
        )

        if not self.gemini_api_key:
            raise RuntimeError(
                "Gemini fallback requested but no Gemini API key is configured."
            )

        genai.configure(api_key=self.gemini_api_key)

        system_instruction = ""
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction += content + "\n"
            else:
                gemini_role = (
                    "user"
                    if role == "user"
                    else "model"
                )

                contents.append(
                    {
                        "role": gemini_role,
                        "parts": [content],
                    }
                )

        system_instruction = system_instruction.strip()

        for attempt in range(self.max_retries):
            try:
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    system_instruction=(
                        system_instruction
                        if system_instruction
                        else None
                    ),
                )

                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                )

                response = model.generate_content(
                    contents,
                    generation_config=generation_config,
                )

                text = (response.text or "").strip()

                if text:
                    return text

                raise ValueError(
                    "Received empty content from Gemini API."
                )

            except Exception as exc:
                logger.warning(
                    "Gemini attempt %d failed: %s",
                    attempt + 1,
                    exc,
                )

                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

        return ""

    @staticmethod
    def _extract_json(text: str) -> Any:
        """
        Extract JSON from an LLM response.

        Handles:
        - pure JSON
        - ```json ... ```
        - ``` ... ```
        - explanatory text before/after JSON
        - JSON arrays
        - JSON objects
        """

        if not text:
            raise ValueError(
                "LLM returned an empty response while JSON was expected."
            )

        cleaned = text.strip()

        # Strip reasoning wrappers emitted by some models before parsing.
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = cleaned.replace("</think>", "")

        # Remove common markdown fences.
        cleaned = re.sub(
            r"```json\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"```\s*$",
            "",
            cleaned,
        )

        cleaned = cleaned.strip()

        first_nonspace = cleaned[:1]

        # First attempt: response is already valid JSON.
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # If the payload is array-shaped, keep it array-shaped.
        # Do not fall back to object extraction, which can accidentally
        # return the first element of a top-level JSON array.
        if first_nonspace == "[":
            decoder = json.JSONDecoder()
            try:
                parsed, _ = decoder.raw_decode(cleaned)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                raise ValueError(
                    "LLM response looked like a JSON array but could not be parsed.\n"
                    f"Response preview:\n{cleaned[:1000]}"
                )

        # Second attempt:
        # Locate the first JSON object or array and use JSONDecoder
        # to parse exactly that structure.
        decoder = json.JSONDecoder()

        candidates = []

        object_start = cleaned.find("{")
        array_start = cleaned.find("[")

        if object_start >= 0:
            candidates.append(object_start)

        if array_start >= 0:
            candidates.append(array_start)

        for start in sorted(candidates):
            try:
                parsed, _ = decoder.raw_decode(cleaned[start:])
                return parsed
            except json.JSONDecodeError:
                continue

        # Nothing could be parsed.
        preview = cleaned[:1000]

        raise ValueError(
            "LLM response did not contain valid JSON.\n"
            f"Response preview:\n{preview}"
        )

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        """
        Chat and robustly parse the response as JSON.

        The model may return markdown or explanatory text around
        the JSON. _extract_json() isolates the actual JSON payload.
        """

        text = self.chat(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )

        logger.info(
            "Parsing LLM JSON response (%d characters)",
            len(text),
        )

        preview = text[:1000]
        if len(text) > 1000:
            preview += "..."
        logger.info("LLM raw text preview: %s", preview)

        parsed = self._extract_json(text)

        if isinstance(parsed, dict):
            logger.info(
                "LLM parsed JSON shape: type=dict keys=%s",
                list(parsed.keys())[:10],
            )
        elif isinstance(parsed, list):
            logger.info(
                "LLM parsed JSON shape: type=list len=%d",
                len(parsed),
            )
        else:
            logger.info(
                "LLM parsed JSON shape: type=%s",
                type(parsed).__name__,
            )

        return parsed

    @staticmethod
    def _summarize_nim_response(data: Any) -> dict[str, Any]:
        """Summarize safe NVIDIA response metadata for truncation diagnosis."""
        summary: dict[str, Any] = {"type": type(data).__name__}

        if not isinstance(data, dict):
            return summary

        summary["keys"] = list(data.keys())[:10]

        usage = data.get("usage")
        if isinstance(usage, dict):
            summary["usage"] = {
                key: usage.get(key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if key in usage
            }

        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
            summary["finish_reason"] = (
                choice.get("finish_reason")
                or choice.get("stop_reason")
                or choice.get("matched_stop")
            )

            message = choice.get("message") or {}
        if isinstance(message, dict):
                summary["message_keys"] = list(message.keys())[:10]
                content = message.get("content")
                if content is not None:
                    summary["content_chars"] = len(str(content))
                reasoning_content = message.get("reasoning_content")
                if reasoning_content is not None:
                    summary["reasoning_chars"] = len(str(reasoning_content))

        return summary
