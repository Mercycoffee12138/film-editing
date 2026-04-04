from __future__ import annotations

import base64
import json
import mimetypes
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "https://api.zhizengzeng.com/alibaba"
DEFAULT_MODEL = "qwen-vl-max-latest"
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0


class QwenVisionContentBlockedError(RuntimeError):
    pass


@dataclass(frozen=True)
class QwenVisionConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS


def load_config_from_env() -> QwenVisionConfig | None:
    api_key = (
        os.getenv("ZZZ_API_KEY")
        or os.getenv("ZHIZENGZENG_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    if not api_key:
        return None

    base_url = os.getenv("ZZZ_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL)
    timeout_seconds = int(os.getenv("QWEN_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    max_retries = int(os.getenv("QWEN_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
    retry_backoff_seconds = float(
        os.getenv("QWEN_RETRY_BACKOFF_SECONDS", str(DEFAULT_RETRY_BACKOFF_SECONDS))
    )
    return QwenVisionConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max(1, max_retries),
        retry_backoff_seconds=max(0.0, retry_backoff_seconds),
    )


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def _path_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_guess_mime_type(path)};base64,{encoded}"


def build_payload(image_paths: list[Path], prompt: str, model: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"image": _path_to_data_url(path)} for path in image_paths]
    content.append({"text": prompt})
    return {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        },
        "parameters": {
            "result_format": "message",
        },
    }


def _endpoint_url(config: QwenVisionConfig) -> str:
    return f"{config.base_url}/api/v1/services/aigc/multimodal-generation/generation"


def _extract_text(payload: dict[str, Any]) -> str:
    code = str(payload.get("code") or "").strip()
    message = str(payload.get("message") or "").strip()
    if code == "DataInspectionFailed" or "DataInspectionFailed" in message:
        raise QwenVisionContentBlockedError(
            f"Qwen vision request was blocked by content inspection: {message or code}"
        )

    output = payload.get("output") or {}
    choices = output.get("choices") or []
    for choice in choices:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                text = item.get("text")
                if text:
                    text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)
    if code or message:
        raise RuntimeError(
            f"Qwen vision response did not include text. code={code or 'unknown'}, message={message or 'unknown'}"
        )
    raise RuntimeError(f"No text found in vision response: {json.dumps(payload, ensure_ascii=False)}")


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def _is_retryable_url_reason(reason: Any) -> bool:
    if isinstance(reason, str):
        normalized = reason.lower()
        return any(
            token in normalized
            for token in (
                "timed out",
                "timeout",
                "temporarily unavailable",
                "connection reset",
                "connection aborted",
                "connection refused",
                "network is unreachable",
                "remote end closed connection",
            )
        )
    return isinstance(
        reason,
        (
            socket.timeout,
            TimeoutError,
            ConnectionResetError,
            ConnectionAbortedError,
            ConnectionRefusedError,
            ConnectionError,
            OSError,
        ),
    )


def _retry_delay_seconds(config: QwenVisionConfig, attempt: int) -> float:
    if config.retry_backoff_seconds <= 0.0:
        return 0.0
    return config.retry_backoff_seconds * (2 ** max(attempt - 1, 0))


def analyze_images(
    image_paths: list[Path],
    prompt: str,
    config: QwenVisionConfig,
) -> dict[str, Any]:
    payload = build_payload(image_paths=image_paths, prompt=prompt, model=config.model)
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        _endpoint_url(config),
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )

    last_error: RuntimeError | None = None
    for attempt in range(1, max(config.max_retries, 1) + 1):
        try:
            with request.urlopen(http_request, timeout=config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            return {
                "text": _extract_text(parsed),
                "raw_response": parsed,
                "request_url": _endpoint_url(config),
            }
        except socket.timeout as exc:
            last_error = RuntimeError(
                f"Qwen vision request timed out after {config.timeout_seconds}s "
                f"(attempt {attempt}/{config.max_retries})."
            )
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            if not _is_retryable_http_status(exc.code) or attempt >= config.max_retries:
                raise RuntimeError(f"Qwen vision request failed with HTTP {exc.code}: {details}") from exc
            last_error = RuntimeError(
                f"Qwen vision request failed with HTTP {exc.code} "
                f"(attempt {attempt}/{config.max_retries}): {details}"
            )
        except error.URLError as exc:
            if not _is_retryable_url_reason(exc.reason) or attempt >= config.max_retries:
                raise RuntimeError(f"Qwen vision request failed: {exc.reason}") from exc
            last_error = RuntimeError(
                f"Qwen vision request failed: {exc.reason} "
                f"(attempt {attempt}/{config.max_retries})"
            )

        delay_seconds = _retry_delay_seconds(config, attempt)
        if attempt < config.max_retries and delay_seconds > 0.0:
            print(
                f"Qwen vision transient error, retrying in {delay_seconds:.1f}s "
                f"(attempt {attempt + 1}/{config.max_retries}): {last_error}",
                flush=True,
            )
            time.sleep(delay_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Qwen vision request failed for an unknown reason.")
