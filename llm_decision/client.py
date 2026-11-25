"""
DeepSeek API client for multi-asset allocation.
"""

import json
import os
import time

from openai import APIConnectionError, APIStatusError, OpenAI


MODEL_NAME = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 60
MAX_OUTPUT_TOKENS = 3072


class DeepSeekTruncationError(RuntimeError):
    """DeepSeek reached the output or context limit before finishing."""


class DeepSeekClient:

    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY environment variable not found."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )

    def generate(
        self,
        system_prompt: str,
        model_input: dict,
    ) -> tuple[dict, dict]:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("System prompt cannot be empty.")

        if not isinstance(model_input, dict):
            raise TypeError("Model input must be a dictionary.")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    model_input,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            },
        ]

        for attempt in range(1, MAX_RETRIES + 1):
            start_time = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    extra_body={"thinking": {"type": "disabled"}},
                    stream=False,
                )
            except APIStatusError as exc:
                status = exc.status_code
                failure = f"HTTP {status}"
                if status == 402:
                    raise RuntimeError(
                        "DeepSeek API returned HTTP 402 (insufficient balance)."
                    ) from None
                if status not in {408, 429} and not 500 <= status < 600:
                    raise RuntimeError(
                        f"DeepSeek API request failed ({failure})."
                    ) from None
            except APIConnectionError as exc:
                failure = type(exc).__name__
            else:
                break

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"DeepSeek API failed after {attempt} attempts ({failure})."
                )
            time.sleep(2 ** (attempt - 1))

        elapsed = time.perf_counter() - start_time
        if not response.choices:
            raise RuntimeError("DeepSeek API returned no choices.")

        choice = response.choices[0]
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if choice.finish_reason == "length":
            raise DeepSeekTruncationError(
                "DeepSeek output was truncated (finish_reason=length, "
                f"max_tokens={MAX_OUTPUT_TOKENS}, prompt_tokens={prompt_tokens}, "
                f"completion_tokens={completion_tokens}, attempt={attempt}). "
                "The output or context limit was reached."
            )
        if choice.finish_reason != "stop":
            raise RuntimeError(
                f"DeepSeek output did not finish normally (finish_reason={choice.finish_reason})."
            )

        content = choice.message.content
        if not content or not content.strip():
            raise ValueError("DeepSeek returned empty JSON content.")

        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"DeepSeek returned invalid JSON at line {exc.lineno}, column {exc.colno}."
            ) from None

        if not isinstance(result, dict):
            raise ValueError("DeepSeek response must be a JSON object.")

        metadata = {
            "model": response.model,
            "finish_reason": choice.finish_reason,
            "attempt": attempt,
            "elapsed_seconds": round(elapsed, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        return result, metadata
