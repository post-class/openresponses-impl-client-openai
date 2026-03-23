"""Helper builders for OpenResponses payloads used in unit tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_DEFAULT_RESPONSE_ID = "resp_123"
_DEFAULT_RESPONSE_CREATED_AT = 1_700_000_000


def build_response_resource_payload(*, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a ResponseResource-like payload that matches the latest schema."""
    payload: dict[str, Any] = {
        "id": _DEFAULT_RESPONSE_ID,
        "object": "response",
        "created_at": _DEFAULT_RESPONSE_CREATED_AT,
        "completed_at": _DEFAULT_RESPONSE_CREATED_AT + 10,
        "status": "completed",
        "incomplete_details": None,
        "model": "gpt-4",
        "previous_response_id": None,
        "instructions": "Test instructions",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, world!",
                    }
                ],
            }
        ],
        "error": None,
        "tools": [
            {
                "type": "function",
                "name": "tool_name",
                "description": "Example tool",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ],
        "tool_choice": "auto",
        "truncation": "auto",
        "parallel_tool_calls": True,
        "text": {
            "format": {"type": "text"},
            "verbosity": None,
        },
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "top_logprobs": 0,
        "temperature": 1.0,
        "reasoning": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "max_output_tokens": None,
        "max_tool_calls": None,
        "store": False,
        "background": False,
        "service_tier": "default",
        "metadata": {},
        "safety_identifier": None,
        "prompt_cache_key": None,
    }

    if overrides:
        payload.update(deepcopy(overrides))
    return payload


def build_response_stream_event_payload(
    *,
    event_type: str = "response.completed",
    overrides: dict[str, Any] | None = None,
    response_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a streaming event payload embedding a valid response payload."""
    event: dict[str, Any] = {
        "type": event_type,
        "sequence_number": 1,
        "response": build_response_resource_payload(overrides=response_overrides),
    }

    if overrides:
        event.update(deepcopy(overrides))
    return event
