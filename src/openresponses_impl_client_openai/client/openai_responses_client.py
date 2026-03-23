from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any, Literal, override

from openai import AsyncOpenAI
from openresponses_impl_core.client.base_responses_client import BaseResponsesClient
from openresponses_impl_core.models.openresponses_models import (
    CreateResponseBody,
    ResponseResource,
)
from openresponses_impl_core.models.response_event_types import ResponseStreamingEvent

from openresponses_impl_client_openai.utils.copy_util import CopyUtil
from openresponses_impl_client_openai.utils.openai_response_model_util import (
    OpenAIResponseModelUtil,
)


class OpenAIResponsesClient(BaseResponsesClient):
    """Responses client for OpenAI/Azure OpenAI"""

    def __init__(
        self,
        *,
        vendor: Literal["openai", "azure"],
        model: str,
        openai_api_key: str | None = None,
        azure_openai_endpoint: str | None = None,
        azure_openai_api_key: str | None = None,
        azure_openai_api_version: str = "2024-10-21",
    ) -> None:
        """Responses client for OpenAI/Azure OpenAI

        Args:
            vendor: Specify "openai" or "azure"
            model: Model name (e.g., "gpt-4", "gpt-5-mini")
            openai_api_key: OpenAI API key (required when vendor="openai")
            azure_openai_endpoint: Azure OpenAI endpoint (required when vendor="azure")
            azure_openai_api_key: Azure OpenAI API key (required when vendor="azure")
            azure_openai_api_version: Azure OpenAI API version (default: "2024-10-21" when vendor="azure")

        Raises:
            ValueError: When required parameters are missing

        """
        self._provider = vendor
        self._model = model
        self._api_key = openai_api_key
        self._azure_endpoint = azure_openai_endpoint
        self._azure_api_key = azure_openai_api_key
        self._azure_api_version = azure_openai_api_version

        # Validation
        if not model:
            raise ValueError("model is required.")
        if vendor == "openai" and not openai_api_key:
            raise ValueError("openai_api_key is required when vendor='openai'.")
        if vendor == "azure":
            if not azure_openai_endpoint or not azure_openai_api_key:
                raise ValueError(
                    "azure_openai_endpoint and azure_openai_api_key are required when vendor='azure'."
                )

        # Create and store client
        self._client = self._create_client()

    @override
    async def create_response(
        self, payload: CreateResponseBody, **kwargs: Any
    ) -> ResponseResource | AsyncIterator[ResponseStreamingEvent]:
        """Create a response based on the stream field in the payload

        Args:
            payload: Request payload (CreateResponseBody type)
            **kwargs: Additional parameters (passed to extra_body)

        Returns:
            payload.stream=False/None: ResponseResource
            payload.stream=True: AsyncIterator[ResponseStreamingEvent]

        """
        # Branch based on payload.stream value
        if payload.stream:
            return await self._create_response_stream(payload=payload, extra_params=kwargs)
        return await self._create_response_non_stream(payload=payload, extra_params=kwargs)

    async def _create_response_non_stream(
        self, payload: CreateResponseBody, extra_params: dict[str, Any] | None = None
    ) -> ResponseResource:
        """Generate response in non-streaming mode"""
        request_payload = payload.model_copy(deep=True)
        # Use payload.stream value as-is (False or unset)

        request_kwargs = self._build_create_kwargs(
            client=self._client, payload=request_payload, extra_params=extra_params
        )
        response = await self._client.responses.create(**request_kwargs)
        return OpenAIResponseModelUtil.parse_response(payload=response)

    async def _create_response_stream(
        self, payload: CreateResponseBody, extra_params: dict[str, Any] | None = None
    ) -> AsyncIterator[ResponseStreamingEvent]:
        """Generate response in streaming mode"""
        request_payload = payload.model_copy(deep=True)
        # Use payload.stream value as-is (True)
        request_kwargs = self._build_create_kwargs(
            client=self._client, payload=request_payload, extra_params=extra_params
        )
        return self._iter_stream_events(client=self._client, request_kwargs=request_kwargs)

    def _build_create_kwargs(
        self,
        *,
        client: AsyncOpenAI,
        payload: CreateResponseBody,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build kwargs for OpenAI API request"""
        signature = inspect.signature(client.responses.create)
        supported_keys = set(signature.parameters.keys())

        # Convert CreateResponseBody to dictionary
        payload_dict = payload.model_dump(mode="json", exclude_none=True)

        request_kwargs: dict[str, Any] = {}
        passthrough_extra_body: dict[str, Any] = {}

        for key, value in payload_dict.items():
            if key in supported_keys:
                # Use instance variable model name for model key
                if key == "model":
                    request_kwargs[key] = self._model
                else:
                    request_kwargs[key] = value
                continue
            passthrough_extra_body[key] = value

        # Avoid provider-side validation error when stream is not enabled.
        if request_kwargs.get("stream") is not True:
            request_kwargs.pop("stream_options", None)

        # Merge extra_params into passthrough_extra_body if present
        if extra_params:
            passthrough_extra_body.update(extra_params)

        # Bundle fields in CreateResponseBody that don't exist in OpenAI API's official signature as extra_body parameter
        if passthrough_extra_body:
            existing_extra_body = request_kwargs.get("extra_body")
            if isinstance(existing_extra_body, dict):
                merged_extra_body = CopyUtil.deep_copy(existing_extra_body)
                merged_extra_body.update(passthrough_extra_body)
                request_kwargs["extra_body"] = merged_extra_body
            else:
                request_kwargs["extra_body"] = passthrough_extra_body

        return request_kwargs

    async def _close_stream(self, stream: Any) -> None:
        """Close the stream"""
        aclose_method = getattr(stream, "aclose", None)
        if callable(aclose_method):
            result = aclose_method()
            if inspect.isawaitable(result):
                await result
            return

        close_method = getattr(stream, "close", None)
        if callable(close_method):
            result = close_method()
            if inspect.isawaitable(result):
                await result

    async def _iter_stream_events(
        self, *, client: AsyncOpenAI, request_kwargs: dict[str, Any]
    ) -> AsyncIterator[ResponseStreamingEvent]:
        """Generate streaming events"""
        stream = await client.responses.create(**request_kwargs)
        try:
            async for event in stream:
                yield OpenAIResponseModelUtil.parse_stream_event(payload=event)
        finally:
            await self._close_stream(stream)

    def _create_client(self) -> AsyncOpenAI:
        """Create OpenAI/Azure OpenAI client"""
        if self._provider == "azure":
            # Create client for Azure OpenAI
            # Validation already performed in __init__
            assert self._azure_endpoint is not None
            assert self._azure_api_key is not None
            azure_endpoint = self._azure_endpoint.rstrip("/")
            return AsyncOpenAI(
                api_key="",
                base_url=f"{azure_endpoint}/openai",
                default_headers={"api-key": self._azure_api_key},
                default_query={"api-version": self._azure_api_version},
            )
        if self._provider == "openai":
            # Create client for OpenAI
            # Validation already performed in __init__
            assert self._api_key is not None
            return AsyncOpenAI(
                api_key=self._api_key,
            )
        raise ValueError(f"Unsupported provider: {self._provider}")
