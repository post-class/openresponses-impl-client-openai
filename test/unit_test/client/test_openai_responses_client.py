"""Unit tests for OpenAIResponsesClient class."""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src directory to import path (for pytest compatibility)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from openresponses_impl_core.models.openresponses_models import CreateResponseBody, ResponseResource

from openresponses_impl_client_openai.client.openai_responses_client import OpenAIResponsesClient
from test.unit_test.helpers.response_payloads import build_response_resource_payload


class TestOpenAIResponsesClientInit:
    """Tests for OpenAIResponsesClient initialization"""

    def test_init_openai_success(self) -> None:
        """Verify successful initialization with OpenAI vendor"""
        # Act
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )

        # Assert
        assert client._provider == "openai"
        assert client._model == "gpt-4"
        assert client._api_key == "test-api-key"
        assert client._client is not None

    def test_init_azure_success(self) -> None:
        """Verify successful initialization with Azure vendor"""
        # Act
        client = OpenAIResponsesClient(
            vendor="azure",
            model="gpt-4",
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_api_key="test-azure-key",
            azure_openai_api_version="2024-10-21",
        )

        # Assert
        assert client._provider == "azure"
        assert client._model == "gpt-4"
        assert client._azure_endpoint == "https://test.openai.azure.com"
        assert client._azure_api_key == "test-azure-key"
        assert client._azure_api_version == "2024-10-21"
        assert client._client is not None

    def test_init_missing_model(self) -> None:
        """Verify ValueError is raised when model is empty"""
        # Act & Assert
        with pytest.raises(ValueError, match="model is required"):
            OpenAIResponsesClient(
                vendor="openai",
                model="",
                openai_api_key="test-api-key",
            )

    def test_init_openai_missing_api_key(self) -> None:
        """Verify ValueError is raised when api_key is not specified for OpenAI vendor"""
        # Act & Assert
        with pytest.raises(ValueError, match="openai_api_key is required"):
            OpenAIResponsesClient(
                vendor="openai",
                model="gpt-4",
                openai_api_key=None,
            )

    def test_init_azure_missing_endpoint(self) -> None:
        """Verify ValueError is raised when endpoint is not specified for Azure vendor"""
        # Act & Assert
        with pytest.raises(
            ValueError, match="azure_openai_endpoint and azure_openai_api_key are required"
        ):
            OpenAIResponsesClient(
                vendor="azure",
                model="gpt-4",
                azure_openai_endpoint=None,
                azure_openai_api_key="test-key",
            )

    def test_init_azure_missing_api_key(self) -> None:
        """Verify ValueError is raised when api_key is not specified for Azure vendor"""
        # Act & Assert
        with pytest.raises(
            ValueError, match="azure_openai_endpoint and azure_openai_api_key are required"
        ):
            OpenAIResponsesClient(
                vendor="azure",
                model="gpt-4",
                azure_openai_endpoint="https://test.openai.azure.com",
                azure_openai_api_key=None,
            )


class TestOpenAIResponsesClientCreateClient:
    """Tests for _create_client method"""

    def test_create_client_openai(self) -> None:
        """Verify OpenAI client creation"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )

        # Act
        openai_client = client._client

        # Assert
        assert openai_client is not None
        assert openai_client.api_key == "test-api-key"

    def test_create_client_azure(self) -> None:
        """Verify Azure OpenAI client creation"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="azure",
            model="gpt-4",
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_api_key="test-azure-key",
            azure_openai_api_version="2024-10-21",
        )

        # Act
        openai_client = client._client

        # Assert
        assert openai_client is not None
        assert "https://test.openai.azure.com/openai" in str(openai_client.base_url)

    def test_create_client_unsupported_provider(self) -> None:
        """Verify ValueError is raised for unsupported provider"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )
        client._provider = "unsupported"  # type: ignore

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported provider"):
            client._create_client()


class TestOpenAIResponsesClientBuildCreateKwargs:
    """Tests for _build_create_kwargs method"""

    def test_build_create_kwargs_basic(self) -> None:
        """Verify basic request parameter construction"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )
        payload = CreateResponseBody.model_validate(
            {
                "model": "gpt-3.5-turbo",  # This value is ignored and instance model is used
                "instructions": "Test instructions",
            }
        )

        # Act
        kwargs = client._build_create_kwargs(
            client=client._client,
            payload=payload,
            extra_params=None,
        )

        # Assert
        assert kwargs["model"] == "gpt-4"  # Instance model is used
        assert kwargs["instructions"] == "Test instructions"
        assert "extra_body" not in kwargs

    def test_build_create_kwargs_with_stream_false(self) -> None:
        """Verify stream_options is removed when stream=False"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )
        payload = CreateResponseBody.model_validate(
            {
                "model": "gpt-4",
                "instructions": "Test",
                "stream": False,
                "stream_options": {"include_obfuscation": True},
            }
        )

        # Act
        kwargs = client._build_create_kwargs(
            client=client._client,
            payload=payload,
            extra_params=None,
        )

        # Assert
        assert "stream_options" not in kwargs
        assert kwargs.get("stream") is False

    def test_build_create_kwargs_with_stream_true(self) -> None:
        """Verify stream_options is retained when stream=True"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )
        payload = CreateResponseBody.model_validate(
            {
                "model": "gpt-4",
                "instructions": "Test",
                "stream": True,
                "stream_options": {"include_obfuscation": False},
            }
        )

        # Act
        kwargs = client._build_create_kwargs(
            client=client._client,
            payload=payload,
            extra_params=None,
        )

        # Assert
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_obfuscation": False}

    def test_build_create_kwargs_with_extra_params(self) -> None:
        """Verify extra_params is merged into extra_body"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )
        payload = CreateResponseBody.model_validate(
            {
                "model": "gpt-4",
                "instructions": "Test",
            }
        )
        extra_params = {"custom_field": "custom_value"}

        # Act
        kwargs = client._build_create_kwargs(
            client=client._client,
            payload=payload,
            extra_params=extra_params,
        )

        # Assert
        assert "extra_body" in kwargs
        assert kwargs["extra_body"]["custom_field"] == "custom_value"

    def test_build_create_kwargs_merge_extra_body(self) -> None:
        """Verify existing extra_body and extra_params are correctly merged"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )
        # extra_body field does not exist, so add custom field directly
        payload_dict = {
            "model": "gpt-4",
            "instructions": "Test",
            "presence_penalty": 0.5,
        }
        payload = CreateResponseBody.model_validate(payload_dict)
        extra_params = {"new_field": "new_value"}

        # Act
        kwargs = client._build_create_kwargs(
            client=client._client,
            payload=payload,
            extra_params=extra_params,
        )

        # Assert
        # Unsupported fields and extra_params are merged into extra_body
        assert "extra_body" in kwargs
        assert kwargs["extra_body"]["presence_penalty"] == 0.5
        assert kwargs["extra_body"]["new_field"] == "new_value"


class TestOpenAIResponsesClientCreateResponse:
    """Tests for create_response method"""

    @pytest.mark.asyncio
    async def test_create_response_non_stream(self) -> None:
        """Verify response generation in non-streaming mode"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )
        payload = CreateResponseBody.model_validate(
            {
                "model": "gpt-4",
                "instructions": "Test",
                "stream": False,
            }
        )

        mock_response = build_response_resource_payload()

        with patch.object(
            client._client.responses, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = MagicMock(model_dump=lambda mode: mock_response)

            # Act
            result = await client.create_response(payload=payload)

            # Assert
            assert isinstance(result, ResponseResource)
            assert result.id == "resp_123"
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_response_stream(self) -> None:
        """Verify response generation in streaming mode"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )
        payload = CreateResponseBody.model_validate(
            {
                "model": "gpt-4",
                "instructions": "Test",
                "stream": True,
            }
        )

        # Act
        result = await client.create_response(payload=payload)

        # Assert
        # AsyncIterator is returned for streaming
        assert hasattr(result, "__aiter__")


class TestOpenAIResponsesClientCloseStream:
    """Tests for _close_stream method"""

    @pytest.mark.asyncio
    async def test_close_stream_with_aclose(self) -> None:
        """Verify closing stream with aclose method"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )
        mock_stream = MagicMock()
        mock_stream.aclose = AsyncMock()

        # Act
        await client._close_stream(mock_stream)

        # Assert
        mock_stream.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_stream_with_close(self) -> None:
        """Verify closing stream with close method"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )
        mock_stream = MagicMock()
        mock_stream.close = MagicMock()
        del mock_stream.aclose  # Remove aclose method

        # Act
        await client._close_stream(mock_stream)

        # Assert
        mock_stream.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_stream_without_close_methods(self) -> None:
        """Verify handling of stream without close methods"""
        # Arrange
        client = OpenAIResponsesClient(
            vendor="openai",
            model="gpt-4",
            openai_api_key="test-api-key",
        )
        mock_stream = MagicMock(spec=[])  # Has neither aclose nor close

        # Act & Assert (verify no error occurs)
        await client._close_stream(mock_stream)
