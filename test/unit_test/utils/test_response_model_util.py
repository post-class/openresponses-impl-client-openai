"""Unit tests for OpenAIResponseModelUtil class."""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

# Add src directory to import path (for pytest compatibility)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from openresponses_impl_core.models.openresponses_models import (
    ErrorPayload,
    ErrorStreamingEvent,
    ResponseResource,
)

from openresponses_impl_client_openai.utils.openai_response_model_util import (
    OpenAIResponseModelUtil,
)
from test.unit_test.helpers.response_payloads import (
    build_response_resource_payload,
    build_response_stream_event_payload,
)


class TestOpenAIResponseModelUtil:
    """Tests for OpenAIResponseModelUtil class"""

    def test_parse_response_with_response_resource(self) -> None:
        """Verify ResponseResource is returned as-is when already a ResponseResource"""
        # Arrange
        response = ResponseResource.model_validate(build_response_resource_payload())

        # Act
        result = OpenAIResponseModelUtil.parse_response(payload=response)

        # Assert
        assert result is response
        assert result.id == "resp_123"

    def test_parse_response_with_dict(self) -> None:
        """Verify conversion from dictionary to ResponseResource works correctly"""
        # Arrange
        payload = build_response_resource_payload(overrides={"id": "resp_456"})

        # Act
        result = OpenAIResponseModelUtil.parse_response(payload=payload)

        # Assert
        assert isinstance(result, ResponseResource)
        assert result.id == "resp_456"
        assert result.model == "gpt-4"

    def test_parse_response_with_pydantic_model(self) -> None:
        """Verify conversion from Pydantic model works correctly"""

        # Arrange
        class CustomModel(BaseModel):
            model_config = ConfigDict(extra="allow")

            id: str
            object: str
            created_at: int
            model: str
            status: str

        payload = CustomModel(
            **build_response_resource_payload(
                overrides={
                    "id": "resp_789",
                    "completed_at": 1234567999,
                }
            )
        )

        # Act
        result = OpenAIResponseModelUtil.parse_response(payload=payload)

        # Assert
        assert isinstance(result, ResponseResource)
        assert result.id == "resp_789"

    def test_parse_response_with_invalid_payload_type(self) -> None:
        """Verify ValueError is raised for invalid payload type"""
        # Arrange
        payload = "invalid_string"

        # Act & Assert
        with pytest.raises(ValueError, match="payload must be a dict or model"):
            OpenAIResponseModelUtil.parse_response(payload=payload)

    def test_parse_response_with_invalid_data(self) -> None:
        """Verify ValidationError is raised for invalid data"""
        # Arrange
        payload = {"invalid_field": "value"}

        # Act & Assert
        with pytest.raises(ValidationError):
            OpenAIResponseModelUtil.parse_response(payload=payload)

    def test_parse_stream_event_with_valid_dict(self) -> None:
        """Verify conversion of valid streaming event works correctly"""
        # Arrange
        payload = build_response_stream_event_payload(event_type="response.completed")

        # Act
        result = OpenAIResponseModelUtil.parse_stream_event(payload=payload)

        # Assert
        assert result.type == "response.completed"
        assert result.sequence_number == 1

    def test_parse_stream_event_with_pydantic_model(self) -> None:
        """Verify streaming event conversion from Pydantic model works correctly"""

        # Arrange
        class StreamEventModel(BaseModel):
            model_config = ConfigDict(extra="allow")

            type: str
            sequence_number: int
            response: dict[str, Any]

        payload = StreamEventModel(
            **build_response_stream_event_payload(
                event_type="response.completed",
                overrides={"sequence_number": 2},
                response_overrides={"id": "resp_456"},
            )
        )

        # Act
        result = OpenAIResponseModelUtil.parse_stream_event(payload=payload)

        # Assert
        assert result.type == "response.completed"
        assert result.sequence_number == 2

    def test_parse_stream_event_with_non_dict_payload(self) -> None:
        """Verify ErrorStreamingEvent is returned for non-dictionary payload"""
        # Arrange
        payload = "invalid_string"

        # Act
        result = OpenAIResponseModelUtil.parse_stream_event(payload=payload)

        # Assert
        assert isinstance(result, ErrorStreamingEvent)
        assert result.type == "error"
        assert result.sequence_number == 0
        assert "unsupported streaming event payload" in result.error.message

    def test_parse_stream_event_with_validation_error(self) -> None:
        """Verify ErrorStreamingEvent is returned on validation error"""
        # Arrange
        payload = {"invalid_field": "value", "sequence_number": 5}

        # Act
        result = OpenAIResponseModelUtil.parse_stream_event(payload=payload)

        # Assert
        assert isinstance(result, ErrorStreamingEvent)
        assert result.type == "error"
        assert result.sequence_number == 5
        assert "invalid streaming event" in result.error.message

    def test_normalize_payload_with_dict(self) -> None:
        """Verify dictionary normalization works correctly"""
        # Arrange
        payload = {"key": "value"}

        # Act
        result = OpenAIResponseModelUtil._normalize_payload(payload=payload)

        # Assert
        assert result == {"key": "value"}
        assert isinstance(result, dict)

    def test_normalize_payload_with_pydantic_model(self) -> None:
        """Verify Pydantic model normalization works correctly"""

        # Arrange
        class TestModel(BaseModel):
            name: str
            value: int

        payload = TestModel(name="test", value=42)

        # Act
        result = OpenAIResponseModelUtil._normalize_payload(payload=payload)

        # Assert
        assert result == {"name": "test", "value": 42}
        assert isinstance(result, dict)

    def test_normalize_payload_with_invalid_type_strict(self) -> None:
        """Verify ValueError is raised for invalid type when allow_non_dict=False"""
        # Arrange
        payload = "invalid_string"

        # Act & Assert
        with pytest.raises(ValueError, match="payload must be a dict or model"):
            OpenAIResponseModelUtil._normalize_payload(payload=payload, allow_non_dict=False)

    def test_normalize_payload_with_invalid_type_allow_non_dict(self) -> None:
        """Verify invalid type is returned as-is when allow_non_dict=True"""
        # Arrange
        payload = "invalid_string"

        # Act
        result = OpenAIResponseModelUtil._normalize_payload(payload=payload, allow_non_dict=True)

        # Assert
        assert result == "invalid_string"

    def test_build_error_event_with_sequence_number(self) -> None:
        """Verify error event construction with sequence_number works correctly"""
        # Arrange
        payload = {"sequence_number": 10, "other": "data"}
        message = "Test error message"

        # Act
        result = OpenAIResponseModelUtil._build_error_event(payload=payload, message=message)

        # Assert
        assert isinstance(result, ErrorStreamingEvent)
        assert result.type == "error"
        assert result.sequence_number == 10
        assert result.error.type == "invalid_stream_event"
        assert result.error.message == "Test error message"
        assert result.error.code is None
        assert result.error.param is None

    def test_build_error_event_without_sequence_number(self) -> None:
        """Verify 0 is set when sequence_number is missing"""
        # Arrange
        payload: dict[str, Any] = {"other": "data"}
        message = "Test error message"

        # Act
        result = OpenAIResponseModelUtil._build_error_event(payload=payload, message=message)

        # Assert
        assert result.sequence_number == 0

    def test_build_error_event_with_invalid_sequence_number(self) -> None:
        """Verify 0 is set when sequence_number is not an integer"""
        # Arrange
        payload = {"sequence_number": "invalid", "other": "data"}
        message = "Test error message"

        # Act
        result = OpenAIResponseModelUtil._build_error_event(payload=payload, message=message)

        # Assert
        assert result.sequence_number == 0

    def test_build_error_event_error_payload_structure(self) -> None:
        """Verify error payload structure is correct"""
        # Arrange
        payload: dict[str, Any] = {}
        message = "Detailed error message"

        # Act
        result = OpenAIResponseModelUtil._build_error_event(payload=payload, message=message)

        # Assert
        assert isinstance(result.error, ErrorPayload)
        assert result.error.type == "invalid_stream_event"
        assert result.error.code is None
        assert result.error.message == "Detailed error message"
        assert result.error.param is None
        assert result.error.headers is None
