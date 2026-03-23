"""Unit tests for CopyUtil class."""

from __future__ import annotations

import os
import sys
from typing import Any

from pydantic import BaseModel

# Add src directory to import path (for pytest compatibility)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from openresponses_impl_client_openai.utils.copy_util import CopyUtil


class SampleModel(BaseModel):
    """Pydantic model for testing"""

    name: str
    data: dict[str, Any]


class TestCopyUtil:
    """Tests for CopyUtil class"""

    def test_deep_copy_dict(self) -> None:
        """Verify deep copy of dictionary works correctly"""
        # Arrange
        original = {"key": {"nested": "value"}}

        # Act
        copied = CopyUtil.deep_copy(original)
        copied["key"]["nested"] = "modified"

        # Assert
        assert original["key"]["nested"] == "value"
        assert copied["key"]["nested"] == "modified"
        assert original is not copied
        assert original["key"] is not copied["key"]

    def test_deep_copy_list(self) -> None:
        """Verify deep copy of list works correctly"""
        # Arrange
        original = [1, 2, [3, 4]]

        # Act
        copied = CopyUtil.deep_copy(original)
        copied[2][0] = 999

        # Assert
        assert original[2][0] == 3
        assert copied[2][0] == 999
        assert original is not copied
        assert original[2] is not copied[2]

    def test_deep_copy_pydantic_model(self) -> None:
        """Verify deep copy of Pydantic model works correctly"""
        # Arrange
        original = SampleModel(name="test", data={"nested": "value"})

        # Act
        copied = CopyUtil.deep_copy(original)
        copied.data["nested"] = "modified"
        copied.name = "changed"

        # Assert
        assert original.name == "test"
        assert original.data["nested"] == "value"
        assert copied.name == "changed"
        assert copied.data["nested"] == "modified"
        assert original is not copied
        assert original.data is not copied.data

    def test_deep_copy_nested_pydantic_model(self) -> None:
        """Verify deep copy of nested Pydantic model works correctly"""

        # Arrange
        class InnerModel(BaseModel):
            value: str

        class OuterModel(BaseModel):
            inner: InnerModel
            items: list[str]

        original = OuterModel(inner=InnerModel(value="original"), items=["a", "b"])

        # Act
        copied = CopyUtil.deep_copy(original)
        copied.inner.value = "modified"
        copied.items.append("c")

        # Assert
        assert original.inner.value == "original"
        assert len(original.items) == 2
        assert copied.inner.value == "modified"
        assert len(copied.items) == 3
        assert original is not copied
        assert original.inner is not copied.inner
        assert original.items is not copied.items

    def test_deep_copy_complex_structure(self) -> None:
        """Verify deep copy of complex data structure works correctly"""
        # Arrange
        original = {
            "string": "value",
            "number": 42,
            "list": [1, 2, {"nested": "dict"}],
            "dict": {"key": ["nested", "list"]},
        }

        # Act
        copied = CopyUtil.deep_copy(original)
        copied["list"][2]["nested"] = "modified"
        copied["dict"]["key"].append("added")

        # Assert
        assert original["list"][2]["nested"] == "dict"
        assert len(original["dict"]["key"]) == 2
        assert copied["list"][2]["nested"] == "modified"
        assert len(copied["dict"]["key"]) == 3
        assert original is not copied

    def test_deep_copy_empty_dict(self) -> None:
        """Verify deep copy of empty dictionary works correctly"""
        # Arrange
        original: dict[str, Any] = {}

        # Act
        copied = CopyUtil.deep_copy(original)
        copied["new_key"] = "new_value"

        # Assert
        assert len(original) == 0
        assert len(copied) == 1
        assert original is not copied

    def test_deep_copy_empty_list(self) -> None:
        """Verify deep copy of empty list works correctly"""
        # Arrange
        original: list[Any] = []

        # Act
        copied = CopyUtil.deep_copy(original)
        copied.append("item")

        # Assert
        assert len(original) == 0
        assert len(copied) == 1
        assert original is not copied

    def test_deep_copy_primitive_types(self) -> None:
        """Verify deep copy of primitive types works correctly"""
        # Arrange & Act & Assert
        assert CopyUtil.deep_copy(42) == 42
        assert CopyUtil.deep_copy("string") == "string"
        assert CopyUtil.deep_copy(3.14) == 3.14
        assert CopyUtil.deep_copy(True) is True
        assert CopyUtil.deep_copy(None) is None
