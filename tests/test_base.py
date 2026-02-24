import unittest
from typing import Any

from base import ISP_REGISTRY, ISPStep, register_step


class TestBase(unittest.TestCase):
    def test_isp_step_enum_str_conversion(self):
        """Verify that the Enum returns its value when cast to string."""
        step = ISPStep.BLACK_LEVEL_SUBTRACTION
        assert str(step) == "black_level_subtraction"
        # Verify it works with f-strings (common in your logging)
        assert f"{step}" == "black_level_subtraction"

    def test_register_step_decorator(self):
        """Verify the decorator correctly populates the ISP_REGISTRY."""
        test_step = ISPStep.NORMALIZATION

        # Ensure it's clean before testing (optional but safer)
        if test_step in ISP_REGISTRY:
            del ISP_REGISTRY[test_step]

        @register_step(test_step)
        def mock_normalization_func(img: Any, metadata: Any) -> Any:
            return "success"

        # 1. Check if the step is in the registry
        assert test_step in ISP_REGISTRY

        # 2. Check if the registered value is the actual function
        assert ISP_REGISTRY[test_step] == mock_normalization_func

        # 3. Verify the function still works as expected (decorator shouldn't break it)
        result = mock_normalization_func(None, None)
        assert result == "success"

    def test_enum_uniqueness(self):
        """Ensure no two enum members have the same value."""
        values = [member.value for member in ISPStep]
        assert len(values) == len(set(values)), "Duplicate values found in ISPStep enum"
