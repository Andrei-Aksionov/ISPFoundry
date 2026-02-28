import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Sequence
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from base import ISPStep
from pipeline import ISPPipeline


class MockStep:
    """Simulated ISPStep enum for testing purposes."""

    STEP_A = "step_a"
    STEP_B = "step_b"


class TestISPPipeline(unittest.TestCase):
    def setUp(self):
        """Setup common data and path for all tests."""
        self.image_input = [
            np.array([1] * 4, dtype=np.float32),
            np.array([2] * 4, dtype=np.float32),
        ]
        self.metadata = [{"id": 101}, {"id": 102}]
        self.test_dir = Path(tempfile.mkdtemp())

        # Standard mock config used across tests
        self.mock_cfg = OmegaConf.create({"pipeline": {"default_steps": [MockStep.STEP_A, MockStep.STEP_B]}})

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def _get_fake_registry(self):
        """Helper to create a registry that handles both mock steps."""
        reg = {}

        def dummy_func(image_input, metadata, **kwargs):
            if isinstance(image_input, Sequence):
                return [ii * 2 for ii in image_input]
            return image_input * 2

        reg[MockStep.STEP_A] = MagicMock(side_effect=dummy_func)
        reg[MockStep.STEP_B] = MagicMock(side_effect=dummy_func)
        return reg

    def test_init_non_steps(self):
        with patch("pipeline.config", self.mock_cfg), patch("pipeline.pkgutil.iter_modules", return_value=[]):
            pipeline = ISPPipeline()
            assert len(pipeline.steps) == 2

    def test_init_empty_list_steps(self):
        with patch("pipeline.config", self.mock_cfg), patch("pipeline.pkgutil.iter_modules", return_value=[]):
            pipeline = ISPPipeline(steps=[])
            assert len(pipeline.steps) == 2

    def test_init_custom_list_steps(self):
        custom_steps = [ISPStep.BLACK_LEVEL_SUBTRACTION]
        with patch("pipeline.config", self.mock_cfg), patch("pipeline.pkgutil.iter_modules", return_value=[]):
            pipeline = ISPPipeline(steps=custom_steps)
            assert len(pipeline.steps) == 1
            assert pipeline.steps == custom_steps

    def test_discover_steps_logic(self):
        with patch("pipeline.pkgutil.iter_modules") as mock_iter:
            mock_iter.return_value = [(None, "test_module", False)]
            with patch("pipeline.importlib.import_module") as mock_import, patch("pipeline.config", self.mock_cfg):
                ISPPipeline()
                mock_import.assert_called_with("pipeline_steps.test_module")

    def test_pipeline_transformation_logic(self):
        fake_reg = self._get_fake_registry()
        with (
            patch("pipeline.config", self.mock_cfg),
            patch("pipeline.ISP_REGISTRY", fake_reg),
            patch("pipeline.pkgutil.iter_modules", return_value=[]),
        ):
            pipeline = ISPPipeline()
            results = pipeline.run(self.image_input, self.metadata)

            assert len(results) == 2
            np.testing.assert_equal(results[0], self.image_input[0] * len(pipeline.steps) * 2)
            np.testing.assert_equal(results[1], self.image_input[1] * len(pipeline.steps) * 2)
            assert fake_reg[MockStep.STEP_A].call_count == 1
            assert fake_reg[MockStep.STEP_B].call_count == 1

    def test_config_overrides_propagation(self):
        fake_reg = self._get_fake_registry()
        overrides = {MockStep.STEP_A: {"param_x": 50}}

        with (
            patch("pipeline.config", self.mock_cfg),
            patch("pipeline.ISP_REGISTRY", fake_reg),
            patch("pipeline.pkgutil.iter_modules", return_value=[]),
        ):
            pipeline = ISPPipeline()
            pipeline.run(self.image_input, self.metadata, config_overrides=overrides)  # ty:ignore[invalid-argument-type]

            _, kwargs = fake_reg[MockStep.STEP_A].call_args
            assert kwargs["param_x"] == 50

    def test_missing_step_implementation_raises_error(self):
        # Explicitly empty registry to trigger the error
        with (
            patch("pipeline.config", self.mock_cfg),
            patch("pipeline.ISP_REGISTRY", {}),
            patch("pipeline.pkgutil.iter_modules", return_value=[]),
        ):
            pipeline = ISPPipeline()
            with pytest.raises(ValueError, match="has no implementation"):
                pipeline.run(self.image_input, self.metadata)

    def test_file_saving_and_telemetry(self):
        fake_reg = self._get_fake_registry()
        with (
            patch("pipeline.config", self.mock_cfg),
            patch("pipeline.ISP_REGISTRY", fake_reg),
            patch("pipeline.save_ndarray_as_jpg") as mock_save,
            patch("pipeline.pkgutil.iter_modules", return_value=[]),
        ):
            pipeline = ISPPipeline()
            pipeline.run(self.image_input, self.metadata, save_to_folder=self.test_dir)

            assert mock_save.call_count == 3
            assert (self.test_dir / "time_per_step.txt").exists()

    def test_empty_input_handling(self):
        # Empty inputs shouldn't even look at the registry, but we mock it just in case
        fake_reg = self._get_fake_registry()
        with (
            patch("pipeline.config", self.mock_cfg),
            patch("pipeline.ISP_REGISTRY", fake_reg),
            patch("pipeline.pkgutil.iter_modules", return_value=[]),
        ):
            pipeline = ISPPipeline()
            results = pipeline.run([], [])
            assert results == []
