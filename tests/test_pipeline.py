import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Sequence
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from ispfoundry import ISPStep
from ispfoundry.datasets import Metadata
from ispfoundry.pipeline import ISPPipeline


class MockStep:
    """Simulated ISPStep enum for testing purposes."""

    STEP_A = "step_a"
    STEP_B = "step_b"


class TestISPPipeline(unittest.TestCase):
    def setUp(self):
        """Setup common data and path for all tests."""
        self.image_input = np.array([
            np.array([1] * 4, dtype=np.float32),
            np.array([2] * 4, dtype=np.float32),
        ])
        mtd = Metadata(
            file_path=Path("test_file_path"),
            image_width=2,
            image_height=2,
            black_levels=np.array([50, 60, 70, 80]),  # R, Gr, Gb, B
            white_level=1000,
            color_description="RGBG",  # R=index0, G=index1, B=index2, G=index3
            raw_pattern=np.array([[0, 1], [3, 2]]),  # Standard RGGB
            exposure_time=0.1,
            iso=100,
            cfa_plane_color="Red,Green,Blue",
            # 3 pairs of (Scale, Offset): R=(0.01, 0.001), G=(0.02, 0.002), B=(0.03, 0.003)
            noise_profile=np.array([0.01, 0.001, 0.02, 0.002, 0.03, 0.003]),
            camera_model_name="test_camera",
        )
        self.metadata = [replace(mtd), replace(mtd)]
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
        with (
            patch("ispfoundry.pipeline.config", self.mock_cfg),
            patch("ispfoundry.pipeline.pkgutil.iter_modules", return_value=[]),
        ):
            pipeline = ISPPipeline()
            assert len(pipeline.steps) == 2

    def test_init_empty_list_steps(self):
        with (
            patch("ispfoundry.pipeline.config", self.mock_cfg),
            patch("ispfoundry.pipeline.pkgutil.iter_modules", return_value=[]),
        ):
            pipeline = ISPPipeline(steps=[])
            assert len(pipeline.steps) == 2

    def test_init_custom_list_steps(self):
        custom_steps = [ISPStep.BLACK_LEVEL_SUBTRACTION]
        with (
            patch("ispfoundry.pipeline.config", self.mock_cfg),
            patch("ispfoundry.pipeline.pkgutil.iter_modules", return_value=[]),
        ):
            pipeline = ISPPipeline(steps=custom_steps)
            assert len(pipeline.steps) == 1
            assert pipeline.steps == custom_steps

    def test_discover_steps_logic(self):
        with patch("ispfoundry.pipeline.pkgutil.iter_modules") as mock_iter:
            mock_iter.return_value = [(None, "test_module", False)]
            with (
                patch("ispfoundry.pipeline.importlib.import_module") as mock_import,
                patch("ispfoundry.pipeline.config", self.mock_cfg),
            ):
                ISPPipeline()
                mock_import.assert_called_with("ispfoundry.pipeline_steps.test_module")

    def test_pipeline_transformation_logic(self):
        fake_reg = self._get_fake_registry()
        with (
            patch("ispfoundry.pipeline.config", self.mock_cfg),
            patch("ispfoundry.pipeline.ISP_REGISTRY", fake_reg),
            patch("ispfoundry.pipeline.pkgutil.iter_modules", return_value=[]),
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
            patch("ispfoundry.pipeline.config", self.mock_cfg),
            patch("ispfoundry.pipeline.ISP_REGISTRY", fake_reg),
            patch("ispfoundry.pipeline.pkgutil.iter_modules", return_value=[]),
        ):
            pipeline = ISPPipeline()
            pipeline.run(self.image_input, self.metadata, config_overrides=overrides)  # ty:ignore[invalid-argument-type]

            _, kwargs = fake_reg[MockStep.STEP_A].call_args
            assert kwargs["param_x"] == 50

    def test_missing_step_implementation_raises_error(self):
        # Explicitly empty registry to trigger the error
        with (
            patch("ispfoundry.pipeline.config", self.mock_cfg),
            patch("ispfoundry.pipeline.ISP_REGISTRY", {}),
            patch("ispfoundry.pipeline.pkgutil.iter_modules", return_value=[]),
        ):
            pipeline = ISPPipeline()
            with pytest.raises(ValueError, match="has no implementation"):
                pipeline.run(self.image_input, self.metadata)

    def test_file_saving_and_telemetry(self):
        fake_reg = self._get_fake_registry()
        with (
            patch("ispfoundry.pipeline.config", self.mock_cfg),
            patch("ispfoundry.pipeline.ISP_REGISTRY", fake_reg),
            patch("ispfoundry.pipeline.save_ndarray_as_jpg") as mock_save,
            patch("ispfoundry.pipeline.pkgutil.iter_modules", return_value=[]),
        ):
            pipeline = ISPPipeline()
            pipeline.run(self.image_input, self.metadata, save_to_folder=self.test_dir)

            assert mock_save.call_count == 3
            assert (self.test_dir / "time_per_step.txt").exists()

    def test_empty_input_handling(self):
        # Empty inputs shouldn't even look at the registry, but we mock it just in case
        fake_reg = self._get_fake_registry()
        with (
            patch("ispfoundry.pipeline.config", self.mock_cfg),
            patch("ispfoundry.pipeline.ISP_REGISTRY", fake_reg),
            patch("ispfoundry.pipeline.pkgutil.iter_modules", return_value=[]),
        ):
            pipeline = ISPPipeline()
            results = pipeline.run(np.array([], dtype=np.float32), [])
            assert results.size == 0
