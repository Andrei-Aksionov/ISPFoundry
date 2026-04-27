import importlib
import pkgutil
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from loguru import logger

from ispfoundry import ISP_REGISTRY, ISPStep, pipeline_steps
from ispfoundry.configs.config_loader import config
from ispfoundry.datasets import Metadata
from ispfoundry.utils import save_ndarray_as_jpg


class ISPPipeline:
    def __init__(self, steps: list[ISPStep] | None = None) -> None:
        """
        Initializes the ISPPipeline with an optional sequence of ISP steps.

        Args:
            steps: A list of ISPStep objects to be executed. If None,
                   default steps from the configuration will be used.

        """
        self.steps = steps or config.pipeline.default_steps
        self._discover_steps()

    def _discover_steps(self) -> None:
        """Scans and imports modules from `pipeline` folder by triggering @register_step decorators."""

        logger.info(" Discovering Pipeline Step Implementations ".center(50, "-"))
        for _, module_name, _ in pkgutil.iter_modules(pipeline_steps.__path__):
            importlib.import_module(f"ispfoundry.pipeline_steps.{module_name}")
            logger.info(f"Loaded: {module_name}")

    def run(
        self,
        raw_input: np.ndarray,
        metadata: Sequence[Metadata],
        config_overrides: dict[ISPStep, Any] | None = None,
        save_to_folder: Path | None = None,
    ) -> np.ndarray:
        """
        Executes the image processing pipeline on a sequence of raw images.

        Args:
            raw_input: 3D Numpy array of shape (N, H, W) containing raw images.
            metadata: A sequence of Metadata classes containing metadata for each image.
            config_overrides: An optional dictionary to override default parameters for specific
                    ISP steps. The keys are ISPStep enum values and values are
                    dictionaries of parameters.
            save_to_folder: An optional Path object. If provided, intermediate and
                            final processed images, as well as performance metrics,
                            will be saved to this folder.

        Returns:
            Processed images as a numpy array of shape (N, H, W).
                If a burst is provided and Align&Merge step is included - the output is a single image
                of shape (H, W).

        """

        # 1. Preparation
        image_input = raw_input.copy()
        config_overrides = config_overrides or {}
        telemetry = []

        if save_to_folder:
            # always saving the first image from the burst
            save_ndarray_as_jpg(image_input[0], save_to_folder / "step_0_raw_image.jpg")

        # 2. Execution Loop
        total_start = perf_counter()

        for step_idx, step in enumerate(self.steps, start=1):
            step_start = perf_counter()
            logger.info(f"Executing step {step_idx}/{len(self.steps)} `{step}` ")

            # Execute pipeline step logic
            image_input = self._execute_step(step, image_input, metadata, config_overrides)

            # Post-step bookkeeping
            elapsed = timedelta(seconds=perf_counter() - step_start)
            logger.info(f"Step {step_idx}/{len(self.steps)} `{step}` took {elapsed}")

            if save_to_folder:
                telemetry.append((step, elapsed))
                # before burst merging the input contains multiple images
                payload_to_save = image_input[0] if image_input.ndim == 3 else image_input
                # if though the image is normalized to range [0, 1] during the pipeline values might
                # exceed this range which is important statistics and clipping should be done only right before saving
                payload_to_save = payload_to_save.clip(0, 1)
                save_ndarray_as_jpg(payload_to_save, save_to_folder / f"step_{step_idx}_{step}.jpg")

        # 3. Finalization
        total_elapsed = timedelta(seconds=perf_counter() - total_start)
        logger.info(f"Full run took {total_elapsed}")

        if save_to_folder:
            self._save_telemetry(save_to_folder, telemetry, total_elapsed)

        return image_input

    def _execute_step(
        self,
        step: ISPStep,
        image_input: np.ndarray,
        metadata: Sequence[Metadata],
        config_overrides: dict,
    ) -> np.ndarray:
        """
        Executes a specific ISP step from the ISP registry with provided configuration overrides.

        Args:
            step: The ISPStep enum value representing the step to execute.
            image_input: 3D Numpy array of shape (N, H, W) containing raw images.
            metadata: A sequence of Metadata classes containing metadata for each image.
            config_overrides: An optional dictionary containing any configuration overrides specific to this step.

        Returns:
            Processed images as a numpy array of shape (N, H, W).

        Raises:
            ValueError: If an ISP step specified in the pipeline has no corresponding implementation registered.

        """

        if step not in ISP_REGISTRY:
            raise ValueError(f"Step `{step}` has no implementation in pipeline_steps/ folder.")

        func = ISP_REGISTRY[step]
        params = config_overrides.get(step, {})

        return func(image_input, metadata, **params)

    def _save_telemetry(self, folder: Path, data: list[tuple[ISPStep, timedelta]], total: timedelta) -> None:
        """
        Saves telemetry data to a text file in the specified folder.

        Args:
            folder: The path to the directory where the telemetry data will be saved.
            data: A list of tuples containing step names and elapsed times for each step.
            total: The total elapsed time for the entire pipeline run.

        """

        with (folder / "time_per_step.txt").open("w") as f:
            f.write(f"{'Step name '.ljust(50, '-')} Elapsed\n\n")
            for name, duration in data:
                f.write(f"{(name + ' ').ljust(50, '.')} {duration}\n")
            f.write(f"{'Full pipeline run '.ljust(50, '.')} {total}\n")
