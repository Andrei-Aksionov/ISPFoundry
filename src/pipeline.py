import importlib
import pkgutil
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from loguru import logger

import pipeline_steps
from base import ISP_REGISTRY, ISPStep
from configs.config_loader import config
from utils import save_ndarray_as_jpg


class ISPPipeline:
    def __init__(self, steps: Sequence[ISPStep] | None = None) -> None:
        """
        Initializes the ISPPipeline with an optional sequence of ISP steps.

        Args:
            steps: A sequence of ISPStep objects to be executed. If None,
                   default steps from the configuration will be used.

        """
        self.steps = steps or config.pipeline.default_steps
        self._discover_steps()

    def _discover_steps(self) -> None:
        """Scans and imports modules from `pipeline` folder by triggering @register_step decorators."""

        logger.info(" Discovering Pipeline Step Implementations ".center(50, "-"))
        for _, module_name, _ in pkgutil.iter_modules(pipeline_steps.__path__):
            importlib.import_module(f"pipeline_steps.{module_name}")
            logger.info(f"Loaded: {module_name}")

    def run(
        self,
        raw_imgs: Sequence[np.ndarray],
        metadata: Sequence[dict],
        config_overrides: dict[ISPStep, Any] | None = None,
        save_to_folder: Path | None = None,
    ) -> list[np.ndarray]:
        """
        Executes the image processing pipeline on a sequence of raw images.

        Args:
            raw_imgs: A sequence of raw input images (NumPy arrays).
            metadata: A sequence of dictionaries containing metadata for each image.
            config_overrides: An optional dictionary to override default parameters for specific
                    ISP steps. The keys are ISPStep enum values and values are
                    dictionaries of parameters.
            save_to_folder: An optional Path object. If provided, intermediate and
                            final processed images, as well as performance metrics,
                            will be saved to this folder.

        Returns:
            A list of processed images as NumPy arrays.

        """

        # 1. Preparation
        image_input = [ri.copy() for ri in raw_imgs]
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
                payload_to_save = image_input[0] if isinstance(image_input, Sequence) else image_input
                # after black level subtraction there might be negative values in the image,
                # that are preserved for better align and merge
                payload_to_save = payload_to_save.clip(0, None)
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
        image_input: np.ndarray | list[np.ndarray],
        metadata: Sequence[dict],
        config_overrides: dict,
    ) -> list[np.ndarray]:
        """
        Executes a specific ISP step from the ISP registry with provided configuration overrides.

        Args:
            step: The ISPStep enum value representing the step to execute.
            image_input: The input images, which can be either a single NumPy array or a list of arrays.
            metadata: A sequence of dictionaries containing metadata for each image.
            config_overrides: An optional dictionary containing any configuration overrides specific to this step.

        Returns:
            A list of processed images as NumPy arrays.

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
