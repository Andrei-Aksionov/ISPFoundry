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
        processed_imgs = [ri.copy() for ri in raw_imgs]
        config_overrides = config_overrides or {}
        telemetry = []

        if save_to_folder:
            # always saving the first image from the burst
            save_ndarray_as_jpg(processed_imgs[0], save_to_folder / "step_0_raw_image.jpg")

        # 2. Execution Loop
        total_start = perf_counter()

        for step_idx, step in enumerate(self.steps, start=1):
            step_start = perf_counter()
            logger.info(f"Executing step {step_idx}/{len(self.steps)} `{step}` ")

            # Execute pipeline step logic
            processed_imgs = self._execute_step(step, processed_imgs, metadata, config_overrides)

            # Post-step bookkeeping
            elapsed = timedelta(seconds=perf_counter() - step_start)
            logger.info(f"Step {step_idx}/{len(self.steps)} `{step}` took {elapsed}")

            if save_to_folder:
                telemetry.append((step, elapsed))
                save_ndarray_as_jpg(processed_imgs[0].clip(0, None), save_to_folder / f"step_{step_idx}_{step}.jpg")

        # 3. Finalization
        total_elapsed = timedelta(seconds=perf_counter() - total_start)
        logger.info(f"Full run took {total_elapsed}")

        if save_to_folder:
            self._save_telemetry(save_to_folder, telemetry, total_elapsed)

        return processed_imgs

    def _execute_step(
        self, step: ISPStep, imgs: list[np.ndarray], metadata: Sequence[dict], config_overrides: dict
    ) -> list[np.ndarray]:
        """
        Handles the actual function lookup and batch application.

        Returns:
            A list of processed images as NumPy arrays.

        Raises:
            ValueError: If an ISP step specified in the pipeline has no
                        corresponding implementation registered.

        """

        if step not in ISP_REGISTRY:
            raise ValueError(f"Step `{step}` has no implementation in pipeline_steps/ folder.")

        func = ISP_REGISTRY[step]
        params = config_overrides.get(step, {})

        return [func(img, mt, **params) for img, mt in zip(imgs, metadata)]

    def _save_telemetry(self, folder: Path, data: list[tuple[ISPStep, timedelta]], total: timedelta) -> None:
        """Separates file I/O from the main logic flow."""

        with (folder / "time_per_step.txt").open("w") as f:
            f.write(f"{'Step name '.ljust(50, '-')} Elapsed\n\n")
            for name, duration in data:
                f.write(f"{(name + ' ').ljust(50, '.')} {duration}\n")
            f.write(f"{'Full pipeline run '.ljust(50, '.')} {total}\n")
