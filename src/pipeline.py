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
        self.__discover_steps()

    def __discover_steps(self) -> None:
        """
        Scans the 'pipeline' folder and imports every module found inside.

        This triggers the @register_step decorators.
        """

        logger.info(" Discovering Pipeline Step Implementations ".center(50, "-"))
        for loader, module_name, is_pkg in pkgutil.iter_modules(pipeline_steps.__path__):
            full_module_name = f"pipeline_steps.{module_name}"
            importlib.import_module(full_module_name)
            logger.info(f"Loaded: {module_name}")
        logger.info("-" * 50)

    def run(
        self,
        raw_imgs: Sequence[np.ndarray],
        metadata: Sequence[dict],
        config: dict[ISPStep, Any] | None = None,
        save_to_folder: Path | None = None,
    ) -> list[np.ndarray]:
        """
        Executes the image processing pipeline on a sequence of raw images.

        Args:
            raw_imgs: A sequence of raw input images (NumPy arrays).
            metadata: A sequence of dictionaries containing metadata for each image.
            config: An optional dictionary to override default parameters for specific
                    ISP steps. The keys are ISPStep enum values and values are
                    dictionaries of parameters.
            save_to_folder: An optional Path object. If provided, intermediate and
                            final processed images, as well as performance metrics,
                            will be saved to this folder.

        Returns:
            A list of processed images as NumPy arrays.

        Raises:
            ValueError: If an ISP step specified in the pipeline has no
                        corresponding implementation registered.

        """

        if save_to_folder:
            # always saving the first image from the burst
            save_ndarray_as_jpg(raw_imgs[0], save_to_folder / "step_0_raw_image.jpg")
            time_per_step = []

        processed_imgs = [ri.copy() for ri in raw_imgs]
        config = config or {}

        run_time_start = perf_counter()
        for step_idx, step in enumerate(self.steps, start=1):
            step_time_start = perf_counter()

            if step not in ISP_REGISTRY:
                raise ValueError(f"Step `{step}` has no implementation in pipeline_steps/ folder.")

            func = ISP_REGISTRY[step]
            params = config.get(step, {})
            logger.info(f"Executing step {step_idx}/{len(self.steps)} `{step}` ")

            for processed_idx, (processed_img, mt) in enumerate(zip(processed_imgs, metadata)):
                processed_img = func(processed_img, mt, **params)
                processed_imgs[processed_idx] = processed_img
                if save_to_folder and processed_idx == 0:
                    save_ndarray_as_jpg(processed_img.clip(0, None), save_to_folder / f"step_{step_idx}_{step}.jpg")

            elapsed = timedelta(seconds=perf_counter() - step_time_start)
            logger.info(f"Step `{step}` executed in {elapsed} sec")
            if save_to_folder:
                time_per_step.append((step, elapsed))

        run_elapsed = timedelta(seconds=perf_counter() - run_time_start)
        logger.info(f"Full run took {run_elapsed} sec")
        if save_to_folder:
            with (save_to_folder / "time_per_step.txt").open("w") as f:
                f.write(f"{'Step name '.ljust(50, '-')} Elapsed in seconds\n\n")
                for s, t in time_per_step:
                    f.write(f"{(s + ' ').ljust(50, '.')} {t}\n")

                f.write(f"{'Full pipeline run '.ljust(50, '.')} {run_elapsed}\n")

        return processed_imgs
