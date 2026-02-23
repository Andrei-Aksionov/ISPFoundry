# TODO (andrei aksionau): remove noqa
# flake8: noqa
from pathlib import Path

import importlib
import pkgutil
from typing import Sequence, Any
from base import ISP_REGISTRY, ISPStep
import pipeline_steps
from loguru import logger
import numpy as np
from configs.config_loader import config
from utils import save_ndarray_as_jpg


class ISPPipeline:
    def __init__(self, steps: Sequence[ISPStep] | None = None) -> None:
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
        raw_img: np.ndarray,
        config: dict[ISPStep, Any] | None = None,
        save_to_folder: Path | None = None,
    ) -> np.ndarray | list[np.ndarray]:

        if save_to_folder:
            save_ndarray_as_jpg(raw_img, save_to_folder / "step_0_raw_image.jpg")

        processed_img = raw_img.copy()
        config = config or {}

        for step_idx, step in enumerate(self.steps):
            print(f"{step=}")
            if step not in ISP_REGISTRY:
                raise ValueError(f"Step `{step}` has no implementation in pipeline_steps/ folder.")

            logger.info(f"Executing step `{step}`")

            func = ISP_REGISTRY[step]
            params = config.get(step, {})

            processed_img = func(processed_img, **params)

            if save_to_folder:
                save_ndarray_as_jpg(processed_img.clip(0, None), save_to_folder / f"step_{step_idx + 1}_{step}.jpg")

        return processed_img
