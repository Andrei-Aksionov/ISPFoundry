from enum import Enum
from typing import Any, Callable


class ISPStep(str, Enum):
    BLACK_LEVEL_SUBTRACTION = "black_level_subtraction"
    NORMALIZATION = "normalization"

    def __str__(self):  # noqa: D105
        return self.value


ISP_REGISTRY = {}


def register_step(step: ISPStep) -> Callable[..., Any]:
    """Decorator to register step into ISP registry."""  # noqa: DOC201

    def decorator(func):  # noqa: ANN001
        ISP_REGISTRY[step] = func
        return func

    return decorator
