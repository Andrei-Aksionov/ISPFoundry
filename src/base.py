from enum import Enum
from typing import Any, Callable


class ISPStep(str, Enum):
    BLACK_LEVEL_SUBTRACTION = "black_level_subtraction"
    LENS_SHADING_CORRECTION = "lens_shading_correction"

    def __str__(self) -> str:
        """Return a string representation of the ISP step."""  # noqa: DOC201
        return self.value


ISP_REGISTRY = {}


def register_step(step: ISPStep) -> Callable[..., Any]:
    """Decorator to register step into ISP registry."""  # noqa: DOC201

    def decorator(func: Any) -> Any:
        ISP_REGISTRY[step] = func
        return func

    return decorator
