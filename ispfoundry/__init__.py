from enum import Enum
from typing import Any, Callable

__all__ = [
    "ISPStep",
    "register_step",
]


class ISPStep(str, Enum):
    BLACK_LEVEL_SUBTRACTION = "black_level_subtraction"
    LENS_SHADING_CORRECTION = "lens_shading_correction"
    ALIGN_AND_MERGE = "align_and_merge"

    def __str__(self) -> str:
        """
        String representation of the ISP step.

        Returns:
            ISP step name as a lower-case string.

        """
        return self.value


ISP_REGISTRY = {}


def register_step(step: ISPStep) -> Callable[..., Any]:
    """
    Decorator to register an ISP step function into the global registry.

    This decorator allows functions to be associated with specific ISP steps for later retrieval or execution.

    Args:
        step: The ISP step enum value to associate the function with.

    Returns:
        The original function, now registered in the ISP_REGISTRY dictionary.

    """

    def decorator(func: Any) -> Any:
        ISP_REGISTRY[step] = func
        return func

    return decorator
