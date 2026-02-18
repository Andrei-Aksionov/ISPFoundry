import math
import shutil
from collections.abc import Sequence
from pathlib import Path

import exiftool
import matplotlib.pyplot as plt
import numpy as np


def get_git_root() -> Path:
    """Returns path to the root of the git repository.

    Raises:
        FileNotFoundError: If no .git directory is found in any parent directory.

    """  # noqa: DOC201

    try:
        return next(parent for parent in [Path.cwd()] + list(Path.cwd().parents) if (parent / ".git").is_dir())
    except StopIteration:
        raise FileNotFoundError("No .git directory found in any parent directory")


def get_exif_metadata(path: Path) -> dict:
    """Retrieves EXIF metadata from the specified file using ExifTool.

    Args:
        path (Path): The path to the image file.

    Returns:
        dict: A dictionary containing the EXIF metadata.

    Raises:
        RuntimeError: If ExifTool is not installed on the system.

    """
    if shutil.which("exiftool") is None:
        raise RuntimeError(
            "ExifTool needs to be installed on your system (https://exiftool.org/install.html). On MacOS run `brew install exiftool`"
        )

    with exiftool.ExifToolHelper(common_args=[]) as et:
        return et.get_metadata(path)


def plot_histograms(
    datasets: Sequence[np.ndarray],
    titles: None | Sequence[str] = None,
    xlim: None | Sequence[int] = None,
    plot_comparison: bool = True,
) -> None:
    """Plots histograms for two datasets, optionally including a comparison plot.

    Args:
        datasets (Sequence of np.ndarray): A list containing two numpy arrays representing the datasets.
        titles (Sequence of str, optional): A list of titles for each dataset. Defaults to None.
        xlim (Sequence of int, optional): The x-axis limits for the comparison plot. Defaults to None.
        plot_comparison (bool): Whether to include a third plot comparing the datasets. Defaults to True.

    """
    assert len(datasets) == 2, "Only two histograms are supported"
    assert titles is None or len(titles) == len(datasets), (
        "Number of titles should be equal to number of histograms or be None"
    )
    titles = titles or [""] * len(datasets)

    _, axes = plt.subplots(nrows=1, ncols=3 if plot_comparison else 2, figsize=(20, 5))
    axes = axes.flat

    colors = ("red", "blue")

    # plotting separately
    for data, title, color in zip(datasets, titles, colors):
        ax = next(axes)
        ax.set_title(title)
        ax.hist(data.ravel(), bins=128, color=color)
        ax.grid(True)

    # plotting on the same chart
    if plot_comparison:
        ax = next(axes)
        for data, label, color in zip(datasets, titles, colors):
            ax.set_title(f"{titles[0]} vs {titles[1]}")
            ax.hist(data.ravel(), bins=128, color=color, label=label, alpha=0.25)
            ax.grid(True)

        if xlim is not None:
            ax.set_xlim(xlim)

        if all(titles):
            ax.legend()

    plt.show()


def plot_images(
    images: np.ndarray | Sequence[np.ndarray],
    titles: str | Sequence[str] | None = None,
    fig_size: tuple | None = None,
    inch_width_pre_image: int | None = None,
    max_per_row: int = 3,
) -> None:
    """Display a list of images with optional titles in a grid layout.

    Args:
        images (np.ndarray or Sequence of np.ndarray): List of images to display.
        titles (str, Sequence of str, optional): Titles for each image. Defaults to empty strings.
        fig_size (tuple, optional): Figure size in inches (width, height). If None, calculated automatically.
        inch_width_pre_image (int, optional): Width in inches per image. Used if fig_size is not provided.
        max_per_row (int): Maximum number of plot per row to be displayed.

    """

    if isinstance(images, np.ndarray):
        images = [images]

    if titles is None:
        titles = [""] * len(images)
    else:
        if isinstance(titles, str):
            titles = [titles]
        assert len(images) == len(titles), "Titles were provided, but it's number is not equal to the number of images."

    def _find_best_layout(n: int, max_per_row: int = 3):
        min_per_row = 1 if n == 1 else 2
        min_reserve = n  # start with the worst case
        best_nrow, best_ncol = 1, n

        for nrow in range(1, n + 1):
            ncol = math.ceil(n / nrow)
            if not min_per_row <= ncol <= max_per_row:
                continue
            reserve = nrow * ncol - n
            if reserve < min_reserve:
                min_reserve = reserve
                best_nrow, best_ncol = nrow, ncol

        return best_nrow, best_ncol

    if fig_size is None:
        img_h, img_w = images[0].shape[:2]
        best_nrow, best_ncol = _find_best_layout(len(images), max_per_row)
        aspect_ratio = img_h / img_w
        base_width = inch_width_pre_image or 6  # base width per image in inches
        fig_width = base_width * best_ncol
        fig_height = base_width * aspect_ratio * best_nrow
        fig_size = (fig_width, fig_height)

    _, axes = plt.subplots(best_nrow, best_ncol, figsize=fig_size)
    axes = iter(axes.flatten()) if isinstance(axes, np.ndarray) else iter([axes])

    for img, title in zip(images, titles):
        ax = next(axes)
        ax.set_title(title)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.axis("off")

    # exhaust remaining axes
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
