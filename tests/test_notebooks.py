import subprocess  # noqa: S404
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

from ispfoundry.utils import get_git_root


def get_non_ignored_notebooks(directory: Path) -> list[Path]:
    """Uses git to find all .ipynb files that are not ignored."""
    try:
        # --cached: tracked files
        # --others: untracked files
        # --exclude-standard: use default gitignore rules
        cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.ipynb"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=directory, check=True)
        # Convert relative output paths to absolute Path objects
        paths = [directory / p for p in result.stdout.splitlines() if p.strip()]
        # Still exclude private/temp notebooks starting with _
        return sorted([p for p in paths if not p.name.startswith("_")])
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to standard glob if git is missing or not a repo
        return sorted(directory.glob("[!_]*.ipynb"))


# Configuration
NOTEBOOK_DIR = get_git_root() / "notebooks"
NOTEBOOKS = get_non_ignored_notebooks(NOTEBOOK_DIR)


@pytest.mark.slow
@pytest.mark.parametrize(
    "nb_path",
    NOTEBOOKS,
    ids=lambda p: p.name,  # Shows "test_notebook_execution[my_notebook.ipynb]"
)
def test_notebook_execution(nb_path: Path):
    with open(nb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOK_DIR)}})
    except Exception as e:
        pytest.fail(f"Notebook {nb_path.name} failed execution:\n{e}")
