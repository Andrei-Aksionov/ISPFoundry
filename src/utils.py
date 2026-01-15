from pathlib import Path


def get_git_root() -> Path:
    """Returns path to the root of the git repository."""
    try:
        return next(parent for parent in [Path.cwd()] + list(Path.cwd().parents) if (parent / ".git").is_dir())
    except StopIteration:
        raise FileNotFoundError("No .git directory found in any parent directory")
