import os
import platform
import subprocess  # noqa: S404
from pathlib import Path
from shutil import rmtree

from loguru import logger

from configs.config_loader import config
from utils import get_git_root


class HDRPlusDatasetDownloader:
    def __init__(self, enable_multiprocessing: bool = False) -> None:
        """The HDRPlus dataset loader with multiprocessing capabilities for Google Cloud Storage.

        Args:
            enable_multiprocessing (bool): If True, enables multiprocessing for 'gsutil' commands
                when copying data from Google Cloud Storage, which can speed up transfers
                for large numbers of files. At the moment fails on MacOS. Defaults to False.

        """

        self.multiprocessing_enabled = enable_multiprocessing
        mp_flag = "" if self.multiprocessing_enabled else "-o GSUtil:parallel_process_count=1"
        self.cmd_template = f"gsutil {mp_flag} -m cp -r gs://{{source_path}} {{destination_path}}"

    def download(
        self,
        source_path: str | Path,
        destination_path: str | Path | None = None,
        force_download: bool = False,
    ) -> Path:
        """Downloads the HDR+ dataset from a Google Storage bucket to a local destination.

        Args:
            source_path (str | Path): The source path or Google Storage bucket URI (e.g., path to the dataset).
            destination_path (str | Path | None, optional): The local directory where the dataset
                will be saved. If None, it defaults to a subfolder within 'data/raw/hdrplus_dataset'
                relative to the git root. Defaults to None.
            force_download (bool, optional): Remove and re-download if True. Defaults to False.

        Returns:
            The folder where data is downloaded.

        Raises:
            RuntimeError: If the `gsutil` command fails during the download process.

        """

        source_path = Path(source_path)

        if not destination_path:
            destination_path = get_git_root() / config.data.hdrplus_dataset
            logger.info(f"Destination path wasn't explicitly set. Downloading into `{destination_path}`")
        destination_path = Path(destination_path)

        destination_folder = destination_path / source_path.name

        if destination_path.exists():
            if not force_download:
                logger.info("Folder already exists. Force download was disabled.")
                return destination_folder
            rmtree(destination_path)
        os.makedirs(destination_path, exist_ok=True)

        cmd = self.cmd_template.format(source_path=source_path.as_posix(), destination_path=destination_path)

        try:
            # Windows needs shell=True to find 'gsutil'
            is_windows = platform.system() == "Windows"
            subprocess.run(cmd.split(), shell=is_windows, check=True)
            logger.info("Download completed.")
            return destination_folder
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed with return code {e.returncode}") from e
