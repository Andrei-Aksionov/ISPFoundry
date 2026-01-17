import pathlib

from dataset.hdrplus_loader import HDRPlusDatasetDownloader


class TestHDRPlusDatasetDownloader:
    source_path = str(pathlib.PurePosixPath("hdrplusdata/20171106_subset/bursts/0006_20160722_115157_431"))
    expected_folder_name = "0006_20160722_115157_431"

    def test_download_default_dst(self, tmp_path) -> None:
        """Tests the basic download functionality, ensuring all expected files are present and meet minimum size requirements."""

        downloader = HDRPlusDatasetDownloader()
        downloader.download(self.source_path, tmp_path, force_download=True)

        target_inner_folder = tmp_path / self.expected_folder_name
        assert target_inner_folder.is_dir()

        folder_content = list(target_inner_folder.iterdir())
        file_names = {p.name for p in folder_content}

        # Assert text files are present
        expected_text_files = {"timing.txt", "rgb2rgb.txt"}
        assert expected_text_files.issubset(file_names), "Missing expected simple files."

        # Define expected file patterns, counts, and minimum sizes
        expected_patterns = [
            ("lens_shading_map", ".tiff", 7, 1024),  # prefix, suffix, expected_count, min_size_bytes
            ("payload", ".dng", 7, 1024 * 1024),  # 1MB minimum for payloads
        ]

        # Assert file patterns and properties
        for prefix, suffix, expected_count, min_size in expected_patterns:
            matching_files = [p for p in folder_content if p.name.startswith(prefix) and p.name.endswith(suffix)]
            assert len(matching_files) == expected_count, (
                f"Expected {expected_count} files starting with '{prefix}' and ending with '{suffix}', but got {len(matching_files)}."
            )
            assert all(p.stat().st_size > min_size for p in matching_files), (
                f"All files matching '{prefix}*{suffix}' should have size > {min_size} bytes."
            )

        # Assert total file count
        expected_total_count = len(expected_text_files) + sum(p[2] for p in expected_patterns)
        assert len(folder_content) == expected_total_count, (
            f"Expected {expected_total_count} files in total, but got {len(folder_content)}."
        )

    def test_download_skips_if_exists_and_not_forced(self, tmp_path) -> None:
        """Tests that downloading to an existing, complete directory with force_download=False does not re-download or alter existing files."""

        downloader = HDRPlusDatasetDownloader()

        # First download to establish initial state (force_download=True to ensure it happens)
        downloader.download(self.source_path, tmp_path, force_download=True)
        target_inner_folder = tmp_path / self.expected_folder_name

        # Record initial timestamps and sizes
        initial_timestamps = {p.name: p.stat().st_mtime for p in target_inner_folder.iterdir()}
        initial_sizes = {p.name: p.stat().st_size for p in target_inner_folder.iterdir()}
        initial_file_count = len(list(target_inner_folder.iterdir()))

        # Attempt to download again with force_download=False
        downloader.download(self.source_path, tmp_path, force_download=False)

        # Verify no changes occurred
        final_timestamps = {p.name: p.stat().st_mtime for p in target_inner_folder.iterdir()}
        final_sizes = {p.name: p.stat().st_size for p in target_inner_folder.iterdir()}
        final_file_count = len(list(target_inner_folder.iterdir()))

        assert initial_file_count == final_file_count, "File count changed unexpectedly."
        assert initial_timestamps == final_timestamps, "File modification times changed unexpectedly."
        assert initial_sizes == final_sizes, "File sizes changed unexpectedly."

    def test_download_overwrites_if_exists_and_forced(self, tmp_path) -> None:
        """Tests that downloading to an existing directory with force_download=True correctly overwrites or cleans up previous content."""

        downloader = HDRPlusDatasetDownloader()

        target_path = tmp_path / self.expected_folder_name
        target_path.mkdir()

        # Create a dummy file in the target directory to simulate pre-existing content
        dummy_file = target_path / "dummy.txt"
        dummy_file.write_text("This is a dummy file that should be removed.")
        assert dummy_file.exists()

        # Download with force_download=True. This should clear or overwrite the existing folder.
        downloader.download(self.source_path, tmp_path, force_download=True)

        # Assert that the dummy file is gone
        assert not dummy_file.exists(), "Dummy file was not removed during forced download."

        # Then assert that the correct content is downloaded, similar to the default test
        folder_content = list(target_path.iterdir())
        file_names = {p.name for p in folder_content}

        expected_simple_files = {"timing.txt", "rgb2rgb.txt"}
        assert expected_simple_files.issubset(file_names)

        expected_patterns = [
            ("lens_shading_map", ".tiff", 7, 1024),
            ("payload", ".dng", 7, 1024 * 1024),
        ]

        for prefix, suffix, expected_count, min_size in expected_patterns:
            matching_files = [p for p in folder_content if p.name.startswith(prefix) and p.name.endswith(suffix)]
            assert len(matching_files) == expected_count
            assert all(p.stat().st_size > min_size for p in matching_files)

        expected_total_count = len(expected_simple_files) + sum(p[2] for p in expected_patterns)
        assert len(folder_content) == expected_total_count
