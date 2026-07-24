import os
import tempfile
import unittest

from wildlife_datasets.datasets import WildlifeDataset


class _D(WildlifeDataset):
    saved_to_system_folder = False
    summary = {}
    calls = []

    @classmethod
    def _download(cls, **kwargs):
        cls.calls.append("download")

    @classmethod
    def _extract(cls, **kwargs):
        cls.calls.append("extract")


class DownloadLifecycleTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        _D.calls = []
        _D.saved_to_system_folder = False
        _D.summary = {}

    def tearDown(self):
        self._tmp.cleanup()

    def marker_path(self, root=None):
        return os.path.join(root if root is not None else self.root, _D.download_mark_name)


class TestDownload(DownloadLifecycleTestCase):
    def test_fresh_download_calls_download_and_creates_marker(self):
        _D.download(self.root)
        self.assertEqual(_D.calls, ["download"])
        self.assertTrue(os.path.exists(self.marker_path()))

    def test_already_downloaded_without_force_skips(self):
        _D.download(self.root)
        _D.calls = []
        _D.download(self.root)
        self.assertEqual(_D.calls, [])

    def test_already_downloaded_with_force_redownloads(self):
        _D.download(self.root)
        _D.calls = []
        _D.download(self.root, force=True)
        self.assertEqual(_D.calls, ["download"])
        self.assertTrue(os.path.exists(self.marker_path()))

    def test_saved_to_system_folder_ignores_marker(self):
        _D.saved_to_system_folder = True
        _D.download(self.root)
        _D.download(self.root)
        self.assertEqual(_D.calls, ["download", "download"])
        self.assertFalse(os.path.exists(self.marker_path()))

    def test_license_file_written_when_present_in_summary(self):
        _D.summary = {"licenses_url": "http://example.com/license"}
        _D.download(self.root)
        license_path = os.path.join(self.root, _D.license_file_name)
        self.assertTrue(os.path.exists(license_path))
        with open(license_path) as file:
            self.assertEqual(file.read(), "http://example.com/license")

    def test_license_file_not_written_when_absent_from_summary(self):
        _D.download(self.root)
        license_path = os.path.join(self.root, _D.license_file_name)
        self.assertFalse(os.path.exists(license_path))

    def test_root_is_created_if_missing(self):
        new_root = os.path.join(self.root, "new_subdir")
        self.assertFalse(os.path.exists(new_root))
        _D.download(new_root)
        self.assertTrue(os.path.exists(new_root))

    def test_cwd_restored_after_download(self):
        cwd_before = os.getcwd()
        _D.download(self.root)
        self.assertEqual(os.getcwd(), cwd_before)


class TestExtract(DownloadLifecycleTestCase):
    def test_extract_calls_hook_and_creates_marker(self):
        _D.extract(self.root)
        self.assertEqual(_D.calls, ["extract"])
        self.assertTrue(os.path.exists(self.marker_path()))

    def test_saved_to_system_folder_skips_marker(self):
        _D.saved_to_system_folder = True
        _D.extract(self.root)
        self.assertEqual(_D.calls, ["extract"])
        self.assertFalse(os.path.exists(self.marker_path()))


class TestGetData(DownloadLifecycleTestCase):
    def test_fresh_dataset_downloads_then_extracts(self):
        _D.get_data(self.root)
        self.assertEqual(_D.calls, ["download", "extract"])

    def test_already_downloaded_without_force_does_nothing(self):
        _D.get_data(self.root)
        _D.calls = []
        _D.get_data(self.root)
        self.assertEqual(_D.calls, [])

    def test_already_downloaded_with_force_runs_again(self):
        _D.get_data(self.root)
        _D.calls = []
        _D.get_data(self.root, force=True)
        self.assertEqual(_D.calls, ["download", "extract"])

    def test_saved_to_system_folder_always_runs(self):
        _D.saved_to_system_folder = True
        _D.get_data(self.root)
        _D.calls = []
        _D.get_data(self.root)
        self.assertEqual(_D.calls, ["download", "extract"])


if __name__ == "__main__":
    unittest.main()
