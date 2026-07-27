import os
import unittest

from huggingface_hub import scan_cache_dir

from wildlife_datasets import datasets

DATA_ROOT = "/data/wildlife_datasets/data"

SKIP_DATASETS = {
    "Drosophila": "slow to load",
    "NewtsKent": "private data",
}

DATA_FOLDER = {
    "BalearicLizardSegmented": "BalearicLizard"
}

@unittest.skipUnless(os.path.isdir(DATA_ROOT), f"Data folder not available: {DATA_ROOT}")
class TestLoadAllDatasets(unittest.TestCase):
    pass


def _hf_dataset_cached(hf_url: str) -> bool:
    cache_info = scan_cache_dir()
    return any(repo.repo_type == "dataset" and repo.repo_id == hf_url for repo in cache_info.repos)


def _make_test(cls):
    def test(self):
        if cls.saved_to_system_folder:
            if not _hf_dataset_cached(cls.hf_url):
                self.skipTest(f"HuggingFace dataset not cached locally: {cls.hf_url}")
            dataset = cls()
        else:
            root_extension = cls.display_name()
            if root_extension in DATA_FOLDER:
                root_extension = DATA_FOLDER[root_extension]

            root = os.path.join(DATA_ROOT, root_extension)
            if not os.path.isdir(root):
                self.skipTest(f"data not present locally: {root}")
            dataset = cls(root)
        self.assertGreater(len(dataset), 0, f"{cls.__name__} loaded with 0 rows")

    return test


for _cls in datasets.names_all:
    if _cls.__name__ in SKIP_DATASETS:
        continue
    setattr(TestLoadAllDatasets, f"test_{_cls.__name__}", _make_test(_cls))


if __name__ == "__main__":
    unittest.main()
