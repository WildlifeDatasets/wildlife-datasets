import argparse
import inspect
import json
import os
import sys
import unittest
import warnings

import pytest
from huggingface_hub import CacheNotFound, scan_cache_dir

from wildlife_datasets import datasets
from wildlife_datasets.tests.conftest import DEFAULT_DATA_ROOT

SKIP_DATASETS = {
    "Drosophila": "slow to load",
    "NewtsKent": "private data",
}

DATA_FOLDER = {
    "BalearicLizardSegmented": "BalearicLizard",
}

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "real_datasets_snapshot.json")


def _load_snapshot() -> dict:
    if not os.path.isfile(SNAPSHOT_PATH):
        return {}
    with open(SNAPSHOT_PATH) as file:
        return json.load(file)


_SNAPSHOT = _load_snapshot()


def _hf_dataset_cached(hf_url: str) -> bool:
    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        return False
    return any(repo.repo_type == "dataset" and repo.repo_id == hf_url for repo in cache_info.repos)


def _optional_load_kwargs(cls, load_keypoints: bool) -> dict:
    """Builds extra create_catalogue kwargs to exercise on every real dataset.

    Currently only load_keypoints: introspects create_catalogue instead of hardcoding
    dataset names, so any dataset that later gains a load_keypoints option is covered
    automatically.
    """

    if not load_keypoints:
        return {}
    if "load_keypoints" in inspect.signature(cls.create_catalogue).parameters:
        return {"load_keypoints": True}
    return {}


def _load_dataset_or_skip(cls, data_root, skip, load_keypoints: bool = True):
    extra_kwargs = _optional_load_kwargs(cls, load_keypoints)
    if cls.saved_to_system_folder:
        if not _hf_dataset_cached(cls.hf_url):
            skip(f"HuggingFace dataset not cached locally: {cls.hf_url}")
        return cls(**extra_kwargs)

    root_extension = cls.display_name()
    if root_extension in DATA_FOLDER:
        root_extension = DATA_FOLDER[root_extension]
    root = os.path.join(data_root, root_extension)
    if not os.path.isdir(root):
        skip(f"data not present locally: {root}")
    return cls(root, **extra_kwargs)


def _snapshot_entry(dataset) -> dict:
    return {
        "length": len(dataset),
        "first_image_ids": [str(x) for x in dataset.df["image_id"].iloc[:5]],
    }


def _check_snapshot(cls, dataset) -> None:
    expected = _SNAPSHOT.get(cls.__name__)
    if expected is None:
        warnings.warn(f"{cls.__name__}: does not have a snapshot. ")
        return
    actual = _snapshot_entry(dataset)
    if actual != expected:
        warnings.warn(f"{cls.__name__}: snapshot mismatch. Got {actual}, expected {expected}. ")


class TestLoadAllDatasets(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _inject_data_root(self, pytestconfig):
        self.data_root = pytestconfig.getoption("--data-root")
        self.load_keypoints = not pytestconfig.getoption("--no-load-keypoints")


def _make_test(cls):
    def test(self):
        dataset = _load_dataset_or_skip(cls, self.data_root, self.skipTest, self.load_keypoints)
        self.assertGreater(len(dataset), 0, f"{cls.__name__} loaded with 0 rows")
        _check_snapshot(cls, dataset)

    return test


for _cls in datasets.names_all:
    if _cls.__name__ in SKIP_DATASETS:
        continue
    setattr(TestLoadAllDatasets, f"test_{_cls.__name__}", _make_test(_cls))


def _raise_skip(msg: str) -> None:
    raise unittest.SkipTest(msg)


def _write_snapshot(data_root: str) -> None:
    snapshot = dict(_SNAPSHOT)
    for cls in datasets.names_all:
        if cls.__name__ in SKIP_DATASETS:
            continue
        try:
            dataset = _load_dataset_or_skip(cls, data_root, _raise_skip, load_keypoints=False)
        except unittest.SkipTest as e:
            print(f"{cls.__name__}: skipped ({e})")
            continue
        snapshot[cls.__name__] = _snapshot_entry(dataset)
        print(f"{cls.__name__}: {snapshot[cls.__name__]}")
    with open(SNAPSHOT_PATH, "w") as file:
        json.dump(snapshot, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-snapshot", action="store_true")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--no-load-keypoints", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.write_snapshot:
        _write_snapshot(args.data_root)
    else:
        sys.argv = sys.argv[:1] + remaining
        unittest.main()
