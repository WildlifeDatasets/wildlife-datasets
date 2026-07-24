import os
import tempfile
import unittest

import pandas as pd

from wildlife_datasets.datasets import IPanda50, MacaqueFaces, WildlifeDataset, names_all

from .utils import create_dataset, load_datasets

dataset_names = [IPanda50, MacaqueFaces]
datasets = load_datasets(dataset_names)


def make_base_df():
    return pd.DataFrame(
        {
            "image_id": [0, 1, 2, 3],
            "identity": ["a", "a", "b", "c"],
            "path": ["img0.png", "img1.png", "img2.png", "img3.png"],
        }
    )


class TestDisplayName(unittest.TestCase):
    def test_all_registered_datasets(self):
        for dataset_class in names_all:
            dataset_class.display_name()

    def test_returns_own_name_without_outdated_ancestor(self):
        class _Fresh(WildlifeDataset):
            pass

        self.assertEqual(_Fresh.display_name(), "_Fresh")

    def test_strips_single_outdated_ancestor(self):
        class _V1(WildlifeDataset):
            outdated_dataset = True

        class _V2(_V1):
            outdated_dataset = False

        self.assertEqual(_V2.display_name(), "_V1")

    def test_strips_multiple_outdated_ancestors(self):
        class _V1(WildlifeDataset):
            outdated_dataset = True

        class _V2(_V1):
            outdated_dataset = True

        class _V3(_V2):
            outdated_dataset = False

        self.assertEqual(_V3.display_name(), "_V1")


class TestGetSubset(unittest.TestCase):
    def test_existing_real_datasets(self):
        n_new = 10
        for dataset_old in datasets:
            n_old = len(dataset_old)
            dataset_new = dataset_old.get_subset(range(n_new))
            for dataset, n in zip([dataset_old, dataset_new], [n_old, n_new]):
                self.assertEqual(n, len(dataset))
                self.assertEqual(n, len(dataset.labels))
                self.assertEqual(dataset.df["identity"].nunique(), len(dataset.labels_map))
                ids1 = dataset.df["identity"].to_numpy()
                ids2 = dataset.labels_map[dataset.labels]
                self.assertEqual(tuple(ids1), tuple(ids2))

    def test_boolean_mask(self):
        dataset = create_dataset(make_base_df())
        subset = dataset.get_subset([True, False, True, False])
        self.assertEqual(len(subset), 2)
        self.assertEqual(list(subset.df["identity"]), ["a", "b"])

    def test_shorter_int_list(self):
        dataset = create_dataset(make_base_df())
        subset = dataset.get_subset([0, 2])
        self.assertEqual(len(subset), 2)
        self.assertEqual(list(subset.df["identity"]), ["a", "b"])

    def test_full_length_int_list_reorders(self):
        dataset = create_dataset(make_base_df())
        subset = dataset.get_subset([3, 2, 1, 0])
        self.assertEqual(list(subset.df["identity"]), ["c", "b", "a", "a"])

    def test_does_not_mutate_original(self):
        dataset = create_dataset(make_base_df())
        subset = dataset.get_subset([0, 2])
        subset.df.loc[0, "identity"] = "changed"
        self.assertEqual(dataset.df.loc[0, "identity"], "a")

    def test_labels_recomputed_for_subset(self):
        dataset = create_dataset(make_base_df())
        subset = dataset.get_subset([0, 1])
        self.assertEqual(subset.df["identity"].nunique(), len(subset.labels_map))
        self.assertEqual(len(subset.labels), 2)


class TestTemporaryAttrs(unittest.TestCase):
    def test_sets_and_restores(self):
        dataset = create_dataset(make_base_df())
        original = dataset.img_load
        with dataset.temporary_attrs(img_load="bbox"):
            self.assertEqual(dataset.img_load, "bbox")
        self.assertEqual(dataset.img_load, original)

    def test_restores_on_exception(self):
        dataset = create_dataset(make_base_df())
        original = dataset.img_load
        with self.assertRaises(RuntimeError):
            with dataset.temporary_attrs(img_load="bbox"):
                raise RuntimeError("boom")
        self.assertEqual(dataset.img_load, original)


class TestSetAbsolutePaths(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_col_path(self):
        df = pd.DataFrame({"image_id": [0], "identity": ["a"], "path": ["img0.png"]})
        dataset = WildlifeDataset(df=df, root=self.root, check_files=False)
        dataset.set_absolute_paths()
        self.assertEqual(dataset.df["path"].iloc[0], os.path.join(self.root, "img0.png"))
        self.assertIsNone(dataset.root)

    def test_custom_col_path(self):
        df = pd.DataFrame({"image_id": [0], "identity": ["a"], "filepath": ["img0.png"]})
        dataset = WildlifeDataset(df=df, root=self.root, check_files=False, col_path="filepath")
        dataset.set_absolute_paths()
        self.assertEqual(dataset.df["filepath"].iloc[0], os.path.join(self.root, "img0.png"))
        self.assertIsNone(dataset.root)


class TestConstructionWarningsErrors(unittest.TestCase):
    def test_missing_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            WildlifeDataset(root="/this/path/does/not/exist/xyz", check_files=False)

    def test_outdated_dataset_warns(self):
        class _Outdated(WildlifeDataset):
            outdated_dataset = True

        df = pd.DataFrame({"image_id": [0], "identity": ["a"], "path": ["img0.png"]})
        with self.assertWarns(UserWarning):
            _Outdated(df=df, root=None, check_files=False)

    def test_not_determined_by_df_warns(self):
        class _NotDetermined(WildlifeDataset):
            determined_by_df = False

        df = pd.DataFrame({"image_id": [0], "identity": ["a"], "path": ["img0.png"]})
        with self.assertWarns(UserWarning):
            _NotDetermined(df=df, root=None, check_files=False)


if __name__ == "__main__":
    unittest.main()
