import os
import tempfile
import unittest

import pandas as pd

from wildlife_datasets.datasets import WildlifeDataset

from .utils import create_dataset


def make_base_row(image_id=0, identity="id0", path="img0.png"):
    return {"image_id": image_id, "identity": identity, "path": path}


class TestRequiredColumns(unittest.TestCase):
    def test_valid_df_passes(self):
        df = pd.DataFrame([make_base_row()])
        create_dataset(df)

    def test_missing_image_id_raises(self):
        df = pd.DataFrame([{"identity": "id0", "path": "img0.png"}])
        with self.assertRaises(ValueError):
            create_dataset(df)

    def test_missing_identity_raises(self) -> None:
        df = pd.DataFrame([{"image_id": 0, "path": "img0.png"}])
        with self.assertRaises(ValueError):
            create_dataset(df)

    def test_missing_path_raises(self):
        df = pd.DataFrame([{"image_id": 0, "identity": "id0"}])
        with self.assertRaises(ValueError):
            create_dataset(df)

    def test_missing_column_with_custom_names_raises(self):
        df = pd.DataFrame([{"image_id": 0}])
        with self.assertRaises(ValueError):
            create_dataset(df, col_label="label", col_path="filepath")


class TestColumnRenames(unittest.TestCase):
    def test_default_aliases_renamed_to_custom_names(self):
        df = pd.DataFrame([make_base_row()])
        dataset = create_dataset(df, col_label="animal", col_path="filepath")
        self.assertIn("animal", dataset.df.columns)
        self.assertIn("filepath", dataset.df.columns)
        self.assertNotIn("identity", dataset.df.columns)
        self.assertNotIn("path", dataset.df.columns)

    def test_rename_raises_when_target_column_already_exists(self):
        df = pd.DataFrame([{**make_base_row(), "filepath": "other.png"}])
        with self.assertRaises(ValueError):
            create_dataset(df, col_path="filepath")


class TestColumnTypes(unittest.TestCase):
    def test_bbox_numeric_list_passes(self):
        df = pd.DataFrame([{**make_base_row(), "bbox": [1, 2, 3, 4]}])
        create_dataset(df)

    def test_bbox_scalar_non_numeric_raises(self):
        df = pd.DataFrame([{**make_base_row(), "bbox": "[1, 2, 3, 4]"}])
        with self.assertRaises(ValueError):
            create_dataset(df)

    def test_bbox_list_of_non_numeric_values_currently_passes(self):
        df = pd.DataFrame([{**make_base_row(), "bbox": ["a", "b", "c", "d"]}])
        create_dataset(df)

    def test_keypoints_numeric_list_passes(self):
        df = pd.DataFrame([{**make_base_row(), "keypoints": [1.0, 2.0, 3.0, 4.0]}])
        create_dataset(df)

    def test_keypoints_scalar_non_numeric_raises(self):
        df = pd.DataFrame([{**make_base_row(), "keypoints": "[1.0, 2.0, 3.0, 4.0]"}])
        with self.assertRaises(ValueError):
            create_dataset(df)

    def test_date_valid_passes(self):
        df = pd.DataFrame([{**make_base_row(), "date": "2020-01-01"}])
        create_dataset(df)

    def test_date_invalid_raises(self):
        df = pd.DataFrame([{**make_base_row(), "date": "banana"}])
        with self.assertRaises(ValueError):
            create_dataset(df)

    def test_species_as_string_passes(self):
        df = pd.DataFrame([{**make_base_row(), "species": "zebra"}])
        create_dataset(df)

    def test_species_as_list_passes(self):
        df = pd.DataFrame([{**make_base_row(), "species": ["zebra", "horse"]}])
        create_dataset(df)

    def test_video_int_passes(self):
        df = pd.DataFrame([{**make_base_row(), "video": 0}])
        create_dataset(df)

    def test_video_non_int_raises(self):
        df = pd.DataFrame([{**make_base_row(), "video": "0"}])
        with self.assertRaises(ValueError):
            create_dataset(df)

    def test_all_null_column_is_skipped_entirely(self):
        df = pd.DataFrame([{**make_base_row(), "bbox": None}])
        create_dataset(df)

    def test_some_null_values_are_ignored_others_checked(self):
        df = pd.DataFrame(
            [
                {**make_base_row(image_id=0, path="img0.png"), "bbox": [1, 2, 3, 4]},
                {**make_base_row(image_id=1, identity="id1", path="img1.png"), "bbox": None},
            ]
        )
        create_dataset(df)


class TestUniqueId(unittest.TestCase):
    def test_unique_ids_pass(self):
        df = pd.DataFrame(
            [
                make_base_row(image_id=0, path="img0.png"),
                make_base_row(image_id=1, identity="id1", path="img1.png"),
            ]
        )
        create_dataset(df)

    def test_duplicate_ids_raise(self):
        df = pd.DataFrame(
            [
                make_base_row(image_id=0, path="img0.png"),
                make_base_row(image_id=0, identity="id1", path="img1.png"),
            ]
        )
        with self.assertRaises(ValueError):
            create_dataset(df)


class TestBadPaths(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, name):
        open(os.path.join(self.root, name), "a").close()

    def test_existing_path_passes(self):
        self._touch("img0.png")
        df = pd.DataFrame([make_base_row(path="img0.png")])
        create_dataset(df, root=self.root, check_files=True)

    def test_missing_path_raises(self):
        df = pd.DataFrame([make_base_row(path="does_not_exist.png")])
        with self.assertRaises(FileNotFoundError):
            create_dataset(df, root=self.root, check_files=True)

    def test_non_iso_8859_1_filename_raises(self):
        emoji_name = "\U0001f41f.png"
        self._touch(emoji_name)
        df = pd.DataFrame([make_base_row(path=emoji_name)])
        with self.assertRaises(ValueError):
            create_dataset(df, root=self.root, check_files=True)

    def test_check_files_false_skips_validation(self):
        df = pd.DataFrame([make_base_row(path="does_not_exist.png")])
        create_dataset(df, root=self.root, check_files=False)

    def test_segmentation_path_column_is_also_checked(self):
        self._touch("img0.png")
        df = pd.DataFrame([{**make_base_row(path="img0.png"), "segmentation": "missing_mask.png"}])
        with self.assertRaises(FileNotFoundError):
            create_dataset(df, root=self.root, check_files=True)


class TestFinalizeCatalogueIntegration(unittest.TestCase):
    def test_update_wrong_labels_true_calls_fix_labels(self):
        calls = []

        class _Tracking(WildlifeDataset):
            def fix_labels(self, df):
                calls.append(True)
                return df

        df = pd.DataFrame([make_base_row()])
        dataset = _Tracking(df=df, root=None, check_files=False, update_wrong_labels=True)
        dataset.df = dataset.finalize_catalogue(dataset.df)
        self.assertEqual(len(calls), 1)

    def test_update_wrong_labels_false_skips_fix_labels(self):
        calls = []

        class _Tracking(WildlifeDataset):
            def fix_labels(self, df):
                calls.append(True)
                return df

        df = pd.DataFrame([make_base_row()])
        dataset = _Tracking(df=df, root=None, check_files=False, update_wrong_labels=False)
        dataset.df = dataset.finalize_catalogue(dataset.df)
        self.assertEqual(len(calls), 0)

    def test_remove_columns_true_drops_constant_column(self):
        df = pd.DataFrame(
            [
                {**make_base_row(image_id=0, path="img0.png"), "video": 0},
                {**make_base_row(image_id=1, identity="id1", path="img1.png"), "video": 0},
            ]
        )
        dataset = create_dataset(df, remove_columns=True)
        self.assertNotIn("video", dataset.df.columns)

    def test_remove_columns_false_keeps_constant_column(self):
        df = pd.DataFrame(
            [
                {**make_base_row(image_id=0, path="img0.png"), "video": 0},
                {**make_base_row(image_id=1, identity="id1", path="img1.png"), "video": 0},
            ]
        )
        dataset = create_dataset(df, remove_columns=False)
        self.assertIn("video", dataset.df.columns)


if __name__ == "__main__":
    unittest.main()
