import unittest

import pandas as pd

from .utils import create_dataset


def make_df(identities):
    n = len(identities)
    return pd.DataFrame(
        {
            "image_id": list(range(n)),
            "identity": identities,
            "path": [f"img{i}.png" for i in range(n)],
        }
    )


class TestFixLabelsReplaceIdentity(unittest.TestCase):
    def test_basic_replace(self):
        dataset = create_dataset(make_df(["a", "b", "a"]))
        result = dataset.fix_labels_replace_identity(dataset.df, [("a", "x")])
        self.assertEqual(list(result["identity"]), ["x", "b", "x"])

    def test_chained_resolution(self):
        dataset = create_dataset(make_df(["a", "b"]))
        result = dataset.fix_labels_replace_identity(dataset.df, [("a", "b"), ("b", "c")])
        self.assertEqual(list(result["identity"]), ["c", "c"])

    def test_cycle_raises(self):
        dataset = create_dataset(make_df(["a", "b"]))
        with self.assertRaises(ValueError):
            dataset.fix_labels_replace_identity(dataset.df, [("a", "b"), ("b", "a")])

    def test_duplicate_source_raises(self):
        dataset = create_dataset(make_df(["a", "b"]))
        with self.assertRaises(ValueError):
            dataset.fix_labels_replace_identity(dataset.df, [("a", "b"), ("a", "c")])

    def test_custom_column(self):
        df = pd.DataFrame(
            {
                "image_id": [0, 1],
                "identity": ["x", "y"],
                "path": ["i0.png", "i1.png"],
                "species": ["a", "b"],
            }
        )
        dataset = create_dataset(df)
        result = dataset.fix_labels_replace_identity(dataset.df, [("a", "z")], col="species")
        self.assertEqual(list(result["species"]), ["z", "b"])

    def test_replacing_absent_identity_is_noop(self):
        dataset = create_dataset(make_df(["a", "b"]))
        result = dataset.fix_labels_replace_identity(dataset.df, [("does_not_exist", "z")])
        self.assertEqual(list(result["identity"]), ["a", "b"])


class TestFixLabelsGroupIdentity(unittest.TestCase):
    def test_merges_connected_group(self):
        dataset = create_dataset(make_df([1, 2, 3, 4]))
        result = dataset.fix_labels_group_identity(dataset.df, [(1, 2, 3), (3, 4)])
        self.assertEqual(list(result["identity"]), [1, 1, 1, 1])

    def test_independent_groups_stay_separate(self):
        dataset = create_dataset(make_df([1, 2, 3, 7, 8]))
        result = dataset.fix_labels_group_identity(dataset.df, [(1, 2, 3), (7, 8)])
        self.assertEqual(list(result["identity"]), [1, 1, 1, 7, 7])

    def test_representative_not_in_df_raises(self):
        dataset = create_dataset(make_df([2, 3]))
        with self.assertRaises(ValueError):
            dataset.fix_labels_group_identity(dataset.df, [(1, 2, 3)])

    def test_empty_group_is_skipped(self):
        dataset = create_dataset(make_df([1, 2]))
        result = dataset.fix_labels_group_identity(dataset.df, [(), (1, 2)])
        self.assertEqual(list(result["identity"]), [1, 1])

    def test_single_element_group_is_noop(self):
        dataset = create_dataset(make_df([5, 6]))
        result = dataset.fix_labels_group_identity(dataset.df, [(5,)])
        self.assertEqual(list(result["identity"]), [5, 6])


class TestFixLabelsRemoveIdentity(unittest.TestCase):
    def test_removes_matching_identities(self):
        dataset = create_dataset(make_df(["a", "b", "a", "c"]))
        result = dataset.fix_labels_remove_identity(dataset.df, ["a"])
        self.assertEqual(list(result["identity"]), ["b", "c"])

    def test_removing_absent_identity_is_noop(self):
        dataset = create_dataset(make_df(["a", "b"]))
        result = dataset.fix_labels_remove_identity(dataset.df, ["does_not_exist"])
        self.assertEqual(list(result["identity"]), ["a", "b"])

    def test_removing_everything_yields_empty(self):
        dataset = create_dataset(make_df(["a", "b"]))
        result = dataset.fix_labels_remove_identity(dataset.df, ["a", "b"])
        self.assertEqual(len(result), 0)


class TestFixLabelsReplaceImages(unittest.TestCase):
    def make_dataset(self, paths, identities):
        df = pd.DataFrame(
            {
                "image_id": list(range(len(paths))),
                "identity": identities,
                "path": paths,
            }
        )
        return create_dataset(df)

    def test_basic_replace(self):
        dataset = self.make_dataset(["a/img1.png", "a/img2.png"], ["x", "x"])
        result = dataset.fix_labels_replace_images(dataset.df, [("img1", "x", "y")])
        self.assertEqual(list(result["identity"]), ["y", "x"])

    def test_no_match_is_noop(self):
        dataset = self.make_dataset(["a/img1.png"], ["x"])
        result = dataset.fix_labels_replace_images(dataset.df, [("does_not_exist", "x", "y")])
        self.assertEqual(list(result["identity"]), ["x"])

    def test_substring_matches_multiple_files(self):
        dataset = self.make_dataset(["a/img1.png", "a/img10.png"], ["x", "x"])
        result = dataset.fix_labels_replace_images(dataset.df, [("img1", "x", "y")])
        self.assertEqual(list(result["identity"]), ["y", "y"])

    def test_multiple_matches_still_replaces_all(self):
        dataset = self.make_dataset(["a/img1.png", "b/img1.png"], ["x", "x"])
        result = dataset.fix_labels_replace_images(dataset.df, [("img1", "x", "y")])
        self.assertEqual(list(result["identity"]), ["y", "y"])


if __name__ == "__main__":
    unittest.main()
