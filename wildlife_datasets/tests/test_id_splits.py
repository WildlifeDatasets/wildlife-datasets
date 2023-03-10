import unittest
from .utils import load_datasets
from wildlife_datasets import datasets, splits

dataset_names = [
    datasets.IPanda50,
    datasets.MacaqueFaces,
]

tol = 0.1
dfs = load_datasets(dataset_names)

class TestIdSplits(unittest.TestCase):
    def test_df(self):
        self.assertGreaterEqual(len(dfs), 1)
    
    def test_closed_set(self):
        ratio_train = 0.5
        for df in dfs:
            splitter = splits.ClosedSetSplit(df)
            idx_train, idx_test = splitter.split(ratio_train)
            df_train = df.loc[idx_train]
            df_test = df.loc[idx_test]

            split_type = splits.recognize_id_split(df_train['identity'], df_test['identity'])
            self.assertEqual(split_type, 'closed-set')

            expected_value = ratio_train*len(df)
            self.assertAlmostEqual(len(df_train), expected_value, delta=expected_value*tol)

    def test_open_set1(self):
        ratio_train = 0.5
        ratio_class_test = 0.1
        for df in dfs:
            splitter = splits.OpenSetSplit(df)
            idx_train, idx_test = splitter.split(ratio_train, ratio_class_test)
            df_train = df.loc[idx_train]
            df_test = df.loc[idx_test]

            split_type = splits.recognize_id_split(df_train['identity'], df_test['identity'])
            self.assertEqual(split_type, 'open-set')

            expected_value = ratio_train*len(df)
            self.assertAlmostEqual(len(df_train), expected_value, delta=expected_value*tol)
            
            ids_train = set(df_train['identity'])
            ids_test = set(df_test['identity'])
            
            n_test_only = sum(sum(df_test['identity'] == id) for id in ids_test - ids_train)
            expected_value = ratio_class_test*len(df)
            self.assertAlmostEqual(n_test_only, expected_value, delta=3*expected_value*tol)
            
    def test_open_set2(self):
        ratio_train = 0.5
        n_class_test = 5
        for df in dfs:
            splitter = splits.OpenSetSplit(df)
            idx_train, idx_test = splitter.split(ratio_train, n_class_test=n_class_test)
            df_train = df.loc[idx_train]
            df_test = df.loc[idx_test]

            split_type = splits.recognize_id_split(df_train['identity'], df_test['identity'])
            self.assertEqual(split_type, 'open-set')

            expected_value = ratio_train*len(df)
            self.assertAlmostEqual(len(df_train), expected_value, delta=expected_value*tol)
            
            ids_train = set(df_train['identity'])
            ids_test = set(df_test['identity'])
            
            self.assertEqual(len(ids_test-ids_train), n_class_test)

    def test_disjoint_set1(self):
        ratio_class_test = 0.1
        for df in dfs:
            splitter = splits.DisjointSetSplit(df)
            idx_train, idx_test = splitter.split(ratio_class_test)
            df_train = df.loc[idx_train]
            df_test = df.loc[idx_test]

            split_type = splits.recognize_id_split(df_train['identity'], df_test['identity'])
            self.assertEqual(split_type, 'disjoint-set')

            ids_test = set(df_test['identity'])
            
            n_test_only = sum(sum(df_test['identity'] == id) for id in ids_test)
            expected_value = ratio_class_test*len(df)
            self.assertAlmostEqual(n_test_only, expected_value, delta=3*expected_value*tol)

    def test_disjoint_set2(self):
        n_class_test = 5
        for df in dfs:
            splitter = splits.DisjointSetSplit(df)
            idx_train, idx_test = splitter.split(n_class_test=n_class_test)
            df_train = df.loc[idx_train]
            df_test = df.loc[idx_test]

            split_type = splits.recognize_id_split(df_train['identity'], df_test['identity'])
            self.assertEqual(split_type, 'disjoint-set')

            ids_test = set(df_test['identity'])
            
            self.assertEqual(len(ids_test), n_class_test)


if __name__ == '__main__':
    unittest.main()
