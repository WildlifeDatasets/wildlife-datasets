import unittest
import numpy as np
from .utils import add_datasets, load_datasets
from wildlife_datasets import splits
from wildlife_datasets.datasets import IPanda50, MacaqueFaces

dataset_names = [IPanda50, MacaqueFaces]
n_orig_datasets = len(dataset_names)
seed1 = 666
seed2 = 12345
splitters1_all = [
    splits.ClosedSetSplit(0.5, seed=seed1),
    splits.OpenSetSplit(0.5, 0.1, seed=seed1),
    splits.OpenSetSplit(0.5, n_class_test=5, seed=seed1),
    splits.DisjointSetSplit(0.1, seed=seed1),
    splits.DisjointSetSplit(n_class_test=5, seed=seed1),   
]
splitters1_date = [
    splits.RandomProportion(seed=seed1)
]
splitters2_all = [
    splits.ClosedSetSplit(0.5, seed=seed2),
    splits.OpenSetSplit(0.5, 0.1, seed=seed2),
    splits.OpenSetSplit(0.5, n_class_test=5, seed=seed2),
    splits.DisjointSetSplit(0.1, seed=seed2),
    splits.DisjointSetSplit(n_class_test=5, seed=seed2)
]
splitters2_date = [
    splits.RandomProportion(seed=seed2)
]
tol = 0.1
datasets = load_datasets(dataset_names)
datasets = add_datasets(datasets)
dfs = [dataset.df for dataset in datasets]

class TestIdSplits(unittest.TestCase):
    def test_df(self):
        self.assertEqual(len(dfs), 7)
    
    def test_unknown(self):
        n_unknown = 0
        for df in dfs:
            if sum(df['identity'] == 'unknown'):
                n_unknown += 1
        self.assertEqual(n_unknown, 2)

    def test_date(self):
        n_date = 0
        for df in dfs:
            if 'date' in df.columns:
                n_date += 1
        self.assertEqual(n_date, 4)

    def test_seed(self):
        for splitter in splitters1_all:
            for df in dfs:
                idx_train1, idx_test1 = splitter.split(df)[0]
                idx_train2, idx_test2 = splitter.split(df)[0]
                self.assertEqual(idx_train1.tolist(), idx_train2.tolist())
                self.assertEqual(idx_test1.tolist(), idx_test2.tolist())
        for splitter in splitters1_date:
            for df in dfs:
                if 'date' in df.columns:
                    idx_train1, idx_test1 = splitter.split(df)[0]
                    idx_train2, idx_test2 = splitter.split(df)[0]
                    self.assertEqual(idx_train1.tolist(), idx_train2.tolist())
                    self.assertEqual(idx_test1.tolist(), idx_test2.tolist())
        for splitter1, splitter2 in zip(splitters1_all, splitters2_all):
            for df in dfs:
                idx_train1, idx_test1 = splitter1.split(df)[0]
                idx_train2, idx_test2 = splitter2.split(df)[0]
                self.assertNotEqual(idx_train1.tolist(), idx_train2.tolist())
                self.assertNotEqual(idx_test1.tolist(), idx_test2.tolist())        
        for splitter1, splitter2 in zip(splitters1_date, splitters2_date):
            for df in dfs:
                if 'date' in df.columns:
                    idx_train1, idx_test1 = splitter1.split(df)[0]
                    idx_train2, idx_test2 = splitter2.split(df)[0]
                    self.assertNotEqual(idx_train1.tolist(), idx_train2.tolist())
                    self.assertNotEqual(idx_test1.tolist(), idx_test2.tolist())        
        
            
    def test_closed_set(self):
        ratio_train = 0.5
        splitter = splits.ClosedSetSplit(ratio_train)
        for df in dfs:
            df_red = df[df['identity'] != 'unknown']            
            for idx_train, idx_test in splitter.split(df):
                df_train = df.loc[idx_train]
                df_test = df.loc[idx_test]

                split_type = splits.recognize_id_split(df_train['identity'], df_test['identity'])
                self.assertEqual(split_type, 'closed-set')

                expected_value = ratio_train*len(df_red)
                self.assertAlmostEqual(len(df_train), expected_value, delta=expected_value*tol)

                expected_value = (1-ratio_train)*len(df_red)
                self.assertAlmostEqual(len(df_test), expected_value, delta=expected_value*tol)

    def test_open_set1(self):
        ratio_train = 0.5
        ratio_class_test = 0.1
        splitter = splits.OpenSetSplit(ratio_train, ratio_class_test)
        for df in dfs:
            df_red = df[df['identity'] != 'unknown']
            for idx_train, idx_test in splitter.split(df):
                df_train = df.loc[idx_train]
                df_test = df.loc[idx_test]

                split_type = splits.recognize_id_split(df_train['identity'], df_test['identity'])
                self.assertEqual(split_type, 'open-set')

                expected_value = ratio_train*len(df_red)
                self.assertAlmostEqual(len(df_train), expected_value, delta=expected_value*tol)
                
                ids_train = set(df_train['identity'])
                ids_test = set(df_test['identity'])
                
                n_test_only = sum(sum(df_test['identity'] == id) for id in ids_test - ids_train)
                expected_value = ratio_class_test*len(df_red)
                self.assertAlmostEqual(n_test_only, expected_value, delta=3*expected_value*tol)
            
    def test_open_set2(self):
        ratio_train = 0.5
        n_class_test = 5
        splitter = splits.OpenSetSplit(ratio_train, n_class_test=n_class_test)
        for df in dfs:
            df_red = df[df['identity'] != 'unknown']
            for idx_train, idx_test in splitter.split(df):
                df_train = df.loc[idx_train]
                df_test = df.loc[idx_test]

                split_type = splits.recognize_id_split(df_train['identity'], df_test['identity'])
                self.assertEqual(split_type, 'open-set')

                expected_value = ratio_train*len(df_red)
                self.assertAlmostEqual(len(df_train), expected_value, delta=expected_value*tol)
                
                ids_train = set(df_train['identity'])
                ids_test = set(df_test['identity'])
                
                self.assertEqual(len(ids_test-ids_train), n_class_test)

    def test_disjoint_set1(self):
        ratio_class_test = 0.1
        splitter = splits.DisjointSetSplit(ratio_class_test)
        for df in dfs:
            df_red = df[df['identity'] != 'unknown']
            for idx_train, idx_test in splitter.split(df):
                df_train = df.loc[idx_train]
                df_test = df.loc[idx_test]

                split_type = splits.recognize_id_split(df_train['identity'], df_test['identity'])
                self.assertEqual(split_type, 'disjoint-set')

                ids_test = set(df_test['identity'])
                
                n_test_only = sum(sum(df_test['identity'] == id) for id in ids_test)
                expected_value = ratio_class_test*len(df_red)
                self.assertAlmostEqual(n_test_only, expected_value, delta=3*expected_value*tol)

    def test_disjoint_set2(self):
        n_class_test = 5
        splitter = splits.DisjointSetSplit(n_class_test=n_class_test)
        for df in dfs:
            for idx_train, idx_test in splitter.split(df):
                df_train = df.loc[idx_train]
                df_test = df.loc[idx_test]

                split_type = splits.recognize_id_split(df_train['identity'], df_test['identity'])
                self.assertEqual(split_type, 'disjoint-set')

                ids_test = set(df_test['identity'])
                
                self.assertEqual(len(ids_test), n_class_test)

    def test_resplit_features(self):
        n_features = 5
        for splitter in splitters1_all:
            for df in dfs:
                features = np.random.randn(len(df), n_features)
                for idx_train1, idx_test1 in splitter.split(df):
                    idx_train2, idx_test2 = splitter.resplit_by_features(df, features, idx_train1)
                    
                    idx1 = list(idx_train1) + list(idx_test1)
                    idx2 = list(idx_train2) + list(idx_test2)
                    df_train1 = df.loc[idx_train1]
                    df_test1 = df.loc[idx_test1]
                    df_train2 = df.loc[idx_train2]
                    df_test2 = df.loc[idx_test2]

                    self.assertEqual(np.sort(idx1).tolist(), np.sort(idx2).tolist())
                    self.assertEqual(set(df_train1['identity']), set(df_train2['identity']))
                    self.assertEqual(set(df_test1['identity']), set(df_test2['identity']))
                    for id in set(df_train1['identity']):
                        self.assertEqual(len(df_train1['identity']==id), len(df_train2['identity']==id))
                    for id in set(df_test1['identity']):
                        self.assertEqual(len(df_test1['identity']==id), len(df_test2['identity']==id))

if __name__ == '__main__':
    unittest.main()

