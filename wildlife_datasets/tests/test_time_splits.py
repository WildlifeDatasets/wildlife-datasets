import unittest
import numpy as np
import pandas as pd
from .utils import load_datasets, add_datasets
from wildlife_datasets import splits
from wildlife_datasets.datasets import IPanda50, MacaqueFaces

dataset_names = [IPanda50, MacaqueFaces]

tol = 0.1
datasets = load_datasets(dataset_names)
datasets = add_datasets(datasets)
dfs = [dataset.df for dataset in datasets]

class TestTimeSplits(unittest.TestCase):
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
        splitter = splits.TimeProportionSplit()
        for df in dfs:
            if 'date' in df.columns:
                idx_train, idx_test = splitter.split(df)[0]
                idx_train1, idx_test1 = splitter.resplit_random(df, idx_train, idx_test)
                idx_train2, idx_test2 = splitter.resplit_random(df, idx_train, idx_test)
                self.assertEqual(idx_train1.tolist(), idx_train2.tolist())
                self.assertEqual(idx_test1.tolist(), idx_test2.tolist())

    def test_time_proportion(self):
        splitter = splits.TimeProportionSplit()
        for df in dfs:
            if 'date' not in df.columns:
                splitter = splits.TimeProportionSplit()
                self.assertRaises(Exception, splitter.split, df)
            else:
                for idx_train, idx_test in splitter.split(df):
                    df_train = df.loc[idx_train]
                    df_test = df.loc[idx_test]

                    split_type = splits.recognize_time_split(df_train, df_test)
                    self.assertEqual(split_type, 'time-proportion')

    def test_time_cutoff(self):
        for df in dfs:
            if 'date' not in df.columns:
                splitter = splits.TimeCutoffSplit(0)
                self.assertRaises(Exception, splitter.split, df)
            else:
                years = pd.to_datetime(df['date']).apply(lambda x: x.year)
                splitter = splits.TimeCutoffSplit(max(years))
                for idx_train, idx_test in splitter.split(df):
                    df_train = df.loc[idx_train]
                    df_test = df.loc[idx_test]

                    split_type = splits.recognize_time_split(df_train, df_test)
                    self.assertEqual(split_type, 'time-cutoff')

    def test_time_cutoff_all(self):
        for df in dfs:
            if 'date' not in df.columns:
                splitter = splits.TimeCutoffSplitAll(0)
                self.assertRaises(Exception, splitter.split, df)
            else:
                years = pd.to_datetime(df['date']).apply(lambda x: x.year)
                splitter = splits.TimeCutoffSplitAll()
                for idx_train, idx_test in splitter.split(df):
                    df_train = df.loc[idx_train]
                    df_test = df.loc[idx_test]

                    split_type = splits.recognize_time_split(df_train, df_test)
                    self.assertEqual(split_type, 'time-cutoff')
                    
    def test_resplit_random(self):
        for df in dfs:
            if 'date' not in df.columns:
                splitter = splits.TimeProportionSplit()
                self.assertRaises(Exception, splitter.split, df)
            else:
                splitter = splits.TimeProportionSplit()
                for idx_train1, idx_test1 in splitter.split(df):
                    idx_train2, idx_test2 = splitter.resplit_random(df, idx_train1, idx_test1)
                    
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

    def test_resplit_features(self):
        n_features = 5
        for df in dfs:
            if 'date' in df.columns:
                years = pd.to_datetime(df['date']).apply(lambda x: x.year)
                splitters = [
                    splits.TimeProportionSplit(),
                    splits.TimeCutoffSplit(max(years))
                ]                        
                for splitter in splitters:
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

