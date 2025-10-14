import unittest
from wildlife_datasets.datasets import names_all, IPanda50, MacaqueFaces
from .utils import load_datasets, add_datasets

dataset_names = [IPanda50, MacaqueFaces]
datasets = load_datasets(dataset_names)

class TestDatasets(unittest.TestCase):
    def test_display_names(self):
        for dataset_class in names_all:
            dataset_class.display_name()
    
    def test_get_subset(self):
        n_new = 10
        for dataset_old in datasets:
            n_old = len(dataset_old)
            dataset_new = dataset_old.get_subset(range(n_new))
            for (dataset, n) in zip([dataset_old, dataset_new], [n_old, n_new]):
                self.assertEqual(n, len(dataset))
                self.assertEqual(n, len(dataset.labels))
                self.assertEqual(dataset.df['identity'].nunique(), len(dataset.labels_map))
                ids1 = dataset.df['identity'].to_numpy()
                ids2 = dataset.labels_map[dataset.labels]
                self.assertEqual(tuple(ids1), tuple(ids2))
