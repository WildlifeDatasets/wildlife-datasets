import unittest
from wildlife_datasets import datasets

class TestDatasets(unittest.TestCase):
    def test_display_names(self):
        for dataset_class in datasets.names_all:
            dataset_class.display_name()
    