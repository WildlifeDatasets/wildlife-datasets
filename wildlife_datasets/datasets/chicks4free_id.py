import numpy as np
import pandas as pd
from .datasets import DatasetFactory
from .summary import summary
from datasets import load_dataset

class Chicks4FreeID(DatasetFactory):
    summary = summary['Chicks4FreeID']
    hf_url = 'dariakern/Chicks4FreeID'

    @classmethod
    def _download(cls):
        load_dataset(cls.hf_url)

    @classmethod
    def _extract(cls):
        pass

    def create_catalogue(self) -> pd.DataFrame:
        dataset = load_dataset(self.hf_url)
        
        self.n_train = dataset['train'].num_rows
        self.n_test = dataset['test'].num_rows
        self.dataset = dataset
        return pd.DataFrame({
            'identity': dataset['train']['identity'] + dataset['test']['identity'],
            'path': np.nan,
            'split_original': self.n_train*['train'] + self.n_test*['test']
        })

    def get_image(self, idx):
        if idx < self.n_train:
            return self.dataset['train'][int(idx)]['crop']
        else:
            return self.dataset['test'][int(idx)-self.n_train]['crop']
