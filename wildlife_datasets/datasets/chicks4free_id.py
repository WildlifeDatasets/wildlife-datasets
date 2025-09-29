import numpy as np
import pandas as pd
from .datasets import DatasetFactory
from .downloads import DownloadHuggingFace
from datasets import load_dataset

summary = {
    'licenses': 'Attribution 4.0 International (CC BY 4.0)',
    'licenses_url': 'https://creativecommons.org/licenses/by/4.0/',
    'url': 'https://huggingface.co/datasets/dariakern/Chicks4FreeID',
    'publication_url': None,
    'cite': 'kern2024towards',
    'animals': {'chickens'},
    'animals_simple': 'chickens',
    'real_animals': True,
    'year': 2024,
    'reported_n_total': 1146,
    'reported_n_individuals': 50,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': False,
    'from_video': False,
    'cropped': True,
    'span': 'short',
    'size': 1401,
}

class Chicks4FreeID(DownloadHuggingFace, DatasetFactory):
    summary = summary
    hf_url = 'dariakern/Chicks4FreeID'

    @classmethod
    def _download(cls, hf_option = 'chicken-re-id-all-visibility'):
        super()._download(hf_option)

    def create_catalogue(self, hf_option = 'chicken-re-id-all-visibility') -> pd.DataFrame:
        dataset = load_dataset(self.hf_url, hf_option)
        
        self.n_train = dataset['train'].num_rows
        self.n_test = dataset['test'].num_rows
        self.dataset = dataset
        return pd.DataFrame({
            'image_id': range(self.n_train+self.n_test),
            'identity': dataset['train']['identity'] + dataset['test']['identity'],
            'path': np.nan,
            'split_original': self.n_train*['train'] + self.n_test*['test']
        })

    def get_image(self, idx):
        if idx < self.n_train:
            return self.dataset['train'][int(idx)]['crop']
        else:
            return self.dataset['test'][int(idx)-self.n_train]['crop']
