import os
import pandas as pd
from . import utils
from .datasets import WildlifeDataset
from .summary import summary

class AnimalCLEF2025(WildlifeDataset):    
    # TODO: add summary
    # TODO: add download
    #summary = summary['AnimalCLEF2025']
    #archive = 'amvrakikosturtles.zip'

    @classmethod
    def _download(cls):
        raise NotImplementedError
        command = f"datasets download -d wildlifedatasets/amvrakikosturtles --force"
        exception_text = '''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#amvrakikosturtles'''
        utils.kaggle_download(command, exception_text=exception_text, required_file=cls.archive)

    @classmethod
    def _extract(cls):
        raise NotImplementedError
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self, split='train') -> pd.DataFrame:
        if split in ['train', 'test']:
            file_name = split + '.csv'            
        else:
            raise Exception('Split not known. Choose train or test.')
        metadata = pd.read_csv(os.path.join(self.root, file_name), low_memory=False)        
        metadata['image_id'] = range(len(metadata))
        return self.finalize_catalogue(metadata)
