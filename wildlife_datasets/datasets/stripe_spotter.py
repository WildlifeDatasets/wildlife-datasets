import os
import pandas as pd
from . import utils
from .datasets import DatasetFactory
from .summary import summary

class StripeSpotter(DatasetFactory):
    summary = summary['StripeSpotter']

    @classmethod
    def _download(cls):
        downloads = [
            ('https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.zip', 'data-20110718.zip'),
            ('https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.z02', 'data-20110718.z02'),
            ('https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.z01', 'data-20110718.z01'),
        ]
        for url, archive in downloads:
            utils.download_url(url, archive)

    @classmethod
    def _extract(cls):
        exception_text = '''Extracting works only on Linux. Please extract it manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#stripespotter'''
        if os.name == 'posix':
            os.system(f"zip -s- data-20110718.zip -O data-full.zip")
            if not os.path.exists('data-full.zip'):
                raise Exception('Download or extraction failed. Check if zip is installed.')
            os.system(f"unzip data-full.zip")
            os.remove('data-20110718.zip')
            os.remove('data-20110718.z01')
            os.remove('data-20110718.z02')
            os.remove('data-full.zip')
        else:
            raise Exception(exception_text)       

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        data = utils.find_images(self.root)

        # Extract information about the images
        data['index'] = data['file'].str[-7:-4].astype(int)
        data = data[data['file'].str.startswith('img')]
        
        # Load additional information
        data_aux = pd.read_csv(os.path.join(self.root, 'data', 'SightingData.csv'))
        data = pd.merge(data, data_aux, how='left', left_on='index', right_on='#imgindex')
        data.loc[data['animal_name'].isnull(), 'animal_name'] = self.unknown_name
        
        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path':  data['path'] + os.path.sep + data['file'],
            'identity': data['animal_name'],
            'bbox': pd.Series([[int(a) for a in b.split(' ')] for b in data['roi']]),
            'orientation': data['flank'],
            'photo_quality': data['photo_quality'],
            'date': data['sighting_date']
        })
        return self.finalize_catalogue(df)  
