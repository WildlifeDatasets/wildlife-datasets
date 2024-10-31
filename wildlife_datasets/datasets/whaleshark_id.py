import pandas as pd
from . import utils
from .datasets_wildme import DatasetFactoryWildMe
from .summary import summary

class WhaleSharkID(DatasetFactoryWildMe):
    summary = summary['WhaleSharkID']
    url = 'https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/whaleshark.coco.tar.gz'
    archive = 'whaleshark.coco.tar.gz'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('whaleshark', 2020)
