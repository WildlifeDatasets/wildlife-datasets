import os
import pandas as pd
from . import utils
from .datasets_wildme import DatasetFactoryWildMe
from .summary import summary

class LeopardID2022(DatasetFactoryWildMe):
    summary = summary['LeopardID2022']
    url = 'https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/leopard.coco.tar.gz'
    archive = 'leopard.coco.tar.gz'

    @classmethod
    def _download(cls):
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        utils.extract_archive(cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('leopard', 2022)