import os
from . import utils

def check_attribute(obj, attr):
    if not hasattr(obj, attr):
        raise Exception(f'Object {obj} must have attribute {attr}.')

class DownloadURL:
    @classmethod
    def _download(cls):
        check_attribute(cls, 'url')
        check_attribute(cls, 'archive')
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        check_attribute(cls, 'archive')
        utils.extract_archive(cls.archive, delete=True)

class DownloadKaggle:
    @classmethod
    def _download(cls):
        check_attribute(cls, 'kaggle_url')
        check_attribute(cls, 'kaggle_type')
        check_attribute(cls, 'archive')
        check_attribute(cls, 'display_name')
        display_name = cls.display_name().lower()
        if cls.kaggle_type == 'datasets':
            command = f'datasets download -d {cls.kaggle_url} --force'
        elif cls.kaggle_type == 'competitions':
            command = f'competitions download -c {cls.kaggle_url} --force'
        else:
            raise ValueError(f'cls.kaggle_type must be datasets or competitions.')
        exception_text = f'''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#{display_name}'''
        try:
            os.system(f"kaggle {command}")
        except:
            raise Exception(exception_text)
        if not os.path.exists(cls.archive):
            raise Exception(exception_text)

    @classmethod
    def _extract(cls):
        check_attribute(cls, 'archive')
        display_name = cls.display_name().lower()
        try:
            utils.extract_archive(cls.archive, delete=True)
        except:
            exception_text = f'''Extracting failed.
                Either the download was not completed or the Kaggle terms were not agreed with.
                Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#{display_name}'''
            raise Exception(exception_text)