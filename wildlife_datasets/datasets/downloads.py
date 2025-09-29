import os
import shutil
from datasets import load_dataset
from . import utils

def check_attribute(obj, attr):
    if not hasattr(obj, attr):
        raise Exception(f'Object {obj} must have attribute {attr}.')

class DownloadURL:
    url = None
    archive = None
    downloads = []
    rmtree = None
    extract_add_folder = True

    @classmethod
    def _download(cls):
        if cls.url:
            if cls.archive:
                utils.download_url(cls.url, cls.archive)
            else:
                raise ValueError('When cls.url is specified, cls.archive must also be specified')
        for url, archive in cls.downloads:
            utils.download_url(url, archive)        

    @classmethod
    def _extract(cls, exts = ['.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.rar', '.7z']):
        if cls.archive:
            if any(cls.archive.endswith(ext) for ext in exts):
                utils.extract_archive(cls.archive, delete=True)
        for _, archive in cls.downloads:
            if any(archive.endswith(ext) for ext in exts):
                if cls.extract_add_folder:
                    archive_name = archive.split('.')[0]
                    utils.extract_archive(archive, extract_path=archive_name, delete=True)
                else:
                    utils.extract_archive(archive, delete=True)
        if cls.rmtree:
            shutil.rmtree(cls.rmtree)

class DownloadKaggle:
    @classmethod
    def _download(cls):
        check_attribute(cls, 'kaggle_url')
        check_attribute(cls, 'kaggle_type')
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
        if not os.path.exists(cls.archive_name()):
            raise Exception(exception_text)

    @classmethod
    def _extract(cls):
        display_name = cls.display_name().lower()
        try:
            utils.extract_archive(cls.archive_name(), delete=True)
        except:
            exception_text = f'''Extracting failed.
                Either the download was not completed or the Kaggle terms were not agreed with.
                Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#{display_name}'''
            raise Exception(exception_text)
    
    @classmethod
    def archive_name(cls):
        return cls.kaggle_url.split('/')[-1] + '.zip'
        
class DownloadHuggingFace:
    determined_by_df = False
    saved_to_system_folder = True

    @classmethod
    def _download(cls, *args, **kwargs):
        check_attribute(cls, 'hf_url')
        load_dataset(cls.hf_url, *args, **kwargs)

    @classmethod
    def _extract(cls, **kwargs):
        pass