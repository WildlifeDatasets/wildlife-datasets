import os
import urllib.request
from tqdm import tqdm
import shutil
from contextlib import contextmanager

class ProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path='.'):
    with ProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_archive(archive, extract_path='.', delete=False):
    shutil.unpack_archive(archive, extract_path)
    if delete:
        os.remove(archive)


@contextmanager
def data_directory(dir):
    '''
    Changes context such that data directory is used as current work directory.
    Data directory is created if it does not exist.
    '''
    current_dir = os.getcwd()
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(current_dir)


def kaggle_download(command, exception_text='', required_file=None):
    # TODO: add kaggle as package requirements    
    try:
        os.system(f"kaggle {command}")
    except:
        raise Exception(exception_text)
    if required_file is not None and not os.path.exists(required_file):
        raise Exception(exception_text)
