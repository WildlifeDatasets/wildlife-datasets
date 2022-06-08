import os
import urllib.request
from tqdm import tqdm
import shutil

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


import zipfile
def extract_archive2(archive, extract_path='.', delete=False):
    with zipfile.ZipFile(archive, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    if delete:
        os.remove(archive)