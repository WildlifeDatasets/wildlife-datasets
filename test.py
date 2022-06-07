from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from torchvision.datasets.utils import  download_and_extract_archive
url = "http://www.gutenberg.lib.md.us/4/8/8/2/48824/48824-8.zip"

def download_and_unzip(url, extract_to='.'):
    print(f'Downloading from: {url}')
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))

    print(f'Extracting')
    zipfile.extractall(path=extract_to)

dataset = {

    'hyena': ,
    }

import os
#download_and_unzip(dataset['hyena'], 'hyena')

name = 'hyena'
url = dataset[name]
root = name


download_and_extract_archive(url, root)

# Delete archive - delete in torchvision function does not work.
archive = os.path.join(root, os.path.basename(url))
os.remove(archive)
