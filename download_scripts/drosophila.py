import os
import zipfile
import argparse
from utils import download_url

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='Drosophila',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)


downloads = [
    ('https://dataverse.scholarsportal.info/api/access/datafile/71066', 'week1_Day1_train_01to05.zip'),
]

for url, archive in downloads:
    download_url(url, archive)
    with zipfile.ZipFile(archive, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(archive)