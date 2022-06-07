import os
import zipfile
import argparse
from utils import download_url

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='Test',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

# Download
download_url("http://www.gutenberg.lib.md.us/4/8/8/2/48824/48824-8.zip", '48824-8.zip')

# Extract
archive = '48824-8.zip'
with zipfile.ZipFile(archive, 'r') as zip_ref:
    zip_ref.extractall('.')
os.remove(archive)