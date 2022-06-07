import os
import zipfile
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='HappyWhale',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

# Download
os.system(f"kaggle competitions download -c happy-whale-and-dolphin")

# Extract and Clean-up
archive = 'happy-whale-and-dolphin.zip'
with zipfile.ZipFile(archive, 'r') as zip_ref:
    zip_ref.extractall('.')
os.remove(archive)
