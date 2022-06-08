import os
import argparse
import utils
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='NyalaData',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

# Download and extract
url = 'https://github.com/tvanzyl/wildlife_reidentification/archive/refs/heads/main.zip'
archive = 'main.zip'
utils.download_url(url, archive)
utils.extract_archive(archive, delete=True)

# Cleanup
shutil.rmtree('wildlife_reidentification-main/Lion_Data_Zero')