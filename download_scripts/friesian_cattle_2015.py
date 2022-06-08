import os
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='FriesianCattle2015',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

# Download and extract
url = 'https://data.bris.ac.uk/datasets/wurzq71kfm561ljahbwjhx9n3/wurzq71kfm561ljahbwjhx9n3.zip'
archive = 'wurzq71kfm561ljahbwjhx9n3.zip'
utils.download_url(url, archive)
utils.extract_archive(archive, delete=True)
