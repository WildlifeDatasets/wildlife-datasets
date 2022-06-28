import os
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='BristolGorillas2020',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

# Download and extract
url = 'https://data.bris.ac.uk/datasets/tar/jf0859kboy8k2ufv60dqeb2t8.zip'
archive = 'jf0859kboy8k2ufv60dqeb2t8.zip'
utils.download_url(url, archive)
utils.extract_archive(archive, delete=True)


