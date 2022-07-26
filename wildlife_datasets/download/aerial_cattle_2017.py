import os
import argparse
from . import utils

def get_dataset(output='data', name='AerialCattle2017'):
    directory = os.path.join(output, name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)

    # Download and extract
    url = 'https://data.bris.ac.uk/datasets/tar/3owflku95bxsx24643cybxu3qh.zip'
    archive = '3owflku95bxsx24643cybxu3qh.zip'
    utils.download_url(url, archive)
    utils.extract_archive(archive, delete=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='AerialCattle2017',  help="Dataset name")
    args = parser.parse_args()
    get_dataset(args.output, args.name)
