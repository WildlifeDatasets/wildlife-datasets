import os
import argparse
from . import utils

def get_data(root):
    with utils.data_directory(root):
        url = 'https://data.bris.ac.uk/datasets/tar/jf0859kboy8k2ufv60dqeb2t8.zip'
        archive = 'jf0859kboy8k2ufv60dqeb2t8.zip'
        utils.download_url(url, archive)
        utils.extract_archive(archive, delete=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='BristolGorillas2020',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))